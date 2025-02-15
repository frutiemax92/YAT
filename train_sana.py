import argparse
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from diffusers import SanaTransformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers import SanaPipeline, SanaPAGPipeline
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from common.training_parameters_reader import TrainingParameters
from common.trainer import Trainer

class SanaTrainer(Trainer):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)
        self.pipe = SanaPAGPipeline.from_pretrained(params.pretrained_pipe_path, pag_applied_layers="transformer_blocks.8")
        if params.pretrained_model_path != None:
            transformer = SanaTransformer2DModel.from_pretrained(params.pretrained_model_path) 
            self.pipe.transformer = transformer
        
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(params.pretrained_pipe_path, subfolder='scheduler')
        self.pipe.vae.train(False)
        self.pipe.text_encoder.train(False)

        vae_compression = 32
        resolution = self.pipe.transformer.config.sample_size * vae_compression
        if resolution == 256:
            self.aspect_ratios = ASPECT_RATIO_256_BIN
        elif resolution == 512:
            self.aspect_ratios = ASPECT_RATIO_512_BIN
        elif resolution == 1024:
            self.aspect_ratios = ASPECT_RATIO_1024_BIN
        else:
            self.aspect_ratios = ASPECT_RATIO_2048_BIN

        if params.bfloat16:
            self.pipe = self.pipe.to(torch.bfloat16)
        self.model = self.pipe.transformer
        self.model.enable_gradient_checkpointing()
    
    def initialize(self):
        super().initialize()
        self.pipe = self.pipe.to(self.accelerator.device)
        self.pipe.transformer = self.model
        text_encoder = self.pipe.text_encoder
        vae = self.pipe.vae
        transformer = self.pipe.transformer
        
        transformer = transformer.to(self.accelerator.device)
        vae = vae.to(self.accelerator.device)
        text_encoder = text_encoder.to(self.accelerator.device)
        if self.params.low_vram:
            if self.accelerator.is_main_process:
                vae = vae.to(device=self.accelerator.device)
                text_encoder = text_encoder.to(device=self.accelerator.device)
                transformer = transformer.cpu()
    
    def extract_latents(self, images):
        image_processor = self.pipe.image_processor
        images = image_processor.preprocess(images)
        vae = self.pipe.vae

        # move vae to cuda if it's not already done
        vae = vae.to(device=self.accelerator.device)

        if self.params.low_vram == False or self.params.batch_size >= 8:
            output = self.pipe.vae.encode(images.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)).latent
        else:
            output = []
            for image in images:
                image = image.unsqueeze(0)
                latent = self.pipe.vae.encode(image.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)).latent
                output.append(latent)
                
            output = torch.stack(output)
            output = torch.squeeze(output, dim=1)
        return output * self.pipe.vae.config.scaling_factor

    def extract_embeddings(self, captions):
        # move text_encoder to cuda if not already done
        self.pipe.text_encoder = self.pipe.text_encoder.to(device=self.accelerator.device)
        
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
        self.pipe.encode_prompt(captions, do_classifier_free_guidance=False, device=self.accelerator.device)
        return prompt_embeds, prompt_attention_mask
    
    def validate(self):
        params = self.params
        vae = self.pipe.vae
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder
        transformer = self.pipe.transformer
        scheduler = self.pipe.scheduler
        if params.low_vram:
            vae = vae.cpu()
            text_encoder = self.pipe.text_encoder
            text_encoder = text_encoder.cpu()
            self.pipe.vae = None
            torch.cuda.empty_cache()

        # convert to float16 as inference with bfloat16 is unstable
        self.pipe.to(device=self.accelerator.device, dtype=torch.float16)

        pil_to_tensor = PILToTensor()
        idx = 0
        generator=torch.Generator(device="cuda").manual_seed(42)
        latents = []
        for prompt in tqdm.tqdm(params.validation_prompts, desc='Generating validation latents'):
            if params.low_vram:
                text_encoder = text_encoder.to(device=self.accelerator.device)
                self.pipe.text_encoder = text_encoder
                prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
                    self.pipe.encode_prompt(prompt)
                text_encoder = text_encoder.cpu()
                self.pipe.text_encoder = None
                self.pipe = self.pipe.to(self.accelerator.device)
                torch.cuda.empty_cache()

            latent = self.pipe(
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                pag_scale=2.0,
                guidance_scale=5.0,
                num_inference_steps=20,
                generator=generator,
                output_type='latent'
            )[0]
            latents.append(latent)

        self.pipe.vae = vae
        self.pipe.text_encoder = text_encoder
        if self.params.low_vram:
            transformer = transformer.cpu()
            vae = vae.to(self.accelerator.device)
        
        vae = self.pipe.vae
        idx = 0
        for latent in tqdm.tqdm(latents, desc='Decoding validation latents'):
            prompt = params.validation_prompts[idx]
            latent = latent.to(vae.dtype)
            image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
            image = self.pipe.image_processor.postprocess(image)
            self.logger.add_image(f'validation/{idx}/{prompt}', pil_to_tensor(image[0]), self.global_step)
            idx = idx + 1
        
        if self.params.bfloat16:
            self.pipe.to(device=self.accelerator.device, dtype=torch.bfloat16)
        else:
            self.pipe.to(device=self.accelerator.device, dtype=torch.float32)
    
    def optimize(self, model, batch):
        self.pipe.vae = self.pipe.vae.cpu()
        self.pipe.text_encoder = self.pipe.text_encoder.cpu()
        transformer = self.pipe.transformer.to(self.accelerator.device)
        
        params = self.params
        batch_size = params.batch_size
        latents, embeddings, attention_mask = batch

        loss_fn = torch.nn.MSELoss()
        noise = randn_tensor(latents.shape, device=self.accelerator.device, dtype=self.pipe.dtype)

        u = compute_density_for_timestep_sampling('logit_normal', batch_size, logit_mean=0, logit_std=1.0, mode_scale=1.29)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(self.accelerator.device)
        noisy_model_input = self.scheduler.scale_noise(latents.to(self.accelerator.device), timesteps, noise)

        transformer = model
        noise_pred = transformer(noisy_model_input.to(dtype=self.pipe.vae.dtype),
                                encoder_hidden_states=embeddings.to(dtype=self.pipe.vae.dtype),
                                timestep=timesteps.to(dtype=self.pipe.vae.dtype),
                                encoder_attention_mask=attention_mask.to(dtype=self.pipe.vae.dtype)).sample
        target = noise - latents
        loss = loss_fn(noise_pred.to(dtype=noise.dtype), target)
        return loss
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)

    trainer = SanaTrainer(params)
    trainer.run()