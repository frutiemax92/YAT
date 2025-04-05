import argparse
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from diffusers import SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter
from diffusers import StableDiffusion3Pipeline
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from common.training_parameters_reader import TrainingParameters
from common.trainer import Trainer
from diffusers import BitsAndBytesConfig
from transformers import T5EncoderModel
from utils.compress_caption import compress_caption
import gc

class SD35Trainer(Trainer):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)
        
        if params.bfloat16:
            if self.params.low_vram:
                # put the T5 model as 8 bits
                config_8bit = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                text_encoder_3 = T5EncoderModel.from_pretrained(params.pretrained_pipe_path,
                                                                subfolder='text_encoder_3',
                                                                torch_dtype=torch.bfloat16)
                transformer = SD3Transformer2DModel.from_pretrained(params.pretrained_pipe_path,
                                                                    subfolder='transformer',
                                                                    torch_dtype=torch.bfloat16)
                self.pipe = StableDiffusion3Pipeline.from_pretrained(params.pretrained_pipe_path,
                                                                     text_encoder_3=text_encoder_3,
                                                                     transformer=transformer,
                                                                     torch_dtype=torch.bfloat16)
            else:
                self.pipe = StableDiffusion3Pipeline.from_pretrained(params.pretrained_pipe_path, torch_dtype=torch.bfloat16)
        else:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(params.pretrained_pipe_path)
        if params.pretrained_model_path != None:
            transformer = SD3Transformer2DModel.from_pretrained(params.pretrained_model_path)
            self.pipe.transformer = transformer
        #self.pipe.enable_model_cpu_offload()

        # required for lower vram consumption
        self.pipe.transformer.enable_gradient_checkpointing()
        
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(params.pretrained_pipe_path, subfolder='scheduler')
        self.pipe.vae.train(False)
        self.pipe.text_encoder.train(False)
        self.pipe.text_encoder_2.train(False)
        self.pipe.text_encoder_3.train(False)

        self.aspect_ratios = ASPECT_RATIO_1024_BIN
        if params.bfloat16:
            self.pipe = self.pipe.to(torch.bfloat16)
        self.model = self.pipe.transformer
    
    def initialize(self):
        super().initialize()
    
    def extract_latents(self, images):
        self.pipe.vae.to(self.accelerator.device)
        #self.pipe.text_encoder_3.cpu()

        image_processor = self.pipe.image_processor
        images = image_processor.preprocess(images)

        gc.collect()
        torch.cuda.empty_cache()
        output = self.pipe.vae.encode(images).latent_dist.sample()
        output = (output - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor

        #if self.params.low_vram:
            #vae = vae.cpu()
        return output

    def extract_embeddings(self, caption):
        #self.pipe.text_encoder.to(self.accelerator.device)
        self.pipe.text_encoder.to(self.accelerator.device)
        self.pipe.text_encoder_2.to(self.accelerator.device)
        self.pipe.text_encoder_3.to(self.accelerator.device)
        
        # compress the caption for CLIP models
        compressed_caption = compress_caption(caption)

        gc.collect()
        torch.cuda.empty_cache()
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
        self.pipe.encode_prompt(prompt=compressed_caption, prompt_2=compressed_caption, prompt_3=caption, do_classifier_free_guidance=False, device=self.accelerator.device)
        return prompt_embeds, pooled_prompt_embeds
    
    def validate(self):
        params = self.params
        vae = self.pipe.vae
        tokenizer = self.pipe.tokenizer
        text_encoder_3 = self.pipe.text_encoder_3
        transformer = self.pipe.transformer
        scheduler = self.pipe.scheduler
        if params.low_vram:
            vae = vae.cpu()
            transformer = transformer.cpu()
            self.pipe.vae = None
            self.pipe.transformer = None
            torch.cuda.empty_cache()

        pil_to_tensor = PILToTensor()
        idx = 0
        generator=torch.Generator().manual_seed(42)
        latents = []
        embeds = []
        for prompt in tqdm.tqdm(params.validation_prompts, desc='Generating validation embeddings'):
            compressed_prompt = compress_caption(prompt)
            self.pipe.text_encoder_3 = text_encoder_3
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                self.pipe.encode_prompt(compressed_prompt, compressed_prompt, prompt)
            embeds.append((prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds))
        
        self.pipe.text_encoder.cpu()
        self.pipe.text_encoder_2.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        self.pipe.transformer = transformer
        transformer = transformer.to(self.accelerator.device)
        self.pipe.text_encoder_3 = None
        self.pipe.to(self.accelerator.device)

        for embed in tqdm.tqdm(embeds, desc='Generating validation latents'):
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = embed
            latent = self.pipe(
                negative_prompt=None,
                prompt_embeds=prompt_embeds.to(self.accelerator.device),
                negative_prompt_embeds=negative_prompt_embeds.to(self.accelerator.device),
                pooled_prompt_embeds=pooled_prompt_embeds.to(self.accelerator.device),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.accelerator.device),
                guidance_scale=5.0,
                num_inference_steps=20,
                generator=generator,
                output_type='latent'
            )[0]
            latents.append(latent)

        transformer = transformer.cpu()
        self.pipe.transformer = None

        self.pipe.vae = vae
        vae = vae.to(self.accelerator.device)

        idx = 0
        for latent in tqdm.tqdm(latents, desc='Decoding validation latents'):
            prompt = params.validation_prompts[idx]
            latent = latent.to(vae.dtype)
            image = vae.decode(latent / vae.config.scaling_factor, return_dict=False)[0]
            image = self.pipe.image_processor.postprocess(image)
            self.logger.add_image(f'validation/{idx}/{prompt}', pil_to_tensor(image[0]), self.global_step)
            idx = idx + 1
        
        self.pipe.transformer = transformer
        self.pipe.vae = vae
        self.pipe.text_encoder_3 = text_encoder_3
        self.pipe.to(dtype=torch.bfloat16)
    
    def optimize(self, model, batch):
        text_encoder = self.pipe.text_encoder
        text_encoder_2 = self.pipe.text_encoder_2
        text_encoder_3 = self.pipe.text_encoder_3
        vae = self.pipe.vae
        transformer = self.pipe.transformer
        
        vae = vae.cpu()
        transformer = transformer.to(self.accelerator.device)

        params = self.params
        batch_size = params.batch_size
        latents, embeddings, pooled_projections = batch

        loss_fn = torch.nn.MSELoss()
        noise = randn_tensor(latents.shape, device=latents.device, dtype=latents.dtype)

        u = compute_density_for_timestep_sampling('logit_normal', batch_size, logit_mean=0, logit_std=1.0, mode_scale=1.29)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(latents.device)
        noisy_model_input = self.scheduler.scale_noise(latents, timesteps, noise)

        transformer = model
        noise_pred = transformer(noisy_model_input.to(dtype=latents.dtype),
                                encoder_hidden_states=embeddings.to(dtype=latents.dtype),
                                timestep=timesteps,
                                pooled_projections=pooled_projections.to(dtype=latents.dtype)).sample
        target = noise - latents
        loss = loss_fn(noise_pred.to(dtype=noise.dtype), target)
        return loss
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)

    trainer = SD35Trainer(params)
    trainer.run()