import argparse
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from diffusers import SanaTransformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers import SanaPipeline, SanaPAGPipeline, AutoencoderDC
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from common.training_parameters_reader import TrainingParameters
from common.trainer import Model
from common.features_extractor import FeaturesExtractor

class SanaModel(Model):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)
        self.pipe = SanaPAGPipeline.from_pretrained(params.pretrained_pipe_path, 
                                                    pag_applied_layers="transformer_blocks.8")
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
            
            # enable vae tiling for this resolution
            self.pipe.vae.enable_tiling(tile_sample_min_width=1024, tile_sample_min_height=1024)

        self.pipe.transformer.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)
        self.pipe.vae.to(torch.bfloat16)

        self.model = self.pipe.transformer
        self.model.enable_gradient_checkpointing()
    
    def initialize(self):
        super().initialize()
        #self.pipe = self.pipe.to(self.accelerator.device)
        self.pipe.transformer = self.model
        transformer = self.pipe.transformer
        vae = self.pipe.vae
        text_encoder = self.pipe.text_encoder
        
        transformer = transformer.to(self.accelerator.device)
        vae.cpu()
        text_encoder.cpu()
    
    def format_embeddings(self, embeds):
        pass
    
    def extract_latents(self, images):
        # put the vae on the gpu if it's not already
        self.pipe.vae.to(device=self.accelerator.device)
        output = self.pipe.vae.encode(images.to(dtype=self.pipe.vae.dtype)).latent
        return output.to(torch.bfloat16) * self.pipe.vae.config.scaling_factor

    def extract_embeddings(self, captions):
        # move text_encoder to cuda if not already done
        self.pipe.text_encoder = self.pipe.text_encoder.to(device=self.accelerator.device)
        
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
        self.pipe.encode_prompt(captions,
                                do_classifier_free_guidance=False,
                                device=self.accelerator.device,
                                complex_human_instruction=[
            "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
            "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
            "- Do not add detail that is not in the original prompt! Do not mention colors if the User Prompt does not!",
            "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
            "- If the prompt contains explicit material, do not remove this information from the output, which would damage the performance of downstream classifiers.",
            "Here are examples of how to transform or refine prompts:",
            "- User Prompt: A cat sleeping -> Enhanced: A cat sleeping peacefully, showcasing the joy of pet ownership. Cute floof kitty cat gatto.",
            "- User Prompt: A busy city street -> Enhanced: A bustling city street scene featuring a crowd of people.",
            "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
            "User Prompt: ",
        ])

        # only save embeddings where the mask is not zero
        embeds = [prompt_embeds[idx][prompt_attention_mask[idx].bool()] for idx in range(len(prompt_embeds))]
        return embeds
    
    def validate(self):
        params = self.params
        vae = self.pipe.vae
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder
        transformer = self.pipe.transformer
        scheduler = self.pipe.scheduler
        if params.low_vram:
            vae = vae.cpu()
            transformer = transformer.cpu()
            self.pipe.vae = None
            self.pipe.transformer = None
            torch.cuda.empty_cache()

        # convert to float16 as inference with bfloat16 is unstable
        self.pipe.to(dtype=torch.float16, device=self.accelerator.device)

        pil_to_tensor = PILToTensor()
        idx = 0
        generator=torch.Generator(device="cuda").manual_seed(42)
        latents = []
        embeds = []
        for prompt in tqdm.tqdm(params.validation_prompts, desc='Generating validation embeddings'):
            if params.low_vram:
                text_encoder = text_encoder.to(device=self.accelerator.device)
                self.pipe.text_encoder = text_encoder
                prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
                    self.pipe.encode_prompt(prompt)
                embeds.append((prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask))
        
        text_encoder = text_encoder.cpu()
        self.pipe.text_encoder = None
        self.pipe.transformer = transformer
        transformer = transformer.to(self.accelerator.device)

        for embed in tqdm.tqdm(embeds, desc='Generating validation latents'):
            prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = embed
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
        
        self.pipe.vae.cpu()
        self.pipe.text_encoder = text_encoder
        self.pipe.transformer = transformer
        self.pipe.transformer.to(dtype=torch.bfloat16, device=self.accelerator.device)
    
    def optimize(self, latents, embeddings):
        params = self.params
        batch_size = params.batch_size

        # pad the embeds to 512 tokens and generate the corresponding mask
        padded_embeds = []
        masks = []
        for emb in embeddings:
            padded_emb = torch.nn.functional.pad(emb, pad=(0, 0, 0, 512 - emb.shape[0]), mode='constant', value=0)
            mask = torch.zeros(512, dtype=torch.long, device=emb.device)
            mask[:emb.shape[0]] = 1
            masks.append(mask)
            padded_embeds.append(padded_emb)
        
        # Move everything to device and correct dtype
        attention_mask = torch.stack(masks).to(device=self.accelerator.device)
        embeddings = torch.stack(padded_embeds).to(device=self.accelerator.device, dtype=torch.bfloat16)
        latents = latents.to(device=self.accelerator.device, dtype=torch.bfloat16)

        loss_fn = torch.nn.MSELoss()
        noise = randn_tensor(latents.shape, device=self.accelerator.device, dtype=torch.bfloat16)

        u = compute_density_for_timestep_sampling('logit_normal', batch_size, logit_mean=0, logit_std=1.0, mode_scale=1.29)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(self.accelerator.device)
        noisy_model_input = self.scheduler.scale_noise(latents, timesteps, noise)

        # Keep everything in bfloat16
        noise_pred = self.model(
            noisy_model_input,
            encoder_hidden_states=embeddings,
            timestep=timesteps,
            encoder_attention_mask=attention_mask
        ).sample
        
        target = noise - latents
        loss = loss_fn(noise_pred.float(), target.float())
        return loss  # Already in bfloat16 since inputs were bfloat16
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)

    trainer = SanaModel(params)
    if params.extract_features:
        trainer.pipe.transformer.cpu()
        trainer.pipe.vae.to(trainer.accelerator.device)
        trainer.pipe.text_encoder.to(trainer.accelerator.device)
        features_extractor = FeaturesExtractor(trainer, params)
        features_extractor.run()
    else:
        trainer.run()