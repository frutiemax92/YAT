import argparse
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_512_BIN
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from common.training_parameters_reader import TrainingParameters
from common.trainer import Model
from common.features_extractor import FeaturesExtractor
from utils.compress_caption import compress_caption

class SD15Model(Model):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)
        
        # read the single safetensors to build the sd1.5 pipeline to start from the very good finetunes from civitai
        if params.pretrained_pipe_single_file != None:
            self.pipe = StableDiffusionPipeline.from_single_file(
                params.pretrained_pipe_single_file,
                torch_dtype="float16",
                use_safetensors=True)
        else:
            # useful to start from a diffusers pipeline directly
            self.pipe = StableDiffusionPipeline.from_pretrained(params.pretrained_pipe_path)
        if params.pretrained_model_path != None:
            self.pipe.unet = UNet2DConditionModel.from_pretrained(params.pretrained_model_path)
        
        config = self.pipe.scheduler.config
        self.scheduler = DDPMScheduler.from_config(config)
        self.pipe.vae.train(False)
        self.pipe.text_encoder.train(False)

        # only train in 512px mode for now
        self.aspect_ratios = ASPECT_RATIO_512_BIN

        self.pipe.unet.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)
        self.pipe.vae.to(torch.bfloat16)

        self.model = self.pipe.unet
        self.model.enable_gradient_checkpointing()
    
    def initialize(self):
        super().initialize()
        self.pipe.unet = self.model
        unet = self.pipe.unet
        vae = self.pipe.vae
        text_encoder = self.pipe.text_encoder
        
        unet = unet.to(self.accelerator.device)
        vae.cpu()
        text_encoder.cpu()
    
    def format_embeddings(self, embeds):
        pass
    
    def extract_latents(self, images):
        # put the vae on the gpu if it's not already
        vae = self.pipe.vae
        vae = vae.to(device=self.accelerator.device)
        output = self.pipe.vae.encode(images.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)).latent_dist.sample()
        return output * self.pipe.vae.config.scaling_factor

    def extract_embeddings(self, captions):
        # move text_encoder to cuda if not already done
        self.pipe.text_encoder = self.pipe.text_encoder.to(device=self.accelerator.device)

        # we need to compress the caption for SD15
        new_captions = [compress_caption(caption) if caption != '' else '' for caption in captions]
        
        prompt_embeds, negative_prompt_embeds = \
        self.pipe.encode_prompt(new_captions,
                                do_classifier_free_guidance=False,
                                device=self.accelerator.device,
                                num_images_per_prompt=1)
        return prompt_embeds
    
    def validate(self):
        params = self.params
        vae = self.pipe.vae
        text_encoder = self.pipe.text_encoder
        unet = self.pipe.unet

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
                prompt_embeds, negative_prompt_embeds = \
                    self.pipe.encode_prompt(
                        prompt,
                        device=self.accelerator.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True)
                embeds.append((prompt_embeds, negative_prompt_embeds))
        
        text_encoder = text_encoder.cpu()
        self.pipe.text_encoder = None
        self.pipe.unet = unet
        unet = unet.to(self.accelerator.device)

        for embed in tqdm.tqdm(embeds, desc='Generating validation latents'):
            prompt_embeds, negative_prompt_embeds = embed
            latent = self.pipe(
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=5.0,
                num_inference_steps=20,
                generator=generator,
                output_type='latent'
            )[0]
            latents.append(latent)

        unet = unet.cpu()
        self.pipe.unet = None

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
        self.pipe.unet = unet
        self.pipe.unet.to(dtype=torch.bfloat16, device=self.accelerator.device)
    
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
        noisy_model_input = self.scheduler.add_noise(latents.to(self.accelerator.device), noise, timesteps)

        # Keep everything in bfloat16
        noise_pred = self.model(
            noisy_model_input,
            encoder_hidden_states=embeddings,
            timestep=timesteps,
            encoder_attention_mask=attention_mask
        ).sample
        
        target = noise
        loss = loss_fn(noise_pred.float(), target.float())
        return loss  # Already in bfloat16 since inputs were bfloat16
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)

    trainer = SD15Model(params)
    if params.extract_features:
        trainer.pipe.unet.cpu()
        trainer.pipe.vae.to(trainer.accelerator.device)
        trainer.pipe.text_encoder.to(trainer.accelerator.device)
        features_extractor = FeaturesExtractor(trainer, params)
        features_extractor.run()
    else:
        trainer.run()