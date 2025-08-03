import argparse
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from diffusers import PixArtSigmaPipeline, DDPMScheduler, PixArtTransformer2DModel
from utils.expand_pixart_sigma_transformer import PixArtTransformer2DModelWithResNet
from utils.patch_pixart_sigma_transformer import REPAPixArtTransformerModel
from diffusers.training_utils import compute_density_for_timestep_sampling
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from common.training_parameters_reader import TrainingParameters
from utils.patch_pixart_sigma_pipeline import PatchedPixartSigmaPipeline
from common.trainer import Model
from transformers import AutoConfig, PretrainedConfig
import gc

class PixartSigmaTrainer(Model):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)

        if params.use_repa == False:
            self.pipe = PixArtSigmaPipeline.from_pretrained(params.pretrained_pipe_path, torch_dtype=torch.bfloat16)
        else:
            self.pipe = PatchedPixartSigmaPipeline.from_pretrained(params.pretrained_pipe_path, torch_dtype=torch.bfloat16)
        if params.pretrained_model_path != None:
            if params.use_repa == False:
                transformer = PixArtTransformer2DModel.from_pretrained(params.pretrained_model_path)
            else:
                transformer = REPAPixArtTransformerModel.from_pretrained(params.pretrained_model_path)
            self.pipe.transformer = transformer
        self.pipe.transformer.enable_gradient_checkpointing()
        self.pipe.vae.enable_gradient_checkpointing()
        
        self.scheduler = DDPMScheduler.from_pretrained(params.pretrained_pipe_path, subfolder='scheduler')
        self.pipe.vae.train(False)
        self.pipe.text_encoder.train(False)

        vae_compression = self.pipe.vae_scale_factor
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
            self.pipe.transformer = self.pipe.transformer.to(torch.bfloat16)
        self.model = self.pipe.transformer
        
    
    def initialize(self):
        super().initialize()
    
    def extract_latents(self, images):
        # move vae to cuda if it's not already done
        self.pipe.transformer.cpu()
        vae = self.pipe.vae
        vae = vae.to(device=self.accelerator.device)
        image_processor = self.pipe.image_processor
        images = image_processor.preprocess(images)

        gc.collect()
        torch.cuda.empty_cache()

        if self.pipe.vae.config['_class_name'] != 'AutoencoderDC':
            output = self.pipe.vae.encode(images.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)).latent_dist.sample()
        else:
            output = vae.encode(images).latent
        return output * self.pipe.vae.config.scaling_factor

    def extract_embeddings(self, captions):
        # move text_encoder to cuda if not already done
        self.pipe.transformer.cpu()
        self.pipe.text_encoder = self.pipe.text_encoder.to(device=self.accelerator.device)

        gc.collect()
        torch.cuda.empty_cache()
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
        vae = vae.cpu()
        transformer = transformer.cpu()
        self.pipe.vae = None
        self.pipe.transformer = None
        torch.cuda.empty_cache()

        # convert to float16 as inference with bfloat16 is unstable
        self.pipe.to(dtype=torch.bfloat16, device=self.accelerator.device)

        pil_to_tensor = PILToTensor()
        idx = 0
        generator=torch.Generator(device="cuda").manual_seed(42)
        latents = []
        embeds = []
        for prompt in tqdm.tqdm(params.validation_prompts, desc='Generating validation embeddings'):
            text_encoder = text_encoder.to(device=self.accelerator.device)
            self.pipe.text_encoder = text_encoder
            prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
                self.pipe.encode_prompt(prompt)
            prompt_embeds = prompt_embeds.repeat(1, 1, 1)
            prompt_attention_mask = prompt_attention_mask.repeat(1, 1)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, 1 ,1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(1, 1)

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
            image = self.pipe.image_processor.postprocess(image, output_type='pil')[0]
            self.logger.add_image(f'validation/{idx}/{prompt}', pil_to_tensor(image), self.global_step)
            idx = idx + 1
        
        self.pipe.transformer = transformer
        self.pipe.vae = vae
        self.pipe.text_encoder = text_encoder
        self.pipe.to(dtype=torch.bfloat16)
    
    def optimize(self, model, batch):
        self.pipe.vae.cpu()
        self.pipe.text_encoder.cpu()
        self.model.to(self.accelerator.device)
        params = self.params
        batch_size = params.batch_size
        _, latents, embeddings, attention_mask = batch

        loss_fn = torch.nn.MSELoss()
        noise = randn_tensor(latents.shape, device=self.accelerator.device, dtype=self.pipe.dtype)

        u = compute_density_for_timestep_sampling('logit_normal', batch_size, logit_mean=0, logit_std=1.0, mode_scale=1.29)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(latents.device)
        noisy_model_input = self.scheduler.add_noise(latents.to(self.accelerator.device), noise, timesteps)

        transformer = model
        noise_pred = transformer(noisy_model_input.to(dtype=transformer.dtype),
                                encoder_hidden_states=embeddings.to(dtype=transformer.dtype),
                                timestep=timesteps,
                                encoder_attention_mask=attention_mask.to(dtype=transformer.dtype)).sample.chunk(2, 1)[0]
        target = noise[:noise_pred.shape[0], :noise_pred.shape[1], :noise_pred.shape[2], :noise_pred.shape[3]]
        loss = loss_fn(noise_pred.to(dtype=noise.dtype), target)

        if hasattr(model, 'get_alphas'):
            alphas = model.get_alphas()
            count = len(alphas)
            mean_alpha = torch.mean(torch.stack(alphas))
            loss_alpha = loss_fn(mean_alpha, torch.tensor(1.0, device=mean_alpha.device, dtype=mean_alpha.dtype))
            loss = loss + loss_alpha
            if self.logger != None:
                self.logger.add_scalar('train/mean_alpha', mean_alpha.item(), self.global_step)
                self.logger.add_scalar('train/loss_alpha', loss_alpha.item(), self.global_step)
        return loss
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)

    trainer = PixartSigmaTrainer(params)
    trainer.run()