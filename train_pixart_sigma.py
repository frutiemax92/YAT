import argparse
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from diffusers import PixArtSigmaPipeline, DDPMScheduler, PixArtTransformer2DModel
from utils.expand_pixart_sigma_transformer import PixArtTransformer2DModelWithResNet
from diffusers.training_utils import compute_density_for_timestep_sampling
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from common.training_parameters_reader import TrainingParameters
from common.trainer import Trainer
from transformers import AutoConfig, PretrainedConfig

class PixartSigmaTrainer(Trainer):
    def __init__(self, params : TrainingParameters, config):
        super().__init__(params)

        if params.bfloat16:
            self.pipe = PixArtSigmaPipeline.from_pretrained(params.pretrained_pipe_path, torch_dtype=torch.bfloat16)
        else:
            self.pipe = PixArtSigmaPipeline.from_pretrained(params.pretrained_pipe_path)
        if params.pretrained_model_path != None:
            if config._class_name != 'PixArtTransformer2DModelWithResNet':
                transformer = PixArtTransformer2DModel.from_pretrained(params.pretrained_model_path)
            else:
                transformer = PixArtTransformer2DModelWithResNet.from_pretrained(params.pretrained_model_path)
            self.pipe.transformer = transformer
        self.pipe.transformer.enable_gradient_checkpointing()
        
        self.scheduler = DDPMScheduler.from_pretrained(params.pretrained_pipe_path, subfolder='scheduler')
        self.pipe.vae.train(False)
        self.pipe.text_encoder.train(False)

        vae_compression = 8
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
        text_encoder = self.pipe.text_encoder
        vae = self.pipe.vae
        transformer = self.pipe.transformer
        text_encoder, vae, transformer = self.accelerator.prepare(text_encoder,
                                                                  vae,
                                                                  transformer)
    
    def extract_latents(self, images):
        image_processor = self.pipe.image_processor
        images = image_processor.preprocess(images)
        output = self.pipe.vae.encode(images.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)).latent_dist.sample()
        return output * self.pipe.vae.config.scaling_factor

    def extract_embeddings(self, captions):
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
        self.pipe.encode_prompt(captions, do_classifier_free_guidance=False)
        return prompt_embeds, prompt_attention_mask
    
    def validate(self):
        params = self.params
        pil_to_tensor = PILToTensor()
        idx = 0
        generator=torch.Generator(device="cuda").manual_seed(42)
        for prompt in tqdm.tqdm(params.validation_prompts, desc='Generating validation images'):
            image = self.pipe(
                prompt=prompt,
                guidance_scale=5.0,
                num_inference_steps=20,
                generator=generator,
                output_type='latent').images
            image = self.pipe.vae.decode(image.to(self.pipe.dtype) / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = self.pipe.image_processor.postprocess(image, output_type='pt')[0]
            self.logger.add_image(f'validation/{idx}/{prompt}', image, self.global_step)
            idx = idx + 1
    
    def optimize(self, model, batch):
        params = self.params
        batch_size = params.batch_size
        latents, embeddings, attention_mask = batch

        loss_fn = torch.nn.MSELoss()
        noise = randn_tensor(latents.shape, device=self.pipe.device, dtype=self.pipe.dtype)

        u = compute_density_for_timestep_sampling('logit_normal', batch_size, logit_mean=0, logit_std=1.0, mode_scale=1.29)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(latents.device)
        noisy_model_input = self.scheduler.add_noise(latents, noise, timesteps)

        transformer = model
        noise_pred = transformer(noisy_model_input.to(dtype=transformer.dtype),
                                encoder_hidden_states=embeddings.to(dtype=transformer.dtype),
                                timestep=timesteps,
                                encoder_attention_mask=attention_mask.to(dtype=transformer.dtype)).sample.chunk(2, 1)[0]
        target = noise
        loss = loss_fn(noise_pred.to(dtype=noise.dtype), target)
        return loss
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)

    config = PretrainedConfig.from_pretrained(params.pretrained_model_path)

    trainer = PixartSigmaTrainer(params, config)
    trainer.run()