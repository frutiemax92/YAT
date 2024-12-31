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

class SD35Trainer(Trainer):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)
        self.pipe = StableDiffusion3Pipeline.from_pretrained(params.pretrained_pipe_path)
        if params.pretrained_model_path != None:
            transformer = SD3Transformer2DModel.from_pretrained(params.pretrained_model_path)
            self.pipe.transformer = transformer

        # required for lower vram consumption
        self.pipe.transformer.gradient_checkpointing = True
        
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(params.pretrained_pipe_path, subfolder='scheduler')
        self.pipe.vae.train(False)
        self.pipe.text_encoder.train(False)
        self.pipe.text_encoder_2.train(False)
        self.pipe.text_encoder_3.train(False)

        self.aspect_ratios = ASPECT_RATIO_1024_BIN

        self.optimizer = AdamW(self.pipe.transformer.parameters(), lr=params.learning_rate)
        if params.bfloat16:
            self.pipe = self.pipe.to(torch.bfloat16)
        self.model = self.pipe.transformer
    
    def initialize(self):
        super().initialize()
        text_encoder = self.pipe.text_encoder
        text_encoder_2 = self.pipe.text_encoder_2
        text_encoder_3 = self.pipe.text_encoder_3
        vae = self.pipe.vae
        transformer = self.pipe.transformer
        text_encoder, text_encoder_2, text_encoder_3, vae, transformer = \
            self.accelerator.prepare(text_encoder, text_encoder_2, text_encoder_3, vae, transformer)
    
    def extract_latents(self, images):
        image_processor = self.pipe.image_processor
        images = image_processor.preprocess(images)
        output = self.pipe.vae.encode(images.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)).latent_dist.sample()
        return output * self.pipe.vae.config.scaling_factor

    def extract_embeddings(self, captions):
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
        self.pipe.encode_prompt(prompt=captions, do_classifier_free_guidance=False)
        return prompt_embeds, pooled_prompt_embeds
    
    def validate(self):
        params = self.params
        pil_to_tensor = PILToTensor()
        idx = 0
        generator=torch.Generator(device="cuda").manual_seed(42)
        for prompt in tqdm.tqdm(params.validation_prompts, desc='Generating validation images'):
            image = self.pipe(
                width=800,
                height=1200,
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=4.0,
                max_sequence_length=512,
            ).images[0]
            self.logger.add_image(f'validation/{idx}/{prompt}', pil_to_tensor(image[0]), self.global_step)
            idx = idx + 1
        
        # save the transformer
        self.pipe.transformer.save_pretrained(f'{self.global_step}')
    
    def optimize(self, model, batch):
        params = self.params
        batch_size = params.batch_size
        latents, embeddings, pooled_projections = batch

        loss_fn = torch.nn.MSELoss()
        noise = randn_tensor(latents.shape, device=self.pipe.device, dtype=self.pipe.dtype)

        u = compute_density_for_timestep_sampling('logit_normal', batch_size, logit_mean=0, logit_std=1.0, mode_scale=1.29)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(latents.device)
        noisy_model_input = self.scheduler.scale_noise(latents, timesteps, noise)

        transformer = model
        noise_pred = transformer(noisy_model_input.to(dtype=transformer.dtype),
                                encoder_hidden_states=embeddings.to(dtype=transformer.dtype),
                                timestep=timesteps,
                                pooled_projections=pooled_projections.to(dtype=transformer.dtype)).sample
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