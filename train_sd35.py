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

class SD35Trainer(Trainer):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)
        
        if params.bfloat16:
            if self.params.low_vram:
                # put the T5 model as 8 bits
                config_8bit = BitsAndBytesConfig(load_in_8bit=True)
                text_encoder_3 = T5EncoderModel.from_pretrained(params.pretrained_pipe_path,
                                                                subfolder='text_encoder_3',
                                                                quantization_config=config_8bit)
                self.pipe = StableDiffusion3Pipeline.from_pretrained(params.pretrained_pipe_path,
                                                                     text_encoder_3=text_encoder_3,
                                                                     torch_dtype=torch.bfloat16)
            else:
                self.pipe = StableDiffusion3Pipeline.from_pretrained(params.pretrained_pipe_path, torch_dtype=torch.bfloat16)
        else:
            self.pipe = StableDiffusion3Pipeline.from_pretrained(params.pretrained_pipe_path)
        if params.pretrained_model_path != None:
            transformer = SD3Transformer2DModel.from_pretrained(params.pretrained_model_path)
            self.pipe.transformer = transformer
        self.pipe.enable_model_cpu_offload()

        # required for lower vram consumption
        self.pipe.transformer.gradient_checkpointing = True
        self.pipe.text_encoder.gradient_checkpointing = True
        self.pipe.text_encoder_2.gradient_checkpointing = True
        self.pipe.text_encoder_3.gradient_checkpointing = True
        
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
        if self.params.low_vram != True:
            self.pipe = self.pipe.to(self.accelerator.device)
    
    def extract_latents(self, images):
        if self.params.low_vram:
            vae = self.pipe.vae
            vae = vae.to(self.accelerator.device)

        image_processor = self.pipe.image_processor
        images = image_processor.preprocess(images)
        output = self.pipe.vae.encode(images.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)).latent_dist.sample()

        #if self.params.low_vram:
            #vae = vae.cpu()
        return output * self.pipe.vae.config.scaling_factor

    def extract_embeddings(self, captions):
        if self.params.low_vram:
            text_encoder = self.pipe.text_encoder.to(self.accelerator.device)
            text_encoder_2 = self.pipe.text_encoder_2.to(self.accelerator.device)
            #text_encoder_3 = self.pipe.text_encoder_3.to(self.accelerator.device)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
        self.pipe.encode_prompt(prompt=captions, prompt_2=captions, prompt_3=captions, do_classifier_free_guidance=False, device=self.accelerator.device)

        #if self.params.low_vram:
            #text_encoder = text_encoder.cpu()
            #text_encoder_2 = text_encoder_2.cpu()
            #text_encoder_3 = text_encoder_3.cpu()
        return prompt_embeds, pooled_prompt_embeds
    
    def validate(self):
        transformer = self.pipe.transformer
        transformer = transformer.to(self.accelerator.device)
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
            self.logger.add_image(f'validation/{idx}/{prompt}', pil_to_tensor(image), self.global_step)
            idx = idx + 1
        
        # save the transformer
        self.pipe.transformer.save_pretrained(f'{self.global_step}')
        transformer = transformer.cpu()
    
    def optimize(self, model, batch):
        if self.accelerator.is_main_process:
            # we need to swap the vae and text encoders to cpu and transformer to gpu as it takes too much VRAM even on a A100!
            text_encoder = self.pipe.text_encoder
            text_encoder_2 = self.pipe.text_encoder_2
            text_encoder_3 = self.pipe.text_encoder_3
            vae = self.pipe.vae
            transformer = self.pipe.transformer

            text_encoder = text_encoder.cpu()
            text_encoder_2 = text_encoder_2.cpu()
            text_encoder_3 = text_encoder_3.cpu()
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