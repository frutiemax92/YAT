import argparse
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from diffusers import SanaTransformer2DModel, FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel, Flux2KleinPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers import SanaPipeline, SanaPAGPipeline, AutoencoderDC
from utils.patched_sana_transformer import PatchedSanaTransformer2DModel, patch_sana_attention_layers
from utils.sana_patches.add_skip_connections import add_skip_connections
from utils.patch_sana_attention_layers import unfreeze_sana_blocks
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from common.training_parameters_reader import TrainingParameters
from common.trainer import Model
from common.features_extractor import FeaturesExtractor

class KleinModel(Model):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)
        
        if params.pretrained_model_path != None:
            transformer = Flux2Transformer2DModel.from_pretrained(params.pretrained_model_path, 
                                                                 quantization_config=self.quantization_config, 
                                                                 torch_dtype=torch.bfloat16,
                                                                 device_map=f"cuda:{self.accelerator.process_index}")
            self.pipe = Flux2KleinPipeline.from_pretrained(params.pretrained_pipe_path, transformer=transformer, torch_dtype=torch.bfloat16) 
        else:
            self.pipe = Flux2KleinPipeline.from_pretrained(params.pretrained_pipe_path, torch_dtype=torch.bfloat16) 
        
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(params.pretrained_pipe_path, subfolder='scheduler')
        self.pipe.vae.train(False)
        self.pipe.text_encoder.train(False)
        #self.pipe.enable_model_cpu_offload()
        self.aspect_ratios = ASPECT_RATIO_1024_BIN

        self.pipe.text_encoder.to(torch.bfloat16)
        self.pipe.vae.to(torch.bfloat16)

        self.model = self.pipe.transformer
        self.model.enable_gradient_checkpointing()

    def initialize(self):
        super().initialize()
        #self.pipe = self.pipe.to(self.accelerator.device)
        self.pipe.transformer = self.model
        transformer = self.pipe.transformer
        transformer.to(self.accelerator.device)
    
    def format_embeddings(self, embeds):
        pass
    
    def extract_latents(self, images):
        # put the vae on the gpu if it's not already
        self.pipe.vae.to(device=self.accelerator.device)
        output = self.pipe.vae.encode(images.to(dtype=self.pipe.vae.dtype)).latent_dist.sample()
        return output.to(torch.bfloat16)

    def extract_embeddings(self, captions):
        # move text_encoder to cuda if not already done
        self.pipe.text_encoder.to(device=self.accelerator.device)
        prompt_embeds, text_ids = \
        self.pipe.encode_prompt(captions,
                                device=self.accelerator.device)
        return [(prompt_embeds[idx], text_ids[idx]) for idx in range(len(prompt_embeds))]
    
    def validate(self):
        params = self.params
        pil_to_tensor = PILToTensor()

        negative_prompt = ""
        idx = 0

        self.pipe.transformer.cpu()
        self.pipe.vae.cpu()
        torch.cuda.empty_cache()

        embeds = []
        neg_embeds = None
        with torch.no_grad():
            for prompt in tqdm.tqdm(params.validation_prompts, desc='Generating validation prompts'):
                prompt_embeds, text_ids = self.pipe.encode_prompt(prompt)
                embeds.append(prompt_embeds)

            neg_embeds, text_ids = self.pipe.encode_prompt(negative_prompt)

        #embeds = torch.stack(embeds)

        self.pipe.text_encoder.cpu()
        self.pipe.transformer.to(self.accelerator.device)
        self.pipe.vae.to(self.accelerator.device)
        torch.cuda.empty_cache()

        for embed in tqdm.tqdm(embeds, desc='Generating validation images'):
            prompt_embeds = embed
            image = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                height=1200,
                width=800,
                guidance_scale=7.0,
                num_inference_steps=20,
            ).images[0]
            self.logger.add_image(f'validation/{idx}/{prompt}', pil_to_tensor(image), self.global_step)
            idx = idx + 1
        
        self.pipe.text_encoder.to(self.accelerator.device)
    
    def optimize(self, ratio, latents, embeddings):
        params = self.params
        batch_size = params.batch_size

        # pad the embeds to 512 tokens and generate the corresponding mask
        prompt_embeds = []
        text_ids = []
        for emb, text_id in embeddings:
            prompt_embeds.append(emb)
            text_ids.append(text_id)
        
        prompt_embeds = torch.stack(prompt_embeds).to(dtype=torch.bfloat16, device=self.accelerator.device)
        latents = latents.to(device=self.accelerator.device, dtype=torch.bfloat16)
        latents = self.pipe._patchify_latents(latents)
        text_ids = torch.stack(text_ids).to(dtype=torch.bfloat16, device=self.accelerator.device)

        loss_fn = torch.nn.MSELoss()
        noise = randn_tensor(latents.shape, device=self.accelerator.device, dtype=torch.bfloat16)

        u = compute_density_for_timestep_sampling('logit_normal', batch_size, logit_mean=0, logit_std=1.0, mode_scale=1.29)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(self.accelerator.device)

        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = self.scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
            schedule_timesteps = self.scheduler.timesteps.to(self.accelerator.device)
            timesteps = timesteps.to(self.accelerator.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma
        
        sigmas = get_sigmas(timesteps, latents.ndim, dtype=latents.dtype)
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
        
        image_ids = self.pipe._prepare_latent_ids(noisy_model_input).to(device=self.accelerator.device)
        noisy_model_input = self.pipe._pack_latents(noisy_model_input)

        # Keep everything in bfloat16
        #hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
        noise_pred = self.model(
            noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps / 1000,
            txt_ids=text_ids,
            img_ids=image_ids
        ).sample
        
        target = noise - latents
        
        noise_pred = noise_pred[:, : noisy_model_input.size(1) :]
        noise_pred = Flux2KleinPipeline._unpack_latents_with_ids(noise_pred, image_ids)
        loss = loss_fn(noise_pred.float(), target.float())
        return loss  # Already in bfloat16 since inputs were bfloat16
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)

    trainer = KleinModel(params)
    trainer.run()