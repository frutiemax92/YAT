import argparse
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers import Krea2Transformer2DModel, Krea2Pipeline, FlowMatchEulerDiscreteScheduler, BitsAndBytesConfig
from diffusers.training_utils import compute_density_for_timestep_sampling
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from common.training_parameters_reader import TrainingParameters
from common.trainer import Model
from common.features_extractor import FeaturesExtractor
from diffusers.quantizers import PipelineQuantizationConfig

class Krea2Model(Model):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)

        # bnb 4bit for the (huge, resident every step since compute_features runs it live) Qwen3-VL text encoder.
        # Gated on lora_base_model_4bit, NOT use_adamw_8bit (that flag only controls the optimizer choice).
        pipeline_quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=["text_encoder"],
        ) if params.lora_base_model_4bit else None

        # separate bnb config for the transformer: PipelineQuantizationConfig only quantizes components the
        # pipe itself instantiates from the checkpoint, so a transformer passed in pre-built (pretrained_model_path
        # branch) or loaded standalone (subfolder branch below) needs its own quantization_config to actually shrink.
        transformer_quant_config = None
        if params.lora_base_model_4bit:
            transformer_quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )

        if params.pretrained_model_path != None:
            transformer = Krea2Transformer2DModel.from_pretrained(params.pretrained_model_path,
                                                                 quantization_config=transformer_quant_config,
                                                                 torch_dtype=torch.bfloat16,
                                                                 device_map=f"cuda:{self.accelerator.process_index}")

            self.pipe = Krea2Pipeline.from_pretrained(
                params.pretrained_pipe_path,
                quantization_config=pipeline_quant_config,
                transformer=transformer,
                torch_dtype=torch.bfloat16)
        else:
            pipe_kwargs = dict(torch_dtype=torch.bfloat16, quantization_config=pipeline_quant_config)
            if params.lora_base_model_4bit:
                pipe_kwargs['transformer'] = Krea2Transformer2DModel.from_pretrained(
                    params.pretrained_pipe_path,
                    subfolder='transformer',
                    quantization_config=transformer_quant_config,
                    torch_dtype=torch.bfloat16,
                    device_map=f"cuda:{self.accelerator.process_index}")
            self.pipe = Krea2Pipeline.from_pretrained(params.pretrained_pipe_path, **pipe_kwargs)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(params.pretrained_pipe_path, subfolder='scheduler')
        self.pipe.vae.train(False)
        self.pipe.text_encoder.train(False)

        # fixed text sequence length consumed by the transformer, matching the pipeline's own default
        self.max_sequence_length = 512
        self.aspect_ratios = ASPECT_RATIO_1024_BIN

        self.pipe.vae.to(torch.bfloat16)

        self.model = self.pipe.transformer
        self.model.enable_gradient_checkpointing()

    def initialize(self):
        super().initialize()
        self.pipe.transformer = self.accelerator.unwrap_model(self.model)

    def format_embeddings(self, embeds):
        pass

    def extract_latents(self, images):
        # put the vae on the gpu if it's not already
        vae = self.pipe.vae
        vae.to(device=self.accelerator.device)

        # the qwen-image vae is a video vae, so images need a frame dimension of 1
        pixel_values = images.to(device=self.accelerator.device, dtype=vae.dtype).unsqueeze(2)
        latents = vae.encode(pixel_values).latent_dist.sample()[:, :, 0]

        latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1)
        latents = (latents - latents_mean) / latents_std
        return latents.to(torch.bfloat16)

    def extract_embeddings(self, captions):
        # move text_encoder to cuda if not already done
        self.pipe.text_encoder.to(device=self.accelerator.device)
        hidden_states, attention_mask = self.pipe.get_text_hidden_states(
            captions, self.max_sequence_length, device=self.accelerator.device)

        # only save embeddings where the mask is not zero
        embeds = [hidden_states[idx][attention_mask[idx]] for idx in range(len(hidden_states))]
        return embeds

    def enable_efficient_attention(self):
        pass

    def validate(self):
        params = self.params
        vae = self.pipe.vae
        text_encoder = self.pipe.text_encoder
        transformer = self.pipe.transformer

        pil_to_tensor = PILToTensor()
        idx = 0
        generator = torch.Generator(device=self.accelerator.device).manual_seed(42)
        embeds = []

        text_encoder = text_encoder.to(device=self.accelerator.device)
        self.pipe.text_encoder = text_encoder
        negative_prompt_embeds, negative_prompt_embeds_mask = self.pipe.encode_prompt(
            prompt="", device=self.accelerator.device)

        for prompt in tqdm.tqdm(params.validation_prompts, desc='Generating validation embeddings'):
            prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(prompt=prompt, device=self.accelerator.device)
            embeds.append((prompt_embeds, prompt_embeds_mask))

        self.pipe.text_encoder = None
        self.pipe.transformer = self.accelerator.unwrap_model(transformer)

        for embed in tqdm.tqdm(embeds, desc='Generating validation images'):
            prompt_embeds, prompt_embeds_mask = embed
            image = self.pipe(
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                guidance_scale=4.5,
                num_inference_steps=28,
                generator=generator,
                width=512,
                height=512,
            ).images[0]
            self.logger.add_image(f'validation/{idx}/{params.validation_prompts[idx]}', pil_to_tensor(image), self.global_step)
            idx = idx + 1

        self.pipe.text_encoder = text_encoder
        self.pipe.transformer = transformer

    def optimize(self, ratio, latents, embeddings, repa_tokens, generator: torch.Generator = None):
        params = self.params
        batch_size = params.batch_size
        max_sequence_length = self.max_sequence_length

        # pad the embeds to the fixed text sequence length and generate the corresponding mask
        padded_embeds = []
        masks = []
        for emb in embeddings:
            padded_emb = torch.nn.functional.pad(emb, pad=(0, 0, 0, 0, 0, max_sequence_length - emb.shape[0]), mode='constant', value=0)
            mask = torch.zeros(max_sequence_length, dtype=torch.bool, device=emb.device)
            mask[:emb.shape[0]] = True
            masks.append(mask)
            padded_embeds.append(padded_emb)

        # Move everything to device and correct dtype
        encoder_attention_mask = torch.stack(masks).to(device=self.accelerator.device)
        prompt_embeds = torch.stack(padded_embeds).to(device=self.accelerator.device, dtype=torch.bfloat16)
        latents = latents.to(device=self.accelerator.device, dtype=torch.bfloat16)

        loss_fn = torch.nn.MSELoss()
        noise = randn_tensor(latents.shape, device=self.accelerator.device, dtype=torch.bfloat16, generator=generator)

        u = compute_density_for_timestep_sampling(
            'logit_normal',
            batch_size,
            logit_mean=0,
            logit_std=1.0,
            mode_scale=1.29,
            generator=generator)
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
        target = noise - latents

        # pack the spatial latents into the sequence-of-patches layout the transformer expects
        num_channels_latents = latents.shape[1]
        latent_height = latents.shape[2]
        latent_width = latents.shape[3]
        patch_size = self.pipe.patch_size

        packed_noisy_model_input = self.pipe._pack_latents(noisy_model_input, batch_size, num_channels_latents, latent_height, latent_width)
        packed_target = self.pipe._pack_latents(target, batch_size, num_channels_latents, latent_height, latent_width)

        grid_height = latent_height // patch_size
        grid_width = latent_width // patch_size
        position_ids = self.pipe.prepare_position_ids(prompt_embeds.shape[1], grid_height, grid_width, self.accelerator.device)

        # flow-matching time in [0, 1], matching the pipeline's own convention
        timestep = (timesteps / self.scheduler.config.num_train_timesteps).to(dtype=packed_noisy_model_input.dtype)

        # Keep everything in bfloat16
        noise_pred = self.model(
            hidden_states=packed_noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            position_ids=position_ids,
            encoder_attention_mask=encoder_attention_mask
        ).sample

        loss = loss_fn(noise_pred.float(), packed_target.float())
        return loss  # Already in bfloat16 since inputs were bfloat16

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)

    trainer = Krea2Model(params)
    if params.extract_features:
        trainer.pipe.transformer.cpu()
        trainer.pipe.vae.to(trainer.accelerator.device)
        trainer.pipe.text_encoder.to(trainer.accelerator.device)
        features_extractor = FeaturesExtractor(trainer, params)
        features_extractor.run()
    else:
        trainer.run()
