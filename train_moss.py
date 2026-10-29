import argparse
import random
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers import AutoencoderKLFlux2, BitsAndBytesConfig
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.dynamic_modules_utils import get_class_from_dynamic_module
from common.training_parameters_reader import TrainingParameters
from common.trainer import Model
from common.precompute import precomputed_prompts_available
from common.features_extractor import FeaturesExtractor

# MOSS-Image ships its transformer, scheduler and pipeline as remote code in the model repo
CUSTOM_PIPELINE = 'pipeline_moss_image'

# The remote transformer casts the activations of the text projection to `text_embedder.proj.weight.dtype`,
# which is uint8 on a bnb 4 bit layer, so these linears stay in bf16 along with the small embedders and
# the head. Everything inside the blocks is quantized: with the cpu offload, a bf16 linear would have its
# weight sent back to the cpu by the offload hook while the autograd graph still needs it for the backward.
SKIP_4BIT_MODULES = [
    'video_embedder',
    'text_embedder.proj',
    'time_embedding',
    'shared_block_modulation',
    'video_head_modulation',
    'final_linear_video',
]


def enable_block_checkpointing(blocks, use_reentrant):
    """Gradient checkpointing for the main blocks of MOSSImageTransformer.

    The remote model has no gradient checkpointing support of its own, and its block cannot run with a
    quantized query projection, so the block's forward is replaced too. The call of the block is what
    gets checkpointed, `__call__` and not `forward`, so that the hooks of the cpu offload fire again
    when the backward replays the block.
    """
    for block in blocks:
        helpers = type(block).forward.__globals__

        class CheckpointedBlock(type(block)):
            def forward(self, x, rope, modulation):
                # the original reads the dtype of the activations from the weight of the query projection,
                # which is uint8 once it is quantized: take it from the activations instead
                shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
                dtype = x.dtype
                y = (self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]).to(dtype)
                q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
                k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
                x = x + self.proj['o'](helpers['_attention'](q, k, self.proj['v'](y), rope, self.heads)) * gate_a[:, None]
                y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
                hidden, gate = self.proj['ffn_up_gate'](y.to(dtype)).chunk(2, -1)
                return x + self.proj['ffn_down'](helpers['_situ'](hidden, gate)) * gate_f[:, None]

            def __call__(self, x, *args):
                if not torch.is_grad_enabled():
                    return super().__call__(x, *args)
                # the reentrant variant needs an input that requires grad, or nothing downstream of the block
                # does either: the activations come out of frozen embedders when only an adapter trains
                if use_reentrant and not x.requires_grad:
                    x.requires_grad_(True)
                return torch.utils.checkpoint.checkpoint(super().__call__, x, *args, use_reentrant=use_reentrant)
        block.__class__ = CheckpointedBlock


class MossImageModel(Model):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)

        # a previous precompute pass left every embedding the training and the validation need on
        # disk, so the Qwen3.5 text encoder is dead weight: several gigabytes never loaded at all
        self.skip_text_encoder = precomputed_prompts_available(params, self.accelerator.process_index,
                                                               self.accelerator.num_processes)
        if self.skip_text_encoder:
            print('using the precomputed prompt embeddings, the text encoder is not loaded')

        device = f"cuda:{self.accelerator.process_index}"

        # bnb 4bit for the transformer. Gated on lora_base_model_4bit, NOT use_adamw_8bit (that flag only
        # controls the optimizer choice). bnb only quantizes when the weights reach the gpu, so a 4 bit
        # transformer is loaded there (a quarter of the size). The bf16 one is loaded on the cpu when it
        # is going to be offloaded, it is too big to be put whole on the gpu anyway.
        transformer_quant_config = None
        transformer_device = device
        if params.lora_base_model_4bit:
            transformer_quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type='nf4', llm_int8_skip_modules=SKIP_4BIT_MODULES)
        elif params.offload_to_cpu:
            transformer_device = 'cpu'

        # DiffusionPipeline.from_pretrained refuses a component whose class comes from remote code (it tries to
        # import its module by name), so the pipeline is put together from its parts
        def remote_class(name):
            return get_class_from_dynamic_module(params.pretrained_pipe_path, f'{CUSTOM_PIPELINE}.py',
                                                 class_name=name, trust_remote_code=True)

        transformer_class = remote_class('MOSSImageTransformer')
        transformer_kwargs = dict(subfolder='transformer') if params.pretrained_model_path is None else {}
        transformer = transformer_class.from_pretrained(
            params.pretrained_model_path or params.pretrained_pipe_path,
            quantization_config=transformer_quant_config,
            torch_dtype=torch.bfloat16,
            device_map=transformer_device,
            **transformer_kwargs)

        text_encoder = tokenizer = None
        if not self.skip_text_encoder:
            tokenizer = AutoTokenizer.from_pretrained(params.pretrained_pipe_path, subfolder='tokenizer')
            text_encoder_quant_config = TransformersBitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
            ) if (params.lora_base_model_4bit or params.text_encoder_4bit) else None
            text_encoder = AutoModel.from_pretrained(
                params.pretrained_pipe_path,
                subfolder='text_encoder',
                quantization_config=text_encoder_quant_config,
                dtype=torch.bfloat16,
                device_map=device).eval()

        # the vae stays in float32 like the reference pipeline: its batch norm statistics are the
        # normalization of the latents and don't survive bf16
        vae = AutoencoderKLFlux2.from_pretrained(params.pretrained_pipe_path, subfolder='vae',
                                                 torch_dtype=torch.float32)

        # the scheduler takes the training timesteps of the model: 1000 evenly spaced sigmas, the
        # shift of 5 only applies when sampling
        scheduler = remote_class('MOSSImageScheduler').from_pretrained(params.pretrained_pipe_path,
                                                                       subfolder='scheduler')
        self.pipe = remote_class('MOSSImagePipeline')(
            transformer=transformer, scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer, vae=vae)
        self.scheduler = self.pipe.scheduler
        if self.scheduler.timesteps[0].item() != 1000 or self.scheduler.timesteps[-1].item() != 1:
            raise ValueError('the scheduler is expected to hold the 1000 unshifted training timesteps, '
                             f'got {self.scheduler.timesteps[:3].tolist()}')

        # the validation samples through this scheduler, and set_timesteps replaces its timesteps by the ones
        # of the sampling: the training keeps its own copy of the 1000 it draws from
        self.train_timesteps = self.scheduler.timesteps.clone().cpu()

        self.pipe.vae.train(False)

        # the flux 2 vae normalizes its patchified latents with the statistics of its batch norm
        vae = self.pipe.vae
        self.latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).float()
        self.latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).float() + vae.config.batch_norm_eps)
        if self.pipe.text_encoder != None:
            self.pipe.text_encoder.train(False)

        # the chat template of the prompt is part of the tokens, so this is a little above the caption length
        self.max_sequence_length = 1024

        # the vae downsamples 8x and the latents are patchified 2x2 into 128 channels, so the
        # transformer works on a grid 16x smaller than the image
        self.latent_patch_size = 2

        self.model = self.pipe.transformer
        if params.offload_to_cpu:
            # bitsandbytes keeps the dequantization input of every 4 bit layer on the autograd context. With
            # the non reentrant checkpointing the forward builds a graph anyway, so each offloaded block's
            # copy would stay on the gpu until the backward: the whole model, streamed in for nothing. The
            # reentrant variant runs the forward without a graph and rebuilds one block at a time in the backward.
            enable_block_checkpointing(self.model.layers, use_reentrant=True)
        else:
            enable_block_checkpointing(self.model.layers, use_reentrant=False)

    def initialize(self):
        super().initialize()
        self.pipe.transformer = self.accelerator.unwrap_model(self.model)

    def format_embeddings(self, embeds):
        pass

    @staticmethod
    def patchify_latents(latents):
        # (B, 32, H, W) -> (B, 128, H/2, W/2), the inverse of what the pipeline does before decoding
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c, h // 2, 2, w // 2, 2)
        return latents.permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)

    def extract_latents(self, images):
        vae = self.pipe.vae
        vae.to(device=self.accelerator.device)

        # the 1024px encoder activations don't fit next to the model on the gpu. The default tile of this vae
        # is 1024px, which never tiles a 1024px image, so use smaller ones
        vae.enable_tiling()
        vae.tile_sample_min_size = 512
        vae.tile_latent_min_size = 512 // 8

        # the weights stay in float32, only the convolutions of the encoder run in bf16
        with torch.autocast('cuda', dtype=torch.bfloat16):
            latents = vae.encode(images.to(device=self.accelerator.device, dtype=vae.dtype)).latent_dist.mode()
        latents = latents.float()
        latents = self.patchify_latents(latents)
        mean = self.latents_bn_mean.to(latents.device)
        std = self.latents_bn_std.to(latents.device)
        return ((latents - mean) / std).to(torch.bfloat16)

    def extract_embeddings(self, captions):
        # move text_encoder to cuda if not already done
        self.pipe.text_encoder.to(device=self.accelerator.device)
        hidden_states, attention_mask = self.pipe.encode_prompt(captions, self.max_sequence_length)

        # only save embeddings where the mask is not zero
        return [hidden_states[idx][attention_mask[idx]] for idx in range(len(hidden_states))]

    def enable_efficient_attention(self):
        pass

    def encode_validation_prompts(self):
        # the same calls validate() makes, so the precompute pass can memoize them and free the
        # text encoder before the training starts. This matters a lot here: the Qwen3.5 text
        # encoder is several gigabytes that would otherwise sit next to the transformer.
        # when it was never loaded, these all come back from the memoized embeddings
        if self.pipe.text_encoder != None:
            self.pipe.text_encoder.to(device=self.accelerator.device)
        self.pipe.encode_prompt(prompt='')
        for prompt in self.params.validation_prompts:
            self.pipe.encode_prompt(prompt=prompt)
        return True

    def validate(self):
        params = self.params
        text_encoder = self.pipe.text_encoder
        transformer = self.pipe.transformer

        pil_to_tensor = PILToTensor()
        idx = 0
        generator = torch.Generator(device=self.accelerator.device).manual_seed(42)
        embeds = []

        text_encoder = text_encoder.to(device=self.accelerator.device)
        self.pipe.text_encoder = text_encoder
        negative_prompt_embeds, negative_prompt_embeds_mask = self.pipe.encode_prompt(prompt='')

        for prompt in tqdm.tqdm(params.validation_prompts, desc='Generating validation embeddings'):
            prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(prompt=prompt)
            embeds.append((prompt_embeds, prompt_embeds_mask))

        self.pipe.text_encoder = None
        self.pipe.transformer = self.accelerator.unwrap_model(transformer)

        # the offloaded blocks are all put on the gpu for the sampling, the weights stream otherwise
        with self.gpu_resident():
            for embed in tqdm.tqdm(embeds, desc='Generating validation images'):
                prompt_embeds, prompt_embeds_mask = embed
                image = self.pipe(
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_embeds_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_attention_mask=negative_prompt_embeds_mask,
                    guidance_scale=3.0,
                    uncond_skip_layers=[22],
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
        device = self.accelerator.device

        # the transformer drops the padded tokens of each sample itself, so the batch is only padded to
        # its longest caption and the mask tells where the text stops
        max_sequence_length = min(self.max_sequence_length, max(emb.shape[0] for emb in embeddings))
        padded_embeds = []
        masks = []
        for emb in embeddings:
            emb = emb[:max_sequence_length]
            padded_embeds.append(torch.nn.functional.pad(emb, pad=(0, 0, 0, max_sequence_length - emb.shape[0])))
            mask = torch.zeros(max_sequence_length, dtype=torch.bool)
            mask[:emb.shape[0]] = True
            masks.append(mask)

        encoder_attention_mask = torch.stack(masks).to(device=device)
        prompt_embeds = torch.stack(padded_embeds).to(device=device, dtype=torch.bfloat16)

        # the pipeline samples its noise and integrates in float32, so the noising is done there too
        latents = latents.to(device=device, dtype=torch.float32)
        noise = randn_tensor(latents.shape, device=device, dtype=torch.float32, generator=generator)

        if len(self.timesteps) != 0:
            indices = torch.tensor([random.choice(self.timesteps) for _ in range(batch_size)])
        else:
            u = compute_density_for_timestep_sampling(
                'logit_normal',
                batch_size,
                logit_mean=0,
                logit_std=1.0,
                mode_scale=1.29,
                generator=generator)
            indices = (u * len(self.train_timesteps)).long().clamp(max=len(self.train_timesteps) - 1)
        timesteps = self.train_timesteps[indices].to(device)

        # flow matching with sigma = t / 1000: sigma 1 is the pure noise, the transformer is
        # given the timestep on that 0 to 1000 scale, like the scheduler hands it to the pipeline
        sigmas = (timesteps.float() / len(self.train_timesteps)).view(-1, 1, 1, 1)
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
        target = noise - latents

        noise_pred = self.model(
            hidden_states=noisy_model_input.to(torch.bfloat16),
            timestep=timesteps.float(),
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=encoder_attention_mask,
        ).sample

        return torch.nn.functional.mse_loss(noise_pred.float(), target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)

    trainer = MossImageModel(params)
    if params.extract_features:
        trainer.pipe.transformer.cpu()
        trainer.pipe.vae.to(trainer.accelerator.device)
        trainer.pipe.text_encoder.to(trainer.accelerator.device)
        features_extractor = FeaturesExtractor(trainer, params)
        features_extractor.run()
    else:
        trainer.run()
