import argparse
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN
from diffusers.training_utils import compute_density_for_timestep_sampling
from diffusers import UNet2DConditionModel, DDPMScheduler
from diffusers import StableDiffusionXLPipeline
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from common.training_parameters_reader import TrainingParameters
from common.trainer import Model
from common import lw_controlnet
from common.features_extractor import FeaturesExtractor
from utils.compress_caption import compress_caption

class SDXLModel(Model):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)
        
        # read the single safetensors to build the sd1.5 pipeline to start from the very good finetunes from civitai
        if params.pretrained_pipe_single_file != None:
            self.pipe = StableDiffusionXLPipeline.from_single_file(
                params.pretrained_pipe_single_file,
                torch_dtype="float16")
        else:
            # useful to start from a diffusers pipeline directly
            self.pipe = StableDiffusionXLPipeline.from_pretrained(params.pretrained_pipe_path)
        if params.pretrained_model_path != None:
            self.pipe.unet = UNet2DConditionModel.from_pretrained(params.pretrained_model_path)
        
        config = self.pipe.scheduler.config
        self.scheduler = DDPMScheduler.from_config(config)
        self.pipe.vae.train(False)
        self.pipe.text_encoder.train(False)
        self.pipe.text_encoder_2.train(False)
        self.aspect_ratios = ASPECT_RATIO_1024_BIN

        self.pipe.unet.to(torch.bfloat16)
        self.pipe.text_encoder.to(torch.bfloat16)
        self.pipe.text_encoder_2.to(torch.bfloat16)
        self.pipe.vae.to(torch.bfloat16)

        self.model = self.pipe.unet
        self.model.enable_gradient_checkpointing()
    
    def initialize(self):
        super().initialize()
        self.pipe.unet = self.model
        unet = self.pipe.unet
        vae = self.pipe.vae
        text_encoder = self.pipe.text_encoder
    
    def format_embeddings(self, embeds):
        pass
    
    def extract_latents(self, images):
        self.pipe.vae.to(device=self.accelerator.device)
        output = self.pipe.vae.encode(images.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)).latent_dist.sample()
        return output * self.pipe.vae.config.scaling_factor

    def extract_embeddings(self, captions):
        # move text_encoder to cuda if not already done
        self.pipe.text_encoder = self.pipe.text_encoder.to(device=self.accelerator.device)
        self.pipe.text_encoder_2.to(device=self.accelerator.device)

        # we need to compress the caption for SDXL
        new_captions = [compress_caption(caption) if caption != '' else '' for caption in captions]
        
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
        self.pipe.encode_prompt(prompt=new_captions,
                                do_classifier_free_guidance=False,
                                device=self.accelerator.device,
                                num_images_per_prompt=1)
        return [(prompt_embeds[idx], pooled_prompt_embeds[idx]) for idx in range(len(prompt_embeds))]
    
    def validate(self):
        params = self.params

        pil_to_tensor = PILToTensor()
        idx = 0
        generator=torch.Generator(device="cuda").manual_seed(42)
        latents = []
        embeds = []
        for prompt in tqdm.tqdm(params.validation_prompts, desc='Generating validation embeddings'):
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                self.pipe.encode_prompt(
                    prompt,
                    negative_prompt='illustration',
                    device=self.accelerator.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True)
            embeds.append((prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds))
        
        self.pipe.unet = self.accelerator.unwrap_model(self.model).eval()

        # lightweight controlnet: one reference per prompt, encoded before the sampling starts so
        # the frozen dinov2 and the unet never share the vram. None when the run is a plain lora,
        # or when no validation reference was configured, and the patch then behaves as one.
        references = self.validation_references()
        if references != None:
            self.free_reference_encoder()

        # what the pipeline would default to on its own, and what a run with no reference keeps
        default_size = getattr(self.pipe, 'default_sample_size', 128) * self.pipe.vae_scale_factor

        def generation_size(reference):
            """The (height, width) to generate this validation image at.

            A src/ref pair is aligned by construction, so a reference only describes the frame it
            was drawn for. Both sides of the reference attention carry normalized positions, which
            is what lets one resolution transfer to another - but it also means a portrait mask
            generated into a square frame is mapped there by relative position, i.e. stretched by
            the ratio between the two. Nothing tells the reference path the shape of the output
            frame, so it cannot compensate, and the validation image then shows a shape the mask
            never asked for.

            So the size is taken from the reference's own aspect ratio, snapped to the same bucket
            table the training batches are drawn from. Without a reference there is nothing to align
            to and the pipeline's own default stands.
            """
            if reference is None or self.aspect_ratios is None:
                return default_size, default_size
            # ReferenceFeatures carries width / height; the bucket table is keyed on height / width
            width_over_height = max(float(reference.aspect_ratio.reshape(-1)[0]), 1e-6)
            height, width = self.aspect_ratios[self.find_closest_ratio(1.0 / width_over_height)]
            return int(height), int(width)

        for index, embed in enumerate(tqdm.tqdm(embeds, desc='Generating validation latents')):
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = embed
            reference = references[index] if references != None else None
            if reference != None:
                reference = lw_controlnet.features_to(reference, self.accelerator.device)

            height, width = generation_size(reference)
            # the conditioning is set once for the whole denoising loop, the negative pass of the
            # pipeline included: dropping the reference there would be reference guidance, which is
            # a choice of its own and not what the training optimizes
            with self.reference_context(reference):
                latent = self.pipe(
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds.to(device=self.model.device),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device=self.model.device),
                    negative_prompt_embeds=negative_prompt_embeds.to(device=self.model.device),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device=self.model.device),
                    guidance_scale=5.0,
                    num_inference_steps=20,
                    generator=generator,
                    height=height,
                    width=width,
                    output_type='latent',
                    callback_on_step_end=self.validation_step_callback,
                )[0]
            latents.append(latent.to(device=self.pipe.vae.device))

        idx = 0
        print(self.pipe.vae.device)
        for latent in tqdm.tqdm(latents, desc='Decoding validation latents'):
            prompt = params.validation_prompts[idx]
            latent = latent.to(self.pipe.vae.dtype)
            image = self.pipe.vae.decode(latent / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = self.pipe.image_processor.postprocess(image)
            self.logger.add_image(f'validation/{idx}/{prompt}', pil_to_tensor(image[0]), self.global_step)
            idx = idx + 1

    def optimize(self, ratio, latents, embeddings, repa_features, generator):
        params = self.params
        batch_size = params.batch_size
        self.model.to(self.accelerator.device)

        # pad the embeds to 512 tokens and generate the corresponding mask
        embeds = []
        pooled_embeds = []
        masks = []
        for emb in embeddings:
            prompt_embeds, pooled_prompt_embeds = emb
            embeds.append(torch.squeeze(prompt_embeds, 0))
            pooled_embeds.append(torch.squeeze(pooled_prompt_embeds, 0))
            mask = torch.zeros(77, dtype=torch.long, device=prompt_embeds.device)
            mask[:prompt_embeds.shape[1]] = 1
            masks.append(mask)
        
        embeds = torch.stack(embeds).to(device=self.accelerator.device, dtype=torch.bfloat16)
        attention_mask = torch.stack(masks).to(device=self.accelerator.device, dtype=torch.bfloat16)
        pooled_embeds = torch.stack(pooled_embeds).to(device=self.accelerator.device, dtype=torch.bfloat16)
        latents = latents.to(device=self.accelerator.device, dtype=torch.bfloat16)

        loss_fn = torch.nn.MSELoss()
        noise = randn_tensor(latents.shape, device=self.accelerator.device, dtype=torch.bfloat16)
        timesteps = self.get_timesteps(batch_size)
        noisy_model_input = self.scheduler.add_noise(latents.to(self.accelerator.device), noise, timesteps)

        original_size = self.aspect_ratios[str(ratio)]
        original_size = (int(original_size[0]), int(original_size[1]))
        target_size = (latents.shape[-2], latents.shape[-1])
        crop_coords = (0, 0)
        time_ids = torch.tensor(
            [*original_size, *crop_coords, *target_size],
            device=self.accelerator.device,
            dtype=torch.bfloat16
        ).unsqueeze(0).expand(pooled_embeds.shape[0], -1)

        conds = {
            'text_embeds' : pooled_embeds,
            'time_ids' : time_ids
            }
        # Keep everything in bfloat16
        noise_pred = self.model(
            noisy_model_input,
            encoder_hidden_states=embeds,
            timestep=timesteps,
            added_cond_kwargs=conds
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

    trainer = SDXLModel(params)
    if params.extract_features:
        trainer.pipe.unet.cpu()
        trainer.pipe.vae.to(trainer.accelerator.device)
        trainer.pipe.text_encoder.to(trainer.accelerator.device)
        features_extractor = FeaturesExtractor(trainer, params)
        features_extractor.run()
    else:
        trainer.run()