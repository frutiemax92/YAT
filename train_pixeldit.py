import argparse

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(ROOT / "PixelDiT" / "t2i")
)

# Import trainer early to register the model
try:
    from diffusion.model.trainer import PixDiTTrainer  # noqa: F401
except Exception as e:
    print(f"Warning: Could not import PixDiTTrainer: {e}")

from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN
from diffusers import DPMSolverMultistepScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from common.training_parameters_reader import TrainingParameters
from common.trainer import Model
from common.features_extractor import FeaturesExtractor
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.utils.config import model_init_config
from PixelDiT.t2i.inference import PixelDiTInference
import pyrallis
from huggingface_hub import hf_hub_download
from torch.utils.checkpoint import checkpoint

class PixelDITModel(Model):
    def __init__(self, params : TrainingParameters):
        super().__init__(params)
        
        CONFIG_PATH = "PixelDiT/t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml" 
        config = pyrallis.parse(config_class=PixelDiTInference, config_path=CONFIG_PATH)
        self.tokenizer, self.text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name, device=self.accelerator.device)

        model_kwargs = model_init_config(config, latent_size=1024)
        self.model = build_model(config.model.model, use_fp32_attention=config.model.get("fp32_attention", False), **model_kwargs).to(self.accelerator.device, dtype=torch.bfloat16)

        repo = 'nvidia/PixelDiT-1300M-1024px'
        if params.pretrained_model_path != None:
            repo = params.pretrained_model_path
        checkpoint_path = hf_hub_download(repo_id=repo, filename='pixeldit_t2i_v1.pth', local_dir='./checkpoints')
        state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        if "pos_embed" in state_dict["state_dict"]:
            del state_dict["state_dict"]["pos_embed"]
        self.model.load_state_dict(state_dict["state_dict"], strict=False)
        
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            prediction_type="flow_prediction",
            algorithm_type="dpmsolver++",
            use_flow_sigmas=True,  # Enable flow matching mode
            flow_shift=config.scheduler.flow_shift,  # Use flow_shift from config
        )
        self.inference_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            prediction_type="flow_prediction",
            algorithm_type="dpmsolver++",
            use_flow_sigmas=True,  # Enable flow matching mode
            flow_shift=config.scheduler.flow_shift,  # Use flow_shift from config
        )
        self.text_encoder.train(False)
        self.aspect_ratios = ASPECT_RATIO_1024_BIN
    
    def initialize(self):
        super().initialize()
        #self.pipe = self.pipe.to(self.accelerator.device)
        #self.pipe.transformer = self.accelerator.unwrap_model(self.model)
        #transformer = self.pipe.transformer
        #vae = self.pipe.vae
        #text_encoder = self.pipe.text_encoder
        
        self.model.to(self.accelerator.device)
        #vae.cpu()
        #text_encoder.cpu()
    
    def format_embeddings(self, embeds):
        pass
    
    def extract_latents(self, images):
        return images

    def extract_embeddings(self, captions):
        # move text_encoder to cuda if not already done
        self.text_encoder.to(device=self.accelerator.device)

        caption_tokens = self.tokenizer(captions, max_length=300, padding="max_length", truncation=True, return_tensors="pt").to(self.accelerator.device)
        caption_embs = self.text_encoder(caption_tokens.input_ids, caption_tokens.attention_mask)[0][:, None][:, :, [0] + list(range(-300 + 1, 0))]
        return caption_embs.squeeze(0).to(device=self.accelerator.device)
    
    def enable_efficient_attention(self):
        pass
    
    def validate(self):
        HEIGHT = 1024
        WIDTH = 1024
        CFG_SCALE = 4.0
        prompts = self.params.validation_prompts

        idx = 0
        for prompt in prompts:
            caption_embs = self.extract_embeddings(prompt)
            z = torch.randn(
                1,
                3,
                HEIGHT,
                WIDTH,
                device=self.accelerator.device,
            )
            self.inference_scheduler.set_timesteps(20)
            x = z
            for i, t in tqdm.tqdm(enumerate(self.inference_scheduler.timesteps)):
                # The model expects 'y' as the conditioning argument (caption_embs)
                #model_output = model(x, t, y=caption_embs, **model_kwargs)
                model_output = self.model(x, t.to(self.accelerator.device), y=caption_embs)
                if isinstance(model_output, dict):
                    # Try to get the most likely tensor output
                    model_output = model_output.get('sample', list(model_output.values())[0])
                
                # Apply classifier-free guidance
                #model_output_uncond = model(x, t, y=null_y, **model_kwargs)
                model_output_uncond = self.model(x, t, y=self.empty_embeddings)
                if isinstance(model_output_uncond, dict):
                    model_output_uncond = model_output_uncond.get('sample', list(model_output_uncond.values())[0])
                model_output = model_output_uncond + CFG_SCALE * (model_output - model_output_uncond)
                
                # Scheduler step
                step_output = self.inference_scheduler.step(model_output, t, x)
                x = step_output.prev_sample if hasattr(step_output, 'prev_sample') else step_output

            samples = x
            # PixelDiT outputs are in [-1, 1], so normalize before logging.
            image = samples[0].add(1.0).div(2.0).clamp(0.0, 1.0)
            self.logger.add_image(
                f'validation/{idx}/{prompt}',
                image,
                self.global_step,
            )
            idx = idx + 1
    
    def optimize(self, ratio, latents, embeddings):
        caption_embs = torch.stack(embeddings)
        latents = latents.to(device=self.accelerator.device, dtype=torch.bfloat16)
        batch_size = latents.shape[0]

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

        # Keep everything in bfloat16
        noise_pred = checkpoint(self.model, noisy_model_input, timesteps, caption_embs, use_reentrant=False)['x']
        target = noise - latents
        loss = loss_fn(noise_pred.float(), target.float())
        return loss  # Already in bfloat16 since inputs were bfloat16
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)

    trainer = PixelDITModel(params)
    trainer.run()