import argparse
import boto3
from botocore.config import Config
import webdataset as wds
from bucket_sampler import BucketDataset
from torch.utils.data import DataLoader, Subset
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from accelerate import Accelerator, DataLoaderConfiguration
from diffusers import SanaTransformer2DModel, DDPMScheduler, AutoencoderDC, FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers.training_utils import compute_density_for_timestep_sampling
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter
from diffusers import SanaPAGPipeline
import torch
import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from cloudflare import get_secured_urls
import random
import datetime
from webdataset.utils import pytorch_worker_info
import gc
from training_parameters_reader import TrainingParameters

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def extract_embeddings(pipe, captions : list[str]):
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
        pipe.encode_prompt(captions, do_classifier_free_guidance=False)
    return prompt_embeds, prompt_attention_mask

def extract_latents(pipe, images : torch.Tensor):
    image_processor = pipe.image_processor
    images = image_processor.preprocess(images)
    flush()

    output = pipe.vae.encode(images.to(device=pipe.vae.device, dtype=pipe.vae.dtype)).latent
    return output * pipe.vae.config.scaling_factor
    
def validate(logger : SummaryWriter,
             global_step: int,
             pipe : SanaPAGPipeline,
             validation_prompts : list[str]):
    pil_to_tensor = PILToTensor()
    idx = 0
    generator=torch.Generator(device="cuda").manual_seed(42)
    for prompt in tqdm.tqdm(validation_prompts, desc='Generating validation images'):
        image = pipe(
            prompt=prompt,
            guidance_scale=5.0,
            pag_scale=2.0,
            num_inference_steps=20,
            generator=generator,
        )[0]
        logger.add_image(f'validation/{idx}/{prompt}', pil_to_tensor(image[0]), global_step)
        idx = idx + 1
    
    # save the transformer
    pipe.transformer.save_pretrained(f'{global_step}')
    
def optimize(logger : SummaryWriter,
             global_step: int,
             pipe : SanaPAGPipeline,
             latents : torch.Tensor,
             embeddings : torch.Tensor,
             attention_mask : torch.Tensor,
             scheduler : FlowMatchEulerDiscreteScheduler,
             optimizer : AdamW,
             accelerator : Accelerator):
    loss_fn = torch.nn.MSELoss()
    noise = randn_tensor(latents.shape, device=pipe.device, dtype=pipe.dtype)

    u = compute_density_for_timestep_sampling('logit_normal', batch_size, logit_mean=0, logit_std=1.0, mode_scale=1.29)
    indices = (u * scheduler.config.num_train_timesteps).long()
    timesteps = scheduler.timesteps[indices].to(latents.device)
    noisy_model_input = scheduler.scale_noise(latents, timesteps, noise)

    transformer = pipe.transformer
    flush()
    with accelerator.accumulate(transformer):
        noise_pred = transformer(noisy_model_input.to(dtype=transformer.dtype),
                                 encoder_hidden_states=embeddings.to(dtype=transformer.dtype),
                                 timestep=timesteps,
                                 encoder_attention_mask=attention_mask.to(dtype=transformer.dtype)).sample
        target = noise - latents
        loss = loss_fn(noise_pred.to(dtype=noise.dtype), target)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

    if logger != None:
        logger.add_scalar('train/loss', loss.detach().item(), global_step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)

    if params.urls == None:
        urls = get_secured_urls(params.r2_access_key,
                                params.r2_secret_key,
                                params.r2_endpoint,
                                params.r2_bucket_name,
                                params.r2_tar_files)

    pipe = SanaPAGPipeline.from_pretrained(params.pretrained_pipe_path, pag_applied_layers=[f"transformer_blocks.{i}" for i in range(8, 9)])
    if params.pretrained_model_path != None:
        transformer = SanaTransformer2DModel.from_pretrained(params.pretrained_model_path)
        pipe.transformer = transformer
    if params.bfloat16:
        pipe = pipe.to(torch.bfloat16)

    # SANA transformer
    transformer = pipe.transformer
    if params.bfloat16:
        transformer = transformer.to(torch.bfloat16)

    # scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(params.pretrained_pipe_path, subfolder='scheduler')

    # vae
    vae = pipe.vae
    if params.bfloat16:
        vae = vae.to(dtype=torch.bfloat16)
    vae.train(False)

    # tokenizer
    tokenizer = pipe.tokenizer

    # text encoder
    text_encoder = pipe.text_encoder
    
    if params.bfloat16:
        text_encoder = text_encoder.to(dtype=torch.bfloat16)
    text_encoder.train(False)

    vae_compression = 32
    resolution = transformer.sample_size * vae_compression
    if resolution == 256:
        aspect_ratio = ASPECT_RATIO_256_BIN
    elif resolution == 512:
        aspect_ratio = ASPECT_RATIO_512_BIN
    elif resolution == 1024:
        aspect_ratio = ASPECT_RATIO_1024_BIN
    else:
        aspect_ratio = ASPECT_RATIO_2048_BIN
    
    # optimizer
    optimizer = AdamW(transformer.parameters(), lr=params.learning_rate)

    # multi-gpu training
    #dataloader_config = DataLoaderConfiguration(dispatch_batches=True)
    accelerator = Accelerator(gradient_accumulation_steps=params.gradient_accumulation_steps)
    # build the dataset
    def split_only_on_main(src, group=None):
        yield from src

    datasets = [
        wds.WebDataset(url, shardshuffle=1000, handler=wds.warn_and_continue, nodesplitter=split_only_on_main)
        .decode("pil", handler=wds.warn_and_continue)  # Decode images as PIL objects
        .to_tuple(["jpg", 'jpeg'], "txt", handler=wds.warn_and_continue)  # Return image and text
    for url in urls]
    mix = wds.RandomMix(datasets)
    bucket_dataset = BucketDataset(mix,
                                params.batch_size,
                                aspect_ratio,
                                accelerator,
                                pipe,
                                extract_latents_handler=extract_latents,
                                extract_embeddings_handler=extract_embeddings)
    dataloader = DataLoader(bucket_dataset, batch_size=None)
    dataloader = accelerator.prepare_data_loader(dataloader)
    
    transformer, scheduler, vae, tokenizer, text_encoder, optimizer = accelerator.prepare(
        transformer, scheduler, vae, tokenizer, text_encoder, optimizer
    )
    
    global_step = 0

    if accelerator.is_main_process:
        logger = SummaryWriter()
    else:
        logger = None
    
    global_step = 0
    progress_bar = tqdm.tqdm(total=params.steps, desc='Num Steps')
    for latents, embeddings, attention_mask in dataloader:
        if global_step % params.num_steps_per_validation == 0:
            if accelerator.is_main_process:
                with torch.no_grad():
                    validate(logger,
                            global_step,
                            accelerator.unwrap_model(pipe),
                            params.validation_prompts)
        
        optimize(logger,
                    global_step,
                    pipe,
                    latents,
                    embeddings,
                    attention_mask,
                    scheduler,
                    optimizer,
                    accelerator)
        global_step = global_step + 1
        progress_bar.update(1)
        

    # final validation
    if accelerator.is_main_process:
        validate(logger,
            global_step,
            accelerator.unwrap_model(pipe),
            params.validation_prompts)