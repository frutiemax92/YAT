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

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def validate(logger : SummaryWriter,
             global_step: int,
             pipe : SanaPAGPipeline,
             validation_prompts : list[str]):
    pil_to_tensor = PILToTensor()
    idx = 0
    generator=torch.Generator(device="cuda").manual_seed(42)
    for prompt in tqdm(validation_prompts, desc='Generating validation images'):
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
    parser.add_argument('--r2_endpoint', required=True, type=str)
    parser.add_argument('--r2_access_key', required=True, type=str)
    parser.add_argument('--r2_secret_key', required=True, type=str)
    parser.add_argument('--r2_bucket_name', required=True, type=str)
    parser.add_argument('--r2_tar_files', nargs='+', required=True, type=str)

    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--num_processes', required=False, type=int, default=1)
    parser.add_argument('--pretrained_pipe_path',
                        required=False,
                        type=str,
                        default='Efficient-Large-Model/Sana_600M_512px_diffusers')
    parser.add_argument('--pretrained_transformer_path', required=False, type=str, default=None)
    parser.add_argument('--learning_rate', required=False, type=float, default=1e-5)
    parser.add_argument('--steps', required=True, type=int)
    parser.add_argument('--num_steps_per_validation', required=False, type=int, default=5000)
    parser.add_argument('--validation_prompts', required=True, nargs='+', type=str)
    parser.add_argument('--urls', required=False, nargs='+', type=str, default=None)
    parser.add_argument('-v', '--bfloat16', action='store_true', required=False)
    parser.add_argument('--gradient_accumulation_steps', required=False, type=int, default=1)
    parser.add_argument('--output_repo', required=False, type=str, default=None)
    args = parser.parse_args()

    # cloudflare related arguments
    r2_endpoint = args.r2_endpoint
    r2_access_key = args.r2_access_key
    r2_secret_key = args.r2_secret_key
    r2_bucket_name = args.r2_bucket_name
    r2_tar_files = args.r2_tar_files
    urls = args.urls

    # training parameters
    batch_size = args.batch_size
    num_processes = args.num_processes
    pretrained_pipe_path = args.pretrained_pipe_path
    learning_rate = args.learning_rate
    num_steps = args.steps

    pretrained_transformer_path = args.pretrained_transformer_path

    num_steps_per_validation = args.num_steps_per_validation
    validation_prompts = args.validation_prompts
    use_bfloat16 = args.bfloat16
    gradient_accumulation_steps = args.gradient_accumulation_steps

    output_repo = args.output_repo

    if urls == None:
        urls = get_secured_urls(r2_access_key,
                                r2_secret_key,
                                r2_endpoint,
                                r2_bucket_name,
                                r2_tar_files)

    pipe = SanaPAGPipeline.from_pretrained(pretrained_pipe_path)
    if pretrained_transformer_path != None:
        transformer = SanaTransformer2DModel.from_pretrained(pretrained_transformer_path)
        pipe.transformer = transformer
    if use_bfloat16:
        pipe = pipe.to(torch.bfloat16)

    # SANA transformer
    transformer = pipe.transformer
    if use_bfloat16:
        transformer = transformer.to(torch.bfloat16)

    # scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_pipe_path, subfolder='scheduler')

    # vae
    vae = pipe.vae
    if use_bfloat16:
        vae = vae.to(dtype=torch.bfloat16)
    vae.train(False)

    # tokenizer
    tokenizer = pipe.tokenizer

    # text encoder
    text_encoder = pipe.text_encoder
    
    if use_bfloat16:
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

    # batch collater that applies a transformation
    def collate_fn(samples):
        # Unpack individual samples into images and captions
        images = torch.stack([sample[0][0] for sample in samples])  # List of PIL images
        captions = [sample[1] for sample in samples]  # List of strings (captions)
        return images, captions
    
    # optimizer
    optimizer = AdamW(transformer.parameters(), lr=learning_rate)

    # multi-gpu training
    #dataloader_config = DataLoaderConfiguration(dispatch_batches=True)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    # build the dataset
    def split_only_on_main(src, group=None):
        """Split the input sequence by PyTorch distributed rank."""
        rank, world_size, worker, num_workers = pytorch_worker_info(group=group)
        if rank == 0:
            yield from src
        else:
            yield None

    datasets = [
        wds.WebDataset(url, shardshuffle=True, handler=wds.warn_and_continue, nodesplitter=split_only_on_main)
        .shuffle(1000)
        .decode("pil", handler=wds.warn_and_continue)  # Decode images as PIL objects
        .to_tuple(["jpg", 'jpeg'], "txt", handler=wds.warn_and_continue)  # Return image and text
    for url in urls]
    mix = wds.RandomMix(datasets)
    bucket_dataset = BucketDataset(mix,
                                batch_size,
                                aspect_ratio,
                                accelerator,
                                pipe)
    dataloader = DataLoader(bucket_dataset, batch_size=batch_size)
    dataloader = accelerator.prepare_data_loader(dataloader, device_placement=True)
    
    transformer, scheduler, vae, tokenizer, text_encoder, optimizer = accelerator.prepare(
        transformer, scheduler, vae, tokenizer, text_encoder, optimizer
    )
    
    global_step = 0

    if accelerator.is_main_process:
        logger = SummaryWriter()
    else:
        logger = None
    
    global_step = 0
    progress_bar = tqdm.tqdm(total=num_steps, desc='Num Steps:')
    for latents, embeddings, attention_mask in tqdm(dataloader):
        latents = torch.squeeze(latents, dim=1)
        embeddings = torch.squeeze(embeddings, dim=1)
        if global_step % num_steps_per_validation == 0:
            if accelerator.is_main_process:
                with torch.no_grad():
                    validate(logger,
                            global_step,
                            accelerator.unwrap_model(pipe),
                            validation_prompts)
        
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
            validation_prompts)
        
        if output_repo != None:
            transformer.push_to_hub(output_repo)