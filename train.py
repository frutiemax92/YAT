import argparse
import boto3
from botocore.config import Config
import webdataset as wds
from bucket_sampler import BucketDataset
from torch.utils.data import DataLoader, Subset
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from accelerate import Accelerator
from diffusers import SanaTransformer2DModel, DDPMScheduler, AutoencoderDC, FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers.training_utils import compute_density_for_timestep_sampling
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter
from diffusers import SanaPAGPipeline
import torch
from tqdm import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
from cloudflare import get_secured_urls
import random
import gc

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def extract_embeddings(captions : list[str],
                       pipe : SanaPAGPipeline):
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
        pipe.encode_prompt(captions, do_classifier_free_guidance=False)
    return prompt_embeds, prompt_attention_mask

def extract_latents(images : torch.Tensor,
                    pipe : SanaPAGPipeline):
    image_processor = pipe.image_processor
    images = image_processor.pil_to_numpy(images)
    images = torch.tensor(images, device=pipe.device, dtype=pipe.dtype)
    images = image_processor.preprocess(torch.squeeze(images, dim=0))
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
             scheduler : DPMSolverMultistepScheduler,
             optimizer : AdamW,
             accelerator : Accelerator):
    loss_fn = torch.nn.MSELoss()
    noise = randn_tensor(latents.shape, device=pipe.device, dtype=pipe.dtype)
    timestep = torch.tensor(random.choice(scheduler.timesteps)).to(device=latents.device)
    timesteps = timestep.expand(batch_size)
    noisy_model_input = scheduler.add_noise(latents, noise, timesteps)

    transformer = pipe.transformer
    with accelerator.accumulate(transformer):
        noise_pred = transformer(noisy_model_input.to(dtype=transformer.dtype),
                                 encoder_hidden_states=embeddings,
                                 timestep=timesteps,
                                 encoder_attention_mask=attention_mask).sample
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
    parser.add_argument('--pretrained_model_path',
                        required=False,
                        type=str,
                        default='Efficient-Large-Model/Sana_600M_512px_diffusers')
    parser.add_argument('--learning_rate', required=False, type=float, default=1e-5)
    parser.add_argument('--num_epochs', required=False, type=int, default=5)
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
    pretrained_model_path = args.pretrained_model_path
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
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
    # build the dataset
    dataset = (
        wds.WebDataset(urls, handler=wds.warn_and_continue, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker)
        .shuffle(1000, handler=wds.warn_and_continue)  # Shuffle across all shards
        .decode("pil", handler=wds.warn_and_continue)  # Decode images as PIL objects
        .to_tuple("jpg", "txt", handler=wds.warn_and_continue)  # Return image and text
    )

    pipe = SanaPAGPipeline.from_pretrained(pretrained_model_path).to(torch.bfloat16)

    # SANA transformer
    transformer = pipe.transformer
    if use_bfloat16:
        transformer = transformer.to(torch.bfloat16)

    # scheduler
    scheduler = DPMSolverMultistepScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')

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
    bucket_dataset = BucketDataset(dataset, batch_size, aspect_ratio)

    # batch collater that applies a transformation
    def collate_fn(samples):
        # Unpack individual samples into images and captions
        images = torch.stack([sample[0][0] for sample in samples])  # List of PIL images
        captions = [sample[1] for sample in samples]  # List of strings (captions)
        return images, captions
    
    # optimizer
    optimizer = AdamW(transformer.parameters(), lr=learning_rate)
    dataloader = DataLoader(bucket_dataset, batch_size=None)

    # multi-gpu training
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    transformer, scheduler, vae, tokenizer, text_encoder, optimizer = accelerator.prepare(
        transformer, scheduler, vae, tokenizer, text_encoder, optimizer
    )
    global_step = 0

    if accelerator.is_main_process:
        logger = SummaryWriter()
    else:
        logger = None
    
    for epoch in tqdm(range(num_epochs), desc='Num epochs'):
        for images, captions in tqdm(dataloader):
            images = torch.squeeze(images, dim=0)
            if global_step % num_steps_per_validation == 0:
                if accelerator.is_main_process:
                    with torch.no_grad():
                        validate(logger,
                                global_step,
                                accelerator.unwrap_model(pipe),
                                validation_prompts)

            with torch.no_grad():
                embeddings, attention_mask = extract_embeddings(captions, accelerator.unwrap_model(pipe))
                latents = extract_latents(images, accelerator.unwrap_model(pipe))
            
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
            flush()
    accelerator.wait_for_everyone()

    # final validation
    validate(logger,
        global_step,
        accelerator.unwrap_model(pipe),
        validation_prompts)
    
    if output_repo != None:
        transformer.push_to_hub(output_repo)