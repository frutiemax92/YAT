import argparse
import boto3
from botocore.config import Config
import webdataset as wds
from bucket_sampler import BucketDataset
from torch.utils.data import DataLoader, Subset
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from accelerate import Accelerator
from diffusers import SanaTransformer2DModel, DPMSolverMultistepScheduler, AutoencoderDC
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter
from diffusers import SanaPAGPipeline
import torch
from tqdm import tqdm
from torchvision.transforms import PILToTensor
from diffusers.utils.torch_utils import randn_tensor
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

    output = pipe.vae.encode(images.to(device=vae.device, dtype=vae.dtype))
    return output.latent

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
            num_inference_steps=20,
            generator=generator,
        )[0]
        logger.add_image(f'validation/{idx}/{prompt}', pil_to_tensor(image[0]), global_step)
        idx = idx + 1
    
    # save the transformer
    transformer.save_pretrained(f'{global_step}')

def optimize(logger : SummaryWriter,
             global_step: int,
             pipe : SanaPAGPipeline,
             latents : torch.Tensor,
             embeddings : torch.Tensor,
             attention_mask : torch.Tensor,
             optimizer : AdamW,
             accelerator : Accelerator):
    loss_fn = torch.nn.MSELoss()
    noise = randn_tensor(latents.shape, device=pipe.device, dtype=pipe.dtype)
    
    scheduler = pipe.scheduler
    timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=pipe.device)
    timesteps = timestep.expand(latents.shape[0])
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    noisy_latents = scheduler.scale_model_input(noisy_latents, timesteps)

    transformer = pipe.transformer
    with accelerator.accumulate(transformer):
        noise_pred = transformer(noisy_latents.to(dtype=transformer.dtype),
                                 encoder_hidden_states=embeddings,
                                 timestep=timesteps,
                                 encoder_attention_mask=attention_mask).sample
        loss = loss_fn(noise_pred.to(dtype=noise.dtype), noise)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
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

    if urls == None:
        # get the urls from the cloudflare bucket with the keys
        session = boto3.Session(
            aws_access_key_id=r2_access_key,
            aws_secret_access_key=r2_secret_key
        )
        config = Config(signature_version='s3v4')
        s3_client = session.client('s3', endpoint_url=r2_endpoint, config=config)

        # urls with 1 week expiration
        urls = [s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': r2_bucket_name, 'Key':tar_file},
            ExpiresIn=604800
        ) for tar_file in r2_tar_files]

    # build the dataset
    dataset = (
        wds.WebDataset(urls, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker)
        .shuffle(1000)  # Shuffle across all shards
        .decode("pil")  # Decode images as PIL objects
        .to_tuple("jpg", "txt")  # Return image and text
    )

    # dataset = wds.DataPipeline(
    #     wds.SimpleShardList(urls),
    #     wds.detshuffle(),
    #     wds.split_by_node,
    #     wds.split_by_worker,
    #     wds.decode('pil'),
    #     wds.to_tuple('jpg', 'txt')
    # )

    pipe = SanaPAGPipeline.from_pretrained(pretrained_model_path)

    # SANA transformer
    transformer = pipe.transformer

    # scheduler
    scheduler = pipe.scheduler

    # vae
    vae = pipe.vae
    vae = vae.to(dtype=torch.bfloat16)
    vae.train(False)

    # tokenizer
    tokenizer = pipe.tokenizer

    # text encoder
    text_encoder = pipe.text_encoder
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
    accelerator = Accelerator()
    transformer, scheduler, vae, tokenizer, text_encoder, optimizer = accelerator.prepare(
        transformer, scheduler, vae, tokenizer, text_encoder, optimizer
    )
    global_step = 0

    if accelerator.is_main_process:
        logger = SummaryWriter()
    
    for epoch in range(num_epochs):
        for images, captions in tqdm(dataloader):
            images = torch.squeeze(images, dim=0)
            if global_step % num_steps_per_validation == 0:
                if accelerator.is_main_process:
                    with torch.no_grad():
                        validate(logger,
                                global_step,
                                pipe,
                                validation_prompts)
                accelerator.wait_for_everyone()

            with torch.no_grad():
                embeddings, attention_mask = extract_embeddings(captions, pipe)
                latents = extract_latents(images, pipe)
            
            optimize(logger,
                     global_step,
                     pipe,
                     latents,
                     embeddings,
                     attention_mask,
                     optimizer,
                     accelerator)
            global_step = global_step + 1