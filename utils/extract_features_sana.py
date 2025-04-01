import webdataset as wds
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.training_parameters_reader import TrainingParameters
import argparse
from common.cloudflare import get_secured_urls
from accelerate import DataLoaderConfiguration, Accelerator
from diffusers import SanaPipeline, SanaPAGPipeline
from torch.utils.data import DataLoader
import torch
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_256_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from torchvision.transforms import Resize
import tqdm
from common.bucket_sampler import RoundRobinMix
from torch.utils.data import DataLoader
import gc
from common.bucket_sampler_cache import DataExtractor
import itertools

class DatasetFetcher(torch.utils.data.IterableDataset):
    def __init__(self, dataset, process_index, num_processes):
        self.num_processes = num_processes
        self.process_index = process_index
        self.dataset = dataset
    
    def __iter__(self):
        idx = 0
        for img, caption in dataset:
            yield img, caption

def find_closest_ratio(ratio, aspect_ratios):
    # find the closest ratio from the aspect ratios table
    min_distance = 100
    target_ratio = 0.6
    for r in aspect_ratios.keys():
        distance = abs(float(r) - ratio)
        if min_distance > distance:
            target_ratio = r
            min_distance = distance
    
    # Return the result as a tuple (target_ratio, idx)
    return str(target_ratio)

def cache_latents_embeddings(img, caption, cache_idx, aspect_ratios, process_index):
    img = torch.squeeze(img)

    # find the aspect ratio
    width = img.shape[-1]
    height = img.shape[-2]
    ratio = height / width
    closest_ratio = find_closest_ratio(ratio, aspect_ratios)
    height_target = int(aspect_ratios[closest_ratio][0])
    width_target = int(aspect_ratios[closest_ratio][1])

    # resize the image
    resize_transform = Resize((height_target, width_target))
    img = resize_transform(img.cpu())

    # compute the latents and embeddings
    with torch.no_grad():
        latent = extract_latents(img.to(accelerator.device), pipe, accelerator)
        embedding = extract_embeddings(caption, pipe, accelerator)

    # save on the disk
    embedding = [emb.cpu() for emb in embedding]
    to_save = (closest_ratio, latent.cpu(), embedding)
    torch.save(to_save, f'cache/{cache_idx}.npy')

def extract_latents(images, pipe : SanaPAGPipeline, accelerator):
    image_processor = pipe.image_processor
    images = image_processor.preprocess(images)
    vae = pipe.vae

    # move vae to cuda if it's not already done
    vae = vae.to(device=accelerator.device)
    output = pipe.vae.encode(images.to(device=pipe.vae.device, dtype=pipe.vae.dtype)).latent
    return output * pipe.vae.config.scaling_factor

def extract_embeddings(captions, pipe : SanaPAGPipeline, accelerator):
    # move text_encoder to cuda if not already done
    pipe.text_encoder = pipe.text_encoder.to(device=accelerator.device)
    
    prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
    pipe.encode_prompt(captions,
                            do_classifier_free_guidance=False,
                            device=accelerator.device,
                            complex_human_instruction=[
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
        "- Do not add detail that is not in the original prompt! Do not mention colors if the User Prompt does not!",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "- If the prompt contains explicit material, do not remove this information from the output, which would damage the performance of downstream classifiers.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A cat sleeping peacefully, showcasing the joy of pet ownership. Cute floof kitty cat gatto.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene featuring a crowd of people.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ])

    # compress the prompt_embeds
    prompt_embeds = prompt_embeds[prompt_attention_mask.to(torch.bool)]
    return prompt_embeds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)
    pretrained_pipe_path = params.pretrained_pipe_path
    if params.urls == None:
        urls = get_secured_urls(params.r2_access_key,
                                params.r2_secret_key,
                                params.r2_endpoint,
                                params.r2_bucket_name,
                                params.r2_tar_files)
    else:
        urls = params.urls
    
    def node_no_split(src):
        return src
    accelerator = Accelerator()
    dataset = wds.WebDataset(urls, shardshuffle=False, handler=wds.ignore_and_continue, nodesplitter=node_no_split, workersplitter=wds.split_by_worker).\
                decode('pil').to_tuple(["jpg", 'jpeg'], "txt", handler=wds.ignore_and_continue)

    device = accelerator.device
    num_processes = accelerator.num_processes
    pipe = SanaPAGPipeline.from_pretrained(pretrained_pipe_path).to(torch.bfloat16)
    vae_compression = 32
    resolution = pipe.transformer.config.sample_size * vae_compression
    pipe.transformer = None
    gc.collect()
    torch.cuda.empty_cache()

    if resolution == 256:
        aspect_ratios = ASPECT_RATIO_256_BIN
    elif resolution == 512:
        aspect_ratios = ASPECT_RATIO_512_BIN
    elif resolution == 1024:
        aspect_ratios = ASPECT_RATIO_1024_BIN
    else:
        aspect_ratios = ASPECT_RATIO_2048_BIN


    os.makedirs(f'cache', exist_ok=True)
    dataset_fetcher = DatasetFetcher(dataset, accelerator.process_index, accelerator.num_processes)

    device = accelerator.device
    
    def shard_dataset(dataset, rank, world_size):
        return itertools.islice(dataset, rank, None, world_size)

    # Shard dataset based on process index
    sharded_dataset = shard_dataset(dataset_fetcher, accelerator.process_index, accelerator.num_processes)

    k = 0
    j = accelerator.process_index
    for img, caption in tqdm.tqdm(sharded_dataset):
        img = pipe.image_processor.pil_to_numpy(img)
        img = torch.tensor(img, device=device, dtype=pipe.dtype)
        img = torch.moveaxis(img, -1, 1)

        with torch.no_grad():
            cache_latents_embeddings(img, caption, j, aspect_ratios, accelerator.process_index)
        j = j + accelerator.num_processes
        k = k + 1
