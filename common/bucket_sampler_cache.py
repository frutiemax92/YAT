import multiprocessing
from tqdm import tqdm
from torch.utils.data import BatchSampler, IterableDataset
import random
from copy import copy
from torchvision.transforms import Resize, Compose, RandomCrop, RandomHorizontalFlip, CenterCrop, PILToTensor
import torch
from accelerate import Accelerator
from diffusers import SanaPAGPipeline
import os
import gc
import numpy as np
import shutil

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    
class DataExtractor(IterableDataset):
    def __init__(self,
                 dataset,
                 cache_size,
                 pipe,
                 num_processes,
                 seed,
                 process_index):
        super().__init__()
        self.batch_size = 1
        self.dataset = dataset
        self.cache_size = cache_size
        self.dataset_iterator = iter(dataset)
        self.pipe = pipe
        self.num_processes = num_processes
        self.seed = seed
        self.process_index = process_index

    def __iter__(self):
        tqdm.write(f'Process Index = {self.process_index}')
        while True:
            random.seed(self.seed)
            for dataset_url, filename, img, caption in self.dataset:
                img = self.pipe.image_processor.pil_to_numpy(img)
                img = torch.tensor(img).to(dtype=self.pipe.dtype)
                img = torch.moveaxis(img, -1, 1)
                yield dataset_url, filename, img, caption

class DataExtractorFeatures(IterableDataset):
    def __init__(self,
                 dataset,
                 cache_size,
                 pipe,
                 num_processes,
                 seed,
                 process_index):
        super().__init__()
        self.batch_size = 1
        self.dataset = dataset
        self.cache_size = cache_size
        self.dataset_iterator = iter(dataset)
        self.pipe = pipe
        self.num_processes = num_processes
        self.seed = seed
        self.process_index = process_index

    def __iter__(self):
        tqdm.write(f'Process Index = {self.process_index}')
        while True:
            random.seed(self.seed)
            for item in self.dataset:
                yield item

class BucketDatasetWithCache(IterableDataset):
    def __init__(self,
                 batch_size,
                 cache_size,
                 aspect_ratios,
                 accelerator : Accelerator,
                 pipe,
                 embedding_size = (300, 2304)):
        super().__init__()
        self.cache_size = cache_size * accelerator.num_processes
        self.total_batch_size = batch_size * accelerator.num_processes
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.accelerator = accelerator
        self.pipe = pipe
        self.buckets = {}
        self.embedding_size = embedding_size

        # create the cache folder
        if self.accelerator.is_main_process:
            if os.path.exists('cache'):
                shutil.rmtree('cache')
            os.makedirs('cache', exist_ok=True)
    
    def __len__(self):
        return self.cache_size
    
    def __iter__(self):
        for idx in tqdm(range(self.cache_size), desc='Processing cache'):
            ratio, image, latent, embedding = torch.load(f'cache/{idx}.npy')

            # find if this bucket already exists
            if not ratio in self.buckets.keys():
                self.buckets[ratio] = []
            self.buckets[ratio].append((image, latent, embedding))

            # check if the bucket is full
            if len(self.buckets[ratio]) == self.total_batch_size:
                latents = []
                embeddings = []
                images = []
                current_batch = 0
                for elem in self.buckets[ratio]:
                    image_tmp, latent_tmp, embedding_tmp = elem
                    latents.append(latent_tmp)
                    embeddings.append(embedding_tmp)
                    images.append(image_tmp)

                    if len(latents) == self.batch_size:
                        batch = []
                        batch.append(torch.stack(images).squeeze())
                        batch.append(torch.stack(latents).squeeze())

                        # here, we need to make sure all the embeddings fit into a size
                        embeds = []
                        masks = []
                        for idx in range(len(embeddings)):
                            max_len = self.embedding_size  # assuming this is your target sequence length, e.g., 300

                            # Get current embedding
                            emb = embeddings[idx]  # Shape: (93, 2304)
                            length = emb.shape[0]

                            # Initialize padded tensor and mask
                            padded = torch.zeros(max_len)
                            mask = torch.zeros((max_len[0],))

                            # Copy values
                            padded[:length] = emb
                            mask[:length] = 1

                            embeds.append(padded)
                            masks.append(mask)
                        
                        batch.append(torch.stack(embeds).squeeze())
                        batch.append(torch.stack(masks).squeeze())
                        
                        yield batch
                        latents.clear()
                        embeddings.clear()
                        images.clear()
                        current_batch = current_batch + 1
                self.buckets.pop(ratio)
        
        for j in range(self.accelerator.num_processes):
            yield torch.tensor([i for i in range(self.batch_size)])

if __name__ == '__main__':
    test = [1, 3, 4, 5]
    num = 1000
    my_iter = iter(test)

    for j in tqdm(range(num)):
        try:
            b = next(my_iter)
        except StopIteration as e:
            b = iter(my_iter)
        print(b)