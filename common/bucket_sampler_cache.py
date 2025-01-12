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

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    
class DataExtractor(IterableDataset):
    def __init__(self,
                 dataset,
                 cache_size,
                 pipe):
        super().__init__()
        self.batch_size = 1
        self.dataset = dataset
        self.cache_size = cache_size
        self.dataset_iterator = iter(dataset)
        self.pipe = pipe

    def __iter__(self):
        for idx in tqdm(range(self.cache_size), desc='Caching latents and embeddings'):
            # get the next item and make sure to check for iterator exhaust
            try:
                img, caption = next(self.dataset_iterator)
            except StopIteration:
                self.dataset_iterator = iter(self.dataset)
                img, caption = next(self.dataset_iterator)

            img = self.pipe.image_processor.pil_to_numpy(img)
            img = torch.tensor(img).to(dtype=self.pipe.dtype)
            img = torch.moveaxis(img, -1, 1)

            caption = list(caption.encode('utf-8'))
            caption = torch.tensor(caption, dtype=torch.uint8)
            yield img, caption, torch.tensor(idx)

class BucketDatasetWithCache(IterableDataset):
    def __init__(self,
                 batch_size,
                 cache_size,
                 aspect_ratios,
                 accelerator : Accelerator,
                 pipe):
        super().__init__()
        self.cache_size = cache_size
        self.total_batch_size = batch_size * accelerator.num_processes
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.accelerator = accelerator
        self.pipe = pipe
        self.buckets = {}

        # create the cache folder
        os.makedirs('cache', exist_ok=True)

    def __iter__(self):
        for idx in tqdm(range(self.cache_size), desc='Processing cache'):
            ratio, latent, embedding = torch.load(f'cache/{idx}.npy')

            # find if this bucket already exists
            if not ratio in self.buckets.keys():
                self.buckets[ratio] = []
            self.buckets[ratio].append((latent, embedding))

            # check if the bucket is full
            if len(self.buckets[ratio]) == self.total_batch_size:
                # transform the PIL image to a tensor
                latents = []
                embeddings = []
                for elem in self.buckets[ratio]:
                    latent_tmp, embedding_tmp = elem
                    latents.append(latent_tmp)
                    embeddings.append(embedding_tmp)

                    if len(latents) == self.batch_size:
                        batch = []
                        batch.append(torch.stack(latents).squeeze())
                        
                        num_dims = len(embeddings[0])
                        for i in range(num_dims):
                            dim = []
                            for embed in embeddings:
                                dim.append(embed[i])
                            batch.append(torch.stack(dim).squeeze())
                        yield batch
                        latents.clear()
                        embeddings.clear()
                self.buckets.pop(ratio)
        yield torch.tensor(0)

if __name__ == '__main__':
    test = 'hello world'
    encoded = list(test.encode('utf-8'))
    tensor = torch.tensor(encoded, dtype=torch.uint8)
    decoded = bytes(tensor.tolist()).decode('utf-8')
    print(decoded)