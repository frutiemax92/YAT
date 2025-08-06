import multiprocessing
from tqdm import tqdm
from torch.utils.data import BatchSampler, IterableDataset
import random
from copy import copy
from torchvision.transforms import Resize, Compose, RandomCrop, RandomHorizontalFlip, CenterCrop, PILToTensor
import torch
from accelerate import Accelerator
from diffusers import SanaPAGPipeline
import gc
import os
from tarfile import ReadError
from common.cloudflare import get_secured_urls, download_tar
import webdataset as wds
import argparse
from common.training_parameters_reader import TrainingParameters
import io


def flush():
    gc.collect()
    torch.cuda.empty_cache()

class BucketSampler:
    def __init__(self,
                 shards : list[str],
                 features_path : str,
                 accelerator : Accelerator,
                 batch_size : int,
                 r2_access_key : str,
                 r2_secret_key : str,
                 r2_endpoint : str,
                 r2_bucket_name : str,
                 local_temp_dir : str = 'temp'  
                 ):
        self.process_index = accelerator.process_index
        self.num_processes = accelerator.num_processes
        self.device = accelerator.device
        self.shards = shards
        self.buckets = {}
        self.batch_size = batch_size
        self.r2_access_key = r2_access_key
        self.r2_secret_key = r2_secret_key
        self.r2_endpoint = r2_endpoint
        self.r2_bucket_name = r2_bucket_name
        self.local_temp_dir = local_temp_dir
        self.features_path = features_path

    def add_bucket(self, ratio):
        self.buckets[ratio] = torch.tensor([0], device=self.device)
    
    def __iter__(self):
        current_shard_index = 0
        num_shards = len(self.shards)

        def pt_decoder(key, value):
            if key.endswith(".pt"):
                # Load the tensor from the bytes object
                return torch.load(io.BytesIO(value))
            else:
                return value.decode('utf-8')
    
        while True:
            dataset_url = get_secured_urls(self.r2_access_key,
                                           self.r2_secret_key,
                                            self.r2_endpoint,
                                            self.r2_bucket_name,
                                            [self.features_path + '/' + self.shards[current_shard_index]]
                                           )[0]
            local_shard_path = self.local_temp_dir + f'/shard_{self.process_index}.tar'
            download_tar(dataset_url, local_shard_path)

            dataset = (
                wds.WebDataset(local_shard_path, shardshuffle=True, nodesplitter=None, workersplitter=None, handler=wds.ignore_and_continue)
                .decode(pt_decoder)
                .to_tuple('ratio', 'latent.pt', 'emb.pt')
            )

            for ratio, latent, emb in dataset:
                yield ratio, latent, emb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)
    
    shards = [f'shard-{idx:06}.tar' for idx in range(66)]
    shards.extend([f'shard-{idx:06}.tar' for idx in range(70, 200)])
    shards.extend([f'shard-{idx:06}.tar' for idx in range(204, 271)])

    accelerator = Accelerator()
    sampler = BucketSampler(shards,
                            params.r2_dataset_folder,
                            accelerator,
                            params.batch_size,
                            params.r2_access_key,
                            params.r2_secret_key,
                            params.r2_endpoint,
                            params.r2_bucket_name)
    for ratio, latent, emb in sampler:
        print(ratio)