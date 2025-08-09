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
from collections import deque
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
        self.accelerator = accelerator
        self.shards = shards
        self.buckets = {}
        self.valid_buckets = set()
        self.ready_bucket = -1.0

        self.batch_size = batch_size
        self.r2_access_key = r2_access_key
        self.r2_secret_key = r2_secret_key
        self.r2_endpoint = r2_endpoint
        self.r2_bucket_name = r2_bucket_name
        self.local_temp_dir = local_temp_dir
        self.features_path = features_path

    def add_bucket(self, ratio):
        self.buckets[ratio] = deque(maxlen=self.batch_size)

    def find_closest_ratio(self, ratio):
        max_error = 1000.0
        res = -1.0
        for r in self.buckets.keys():
            error = abs(ratio - r)
            if error < max_error:
                max_error = error
                res = r
        return res
    
    def get_next_shard_index(self):
        return random.randint(0, len(self.shards) - 1)
    
    def __iter__(self):
        current_shard_index = self.get_next_shard_index()
        sync_counter = 0
    
        def pt_decoder(key, value):
            if key.endswith(".pt"):
                # Load the tensor from the bytes object
                return torch.load(io.BytesIO(value), map_location=self.accelerator.device)
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

            try:
                download_tar(dataset_url, local_shard_path)
            except:
                current_shard_index = self.get_next_shard_index()
                continue

            dataset = (
                wds.WebDataset(local_shard_path, shardshuffle=True, nodesplitter=None, workersplitter=None, handler=wds.ignore_and_continue)
                .decode(pt_decoder)
                .to_tuple('ratio', 'latent.pt', 'emb.pt')
            )

            self.accelerator.wait_for_everyone()
            for ratio, latent, emb in dataset:
                ratio = float(ratio)
                if not ratio in self.buckets.keys():
                    self.add_bucket(ratio)
                self.buckets[ratio].append((latent, emb))
                
                sync_counter += 1
                if sync_counter % (self.batch_size * 2) != 0:
                    continue
                
                # Check for valid buckets before synchronization
                for r in list(self.buckets.keys()):
                    if len(self.buckets[r]) >= self.batch_size:
                        self.valid_buckets.add(r)
                
                if not self.valid_buckets:
                    continue
                
                bucket_list = list(self.valid_buckets)
                max_buckets = 100
                bucket_list = bucket_list[:max_buckets] + [-1.0] * (max_buckets - len(bucket_list))
                bucket_tensor = torch.tensor(bucket_list, device=self.device, dtype=torch.float32)
                
                try:
                    # Gather valid buckets across processes
                    self.accelerator.wait_for_everyone()
                    dist_valid_buckets = self.accelerator.gather(bucket_tensor)
                    
                    unique_ratios, counts = torch.unique(dist_valid_buckets, return_counts=True)
                    ratio_counts = {float(r): int(c) for r, c in zip(unique_ratios, counts) 
                                  if r != -1.0}
                    
                    for ratio, count in ratio_counts.items():
                        if count >= self.num_processes:
                            closest_ratio = self.find_closest_ratio(ratio)
                            batch = ([self.buckets[closest_ratio][i][0] for i in range(self.batch_size)],
                                    [self.buckets[closest_ratio][i][1] for i in range(self.batch_size)])
                            yield torch.stack(batch[0]), batch[1]
                            
                            # Clean up after yielding
                            self.buckets[closest_ratio].clear()
                            self.valid_buckets.remove(closest_ratio)
                            break
                            
                except Exception as e:
                    print(f"Process {self.process_index} gather error: {e}")
                    continue
            
            # Clean up at end of shard
            if os.path.exists(local_shard_path):
                os.remove(local_shard_path)
            current_shard_index = self.get_next_shard_index()
            self.accelerator.wait_for_everyone()