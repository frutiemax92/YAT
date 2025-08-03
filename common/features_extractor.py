from tqdm import tqdm
from copy import copy
from common.training_parameters_reader import TrainingParameters
from common.cloudflare import get_secured_urls
import torch
from accelerate import Accelerator
import os
import numpy as np
import webdataset as wds
from torchvision import transforms
from common.dataset_fetcher import DatasetFetcher
from common.cloudflare import get_client
import io

class FeaturesExtractor:
    def __init__(self, model, params : TrainingParameters):
        self.model = model
        self.params = params

        # read multi-gpu options
        self.accelerator = Accelerator()
        self.process_index = self.accelerator.process_index
        self.num_processes = self.accelerator.num_processes

        # each gpu should have the same number of shards
        num_shards_per_gpu = self.params.num_shards // self.num_processes

        # allocate a range of shards for our process
        self.shard_index_begin = self.process_index * num_shards_per_gpu
        self.shard_index_end = self.shard_index_begin + num_shards_per_gpu

    def run(self, local_temp_dir = 'temp'):
        # build the webdataset from the shard range
        shards = [(shard_index, f'{self.params.r2_dataset_folder}/shard-{shard_index:06d}.tar') for shard_index in range(self.shard_index_begin, self.shard_index_end)]
        batch_size = self.params.batch_size

        dataset_fetcher = DatasetFetcher([shard[1] for shard in shards],
                                         r2_access_key=self.params.r2_access_key,
                                         r2_secret_key=self.params.r2_secret_key,
                                         r2_endpoint=self.params.r2_endpoint,
                                         r2_bucket_name=self.params.r2_bucket_name,
                                         batch_size=self.params.batch_size,
                                         model=self.model)
        # write shards
        os.makedirs(local_temp_dir, exist_ok=True)
        current_shard = 0
        current_element = 0
        shard_size = int(self.params.r2_upload_shard_size)
        upload_key = self.params.r2_upload_key

        shard_filename = f'shard-{shards[current_shard][0]:06d}.tar'
        local_path = os.path.join(local_temp_dir, shard_filename)
        remote_key = f'{upload_key}/{shard_filename}'

        def tensor_to_bytes(tensor):
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            return buffer.getvalue()

        pbar = tqdm(total=shard_size, desc="processing shard elements")
        stream = open(local_path, 'wb')
        writer = wds.TarWriter(stream)
        for images, captions in dataset_fetcher:
            with torch.no_grad():
                # extract the vae features
                latents = self.model.extract_latents(images)
                embeddings = self.model.extract_embeddings(captions)

            for i in range(batch_size):
                sample = {
                    "__key__": f'{current_element:07d}',
                    'latent.pt': tensor_to_bytes(latents[i]),
                    'emb.pt': tensor_to_bytes(embeddings[i])
                }
                writer.write(sample)
                current_element = current_element + 1
                pbar.update(1)

                if current_element >= shard_size:
                    writer.close()
                    s3_client = get_client(self.params.r2_access_key,
                                        self.params.r2_secret_key,
                                        self.params.r2_endpoint)
                    s3_client.upload_file(local_path, self.params.r2_bucket_name, remote_key)
                    current_shard = current_shard + 1
                    current_element = 0
                    pbar.reset()

                    # delete the file and increase the shard index
                    os.remove(local_path)
                    shard_filename = f'shard-{shards[current_shard][0]:06d}.tar'
                    local_path = os.path.join(local_temp_dir, shard_filename)
                    stream = open(local_path, 'wb')
                    remote_key = f'{upload_key}/{shard_filename}'
                    writer = wds.TarWriter(stream)

            

                

            



            



                