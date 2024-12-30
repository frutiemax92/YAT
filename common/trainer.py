from common.training_parameters_reader import TrainingParameters
from common.cloudflare import get_secured_urls
from accelerate import Accelerator
import webdataset as wds
from torch.utils.data import DataLoader
from common.bucket_sampler import BucketDataset
from torch.utils.tensorboard import SummaryWriter
import torch
import tqdm

class Trainer:
    def __init__(self, params : TrainingParameters):
        self.params = params
        self.global_step = 0

        if params.urls == None:
            urls = get_secured_urls(params.r2_access_key,
                                    params.r2_secret_key,
                                    params.r2_endpoint,
                                    params.r2_bucket_name,
                                    params.r2_tar_files)
        else:
            urls = params.urls
        
        self.accelerator = Accelerator(gradient_accumulation_steps=params.gradient_accumulation_steps)

        def split_only_on_main(src, group=None):
            yield from src

        datasets = [
            wds.WebDataset(url, shardshuffle=1000, handler=wds.warn_and_continue, nodesplitter=split_only_on_main)
            .decode("pil", handler=wds.warn_and_continue)  # Decode images as PIL objects
            .to_tuple(["jpg", 'jpeg'], "txt", handler=wds.warn_and_continue)  # Return image and text
        for url in urls]
        self.mix = wds.RandomMix(datasets)

    def extract_latents(self, images):
        raise NotImplemented
    
    def extract_embeddings(self, captions):
        raise NotImplemented
    
    def initialize(self):
        params = self.params
        bucket_dataset = BucketDataset(self.mix,
                            params.batch_size,
                            self.aspect_ratios,
                            self.accelerator,
                            self.pipe,
                            extract_latents_handler=self.extract_latents,
                            extract_embeddings_handler=self.extract_embeddings)
        self.dataloader = DataLoader(bucket_dataset, batch_size=None)

        if self.accelerator.is_main_process:
            self.logger = SummaryWriter()
        else:
            self.logger = None

    def validate(self):
        raise NotImplemented

    def optimize(self, batch):
        raise NotImplemented

    def run(self):
        params = self.params
        self.initialize()

        progress_bar = tqdm.tqdm(total=params.steps, desc='Num Steps')
        while self.global_step < params.steps:
            for batch in self.dataloader:
                if self.global_step % params.num_steps_per_validation == 0:
                    if self.accelerator.is_main_process:
                        with torch.no_grad():
                            self.validate()
                self.optimize(batch)
                self.global_step = self.global_step + 1
                progress_bar.update(1)
