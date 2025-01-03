from common.training_parameters_reader import TrainingParameters
from common.cloudflare import get_secured_urls
from accelerate import Accelerator
import webdataset as wds
from torch.utils.data import DataLoader
from common.bucket_sampler import BucketDataset
from torch.utils.tensorboard import SummaryWriter
from webdataset.utils import pytorch_worker_info
from torch.optim.adamw import AdamW
import torch
import tqdm
from copy import deepcopy
from lycoris import create_lycoris, LycorisNetwork
import bitsandbytes as bnb

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
            """Split the input sequence by PyTorch distributed rank."""
            yield from src

        datasets = [
            wds.WebDataset(url, shardshuffle=1000, handler=wds.warn_and_continue, nodesplitter=split_only_on_main)
            .decode("pil", handler=wds.warn_and_continue)  # Decode images as PIL objects
            .to_tuple(["jpg", 'jpeg'], "txt", handler=wds.warn_and_continue)  # Return image and text
        for url in urls]
        self.mix = wds.RandomMix(datasets)
        self.preservation_model = None

    def extract_latents(self, images):
        raise NotImplemented
    
    def extract_embeddings(self, captions):
        raise NotImplemented
    
    def initialize(self):
        params = self.params
        self.bucket_dataset = BucketDataset(self.mix,
                            params.batch_size,
                            self.aspect_ratios,
                            self.accelerator,
                            self.pipe,
                            extract_latents_handler=self.extract_latents,
                            extract_embeddings_handler=self.extract_embeddings)
        self.dataloader = DataLoader(self.bucket_dataset, batch_size=None)
        self.dataloader = self.accelerator.prepare_data_loader(self.dataloader)

        if self.accelerator.is_main_process:
            self.logger = SummaryWriter()
        else:
            self.logger = None
        
        if params.use_preservation:
            self.preservation_model = deepcopy(self.model)
            self.preservation_model.train(False)
            self.preservation_model = self.accelerator.prepare(self.preservation_model)
        
        # check for lora training
        if params.lora_rank != None:
            dtype = self.model.dtype
            device = self.model.device
            LycorisNetwork.apply_preset(
                {'target_name': params.lora_target_modules}
            )

            lycoris_net = create_lycoris(
                self.model,
                1.0,
                linear_dim=self.params.lora_rank,
                linear_alpha=self.params.lora_alpha,
                algo=self.params.lora_algo
            )
            for lora in lycoris_net.loras:
                lora = lora.to(dtype=dtype, device=self.accelerator.device)
            lycoris_net.apply_to()

            for param in self.model.parameters():
                param.requires_grad = False

            params_to_optimizer = lycoris_net.parameters()
        else:
            params_to_optimizer = self.model.parameters()

        if params.low_vram:
            self.optimizer = bnb.optim.Adam8bit(params_to_optimizer, lr=params.learning_rate)
        else:
            self.optimizer = AdamW(params_to_optimizer, lr=params.learning_rate)

    def validate(self):
        raise NotImplemented

    def optimize(self, batch, model):
        raise NotImplemented

    def finalize(self):
        pass

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

                with self.accelerator.accumulate(self.model):
                    loss = self.optimize(self.model, batch)
                    if self.preservation_model != None:
                        loss = loss + self.params.preservation_ratio * self.optimize(self.preservation_model, batch)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                if self.logger != None:
                    self.logger.add_scalar('train/loss', loss.detach().item(), self.global_step)
                    progress_bar.update(1)
                
                self.global_step = self.global_step + 1
                self.finalize()