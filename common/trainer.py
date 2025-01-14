from common.training_parameters_reader import TrainingParameters
from common.cloudflare import get_secured_urls
from accelerate import Accelerator
import webdataset as wds
from torch.utils.data import DataLoader
from common.bucket_sampler import BucketDataset
from common.bucket_sampler_cache import DataExtractor, BucketDatasetWithCache
from torch.utils.tensorboard import SummaryWriter
from webdataset.utils import pytorch_worker_info
from torch.optim.adamw import AdamW
import torch
import tqdm
from copy import deepcopy
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, PeftModel
from peft import LoHaConfig, LoKrConfig
from accelerate.data_loader import prepare_data_loader
from torchvision.transforms import Resize
from accelerate.utils import DataLoaderConfiguration
import torch.distributed as dist
import os

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
        
        dataloader_config = DataLoaderConfiguration(dispatch_batches=True, split_batches=True)
        self.accelerator_extractor = Accelerator(dataloader_config=dataloader_config)
        self.accelerator = Accelerator(gradient_accumulation_steps=params.gradient_accumulation_steps)
        
        def node_no_split(src):
            return src

        datasets = [wds.WebDataset(url, shardshuffle=1000, detshuffle=True, seed=0,
                handler=wds.ignore_and_continue,
                nodesplitter=node_no_split).\
                    decode("pil", handler=wds.ignore_and_continue).\
                    to_tuple(["jpg", 'jpeg'], "txt", handler=wds.ignore_and_continue) for url in urls]
        mix = wds.RandomMix(datasets)

        self.mix = mix
        self.preservation_model = None

    def extract_latents(self, images):
        raise NotImplemented
    
    def extract_embeddings(self, captions):
        raise NotImplemented
    
    def find_closest_ratio(self, ratio):
        # find the closest ratio from the aspect ratios table
        min_distance = 100
        target_ratio = 0.6
        for r in self.aspect_ratios.keys():
            distance = abs(float(r) - ratio)
            if min_distance > distance:
                target_ratio = r
                min_distance = distance
        
        # Return the result as a tuple (target_ratio, idx)
        return str(target_ratio)
    
    def initialize(self):
        params = self.params

        self.data_extractor = DataExtractor(self.mix,
                                            self.params.cache_size,
                                            self.pipe,
                                            self.accelerator.num_processes)
        self.dataloader_extractor = DataLoader(self.data_extractor, batch_size=1)
        self.bucket_sampler = BucketDatasetWithCache(self.params.batch_size,
                                                     self.params.cache_size,
                                                     self.aspect_ratios,
                                                     self.accelerator,
                                                     self.pipe)
        self.data_extractor_iter = iter(self.data_extractor)
        
        def collate_fn(batch):
            return batch
        self.dataloader_sampler = DataLoader(self.bucket_sampler, batch_size=None, collate_fn=collate_fn)
        self.dataloader_sampler = self.accelerator.prepare_data_loader(self.dataloader_sampler)


        if self.accelerator.is_main_process:
            self.logger = SummaryWriter()

            # also create a folder for the models under training
            os.makedirs('models', exist_ok=True)
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
            if self.params.lora_pretrained == None:
                if params.lora_algo == 'lora':
                    config = LoraConfig(r=params.lora_rank,
                                        lora_dropout=params.lora_dropout,
                                        target_modules=params.lora_target_modules,
                                        lora_alpha=params.lora_alpha)
                elif params.lora_algo == 'loha':
                    config = LoHaConfig(r=params.lora_rank,
                                        module_dropout=params.lora_dropout,
                                        target_modules=params.lora_target_modules,
                                        alpha=params.lora_alpha)
                elif params.lora_algo == 'lokr':
                    config = LoKrConfig(r=params.lora_rank,
                                        module_dropout=params.lora_dropout,
                                        target_modules=params.lora_target_modules,
                                        alpha=params.lora_alpha,
                                        use_effective_conv2d=True)
                self.model = get_peft_model(self.model, config).to(dtype=dtype)
                
            else:
                self.model = PeftModel.from_pretrained(self.model, params.lora_pretrained, is_trainable=True)
                self.model.print_trainable_parameters()

        if self.params.bfloat16:
            self.model = self.model.to(dtype=torch.bfloat16)
        params_to_optimizer = self.model.parameters()

        if params.low_vram:
            self.optimizer = bnb.optim.Adam8bit(params_to_optimizer,
                                                lr=params.learning_rate,
                                                weight_decay=params.weight_decay)
        else:
            self.optimizer = AdamW(params_to_optimizer,
                                   lr=params.learning_rate,
                                   weight_decay=params.weight_decay)

    def validate(self):
        raise NotImplemented

    def optimize(self, batch, model):
        raise NotImplemented

    def finalize(self):
        pass

    def save_model(self):
        self.model.save_pretrained(f'models/{self.global_step}')
        self.pipe = self.pipe.to(torch.bfloat16)

    def run(self):
        params = self.params
        self.initialize()

        progress_bar = tqdm.tqdm(total=params.steps, desc='Num Steps')
        while self.global_step < params.steps:
            # start with the caching
            for idx in tqdm.tqdm(range(self.params.cache_size), desc='Caching latents and embeddings'):
                idx, img, caption = next(self.data_extractor_iter)
                if idx != self.accelerator.process_index:
                    continue

                # decode the caption into a string
                caption = torch.squeeze(caption)
                img = torch.squeeze(img)

                caption = bytes(caption.tolist()).decode('utf-8')

                # find the aspect ratio
                width = img.shape[-1]
                height = img.shape[-2]
                ratio = height / width
                closest_ratio = self.find_closest_ratio(ratio)
                height_target = int(self.aspect_ratios[closest_ratio][0])
                width_target = int(self.aspect_ratios[closest_ratio][1])

                # resize the image
                resize_transform = Resize((height_target, width_target))
                img = resize_transform(img)

                # compute the latents and embeddings
                with torch.no_grad():
                    latent = self.extract_latents(img.to(self.accelerator.device))
                    embedding = self.extract_embeddings(caption)

                # save on the disk
                embedding = [emb.cpu() for emb in embedding]
                to_save = (closest_ratio, latent.cpu(), embedding)
                torch.save(to_save, f'cache/{idx + self.params.cache_size * torch.cuda.current_device()}.npy')
            
            # then go through the cache items
            for batch in self.dataloader_sampler:
                # in the case you need to start caching new elements
                if isinstance(batch, list) == False:
                    break
                if self.global_step % params.num_steps_per_validation == 0:
                    if self.accelerator.is_main_process:
                        with torch.no_grad():
                            self.validate()
                            self.save_model()

                for elem in batch:
                    elem = elem.to(device=self.accelerator.device)
                    if self.params.bfloat16:
                        elem = elem.to(dtype=torch.bfloat16)
                
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