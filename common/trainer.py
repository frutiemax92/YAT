import PIL.Image
from common.training_parameters_reader import TrainingParameters
from common.cloudflare import get_secured_urls
from accelerate import Accelerator
import webdataset as wds
from torch.utils.data import DataLoader
from common.bucket_sampler import BucketDataset, RoundRobinMix
from common.bucket_sampler_cache import DataExtractor, DataExtractorFeatures, BucketDatasetWithCache
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
from torch.optim.lr_scheduler import CyclicLR
import shutil
import os
import PIL
from diffusers.training_utils import EMAModel
from common.cache import CacheFeaturesCompute, CacheLoadFeatures
from transformers import AutoImageProcessor, AutoModel
import io

#from Sana.diffusion.utils.optimizer import CAME8BitWrapper

class Model:
    def __init__(self, params : TrainingParameters):
        self.accelerator = Accelerator(gradient_accumulation_steps=params.gradient_accumulation_steps)

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
        
        def node_no_split(src):
            return src

        def custom_handler(exn):
            print(f"WebDataset error: {repr(exn)} -- skipping")
            return True  # continue
        
        self.folders = params.urls if params.urls != None else params.r2_tar_files
        if params.use_calculated_features == False:
            datasets = {
                url: wds.WebDataset(
                    url,
                    shardshuffle=False,
                    handler=custom_handler,
                    nodesplitter=node_no_split,
                    resampled=True
                )
                .decode("pil")
                .to_tuple("__key__", ["jpg", "jpeg"], "txt", handler=custom_handler)
                for url in urls
            }
            self.cache_calculator = CacheFeaturesCompute(save_to_disk=params.save_to_disk)
        else:
            def custom_decoder(key, value):
                if key.endswith('.npy') or key.endswith('.npz'):
                    buffer = io.BytesIO(value)
                    return torch.load(buffer)
                return value
            datasets = {url : wds.WebDataset(url, shardshuffle=False, handler=custom_handler, nodesplitter=node_no_split, resampled=True).\
                        decode(custom_decoder).to_tuple('npy') for url in urls}
            self.cache_calculator = CacheLoadFeatures()
        mix = RoundRobinMix(datasets, params.dataset_seed, save_to_disk=self.params.save_to_disk, folders=self.folders)

        self.mix = mix
        self.preservation_model = None
    
        params = self.params

        if params.use_calculated_features == False:
            self.data_extractor = DataExtractor(self.mix,
                                                self.params.cache_size,
                                                self.pipe,
                                                torch.cuda.device_count(),
                                                self.params.dataset_seed,
                                                self.accelerator.process_index)
            
        else:
            self.data_extractor = DataExtractorFeatures(self.mix,
                                    self.params.cache_size,
                                    self.pipe,
                                    torch.cuda.device_count(),
                                    self.params.dataset_seed,
                                    self.accelerator.process_index)
        self.dataloader_extractor = DataLoader(self.data_extractor, batch_size=1)
        # self.dataloader_extractor = prepare_data_loader(self.dataloader_extractor,
        #                                                 split_batches=True,
        #                                                 dispatch_batches=True,
        #                                                 #put_on_device=True
        #                                                 )

        self.bucket_sampler = BucketDatasetWithCache(self.params.batch_size,
                                                     self.params.cache_size,
                                                     self.aspect_ratios,
                                                     self.accelerator,
                                                     self.pipe, bucket_repeat=params.bucket_repeat)
        self.data_extractor_iter = iter(self.dataloader_extractor)
        
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

        self.optimizer = bnb.optim.AdamW8bit(params_to_optimizer,
                                            lr=params.learning_rate,
                                            weight_decay=params.weight_decay)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        # else:
        #     self.optimizer = AdamW(params_to_optimizer,
        #                            lr=params.learning_rate,
        #                            weight_decay=params.weight_decay)
        
        self.lr_scheduler = None
        if params.cyclic_lr_max_lr != None:
            self.lr_scheduler = CyclicLR(optimizer=self.optimizer,
                                    base_lr=params.learning_rate,
                                    max_lr=params.cyclic_lr_max_lr,
                                    step_size_up=params.cyclic_lr_step_size_up,
                                    step_size_down=params.cyclic_lr_step_size_down,
                                    mode=params.cylic_lr_mode)
        # apply EMA
        if self.params.use_ema:
            self.ema_model = EMAModel(self.model.parameters(), decay=0.999)
            self.ema_model.to(self.accelerator.device)
        self.ema_model = None

        # check if we use REPA regularisation loss
        # https://github.com/sihyun-yu/REPA/tree/main
        if self.params.use_repa:
            self.repa_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.repa_model = AutoModel.from_pretrained('facebook/dinov2-base').to(torch.bfloat16)

        # extract empty embedding
        with torch.no_grad():
            self.empty_embeddings = self.extract_embeddings('')

    def validate(self):
        raise NotImplemented

    def optimize(self, batch, model):
        raise NotImplemented

    def finalize(self):
        pass

    def save_model(self):
        self.model.save_pretrained(f'models/{self.global_step}')

    def cache_latents_embeddings(self, img, caption, cache_idx, save_img=False):
        img = torch.squeeze(img)

        # find the aspect ratio
        width = img.shape[-1]
        height = img.shape[-2]
        ratio = height / width
        closest_ratio = self.find_closest_ratio(ratio)
        height_target = int(self.aspect_ratios[closest_ratio][0])
        width_target = int(self.aspect_ratios[closest_ratio][1])

        # resize the image
        resize_transform = Resize((height_target, width_target))
        img = resize_transform(img.cpu())

        # compute the latents and embeddings
        with torch.no_grad():
            latent = self.extract_latents(img.to(self.accelerator.device))
            embedding = self.extract_embeddings(caption)

        # save on the disk
        embedding = embedding.cpu()
        img.cpu()
        if save_img == False:
            img = None
        to_save = (closest_ratio, img, latent.cpu(), embedding)
        torch.save(to_save, f'cache/{cache_idx}.npy')

        # return the tensor in case we save to disk
        return to_save
    
    def run(self):
        params = self.params
        self.initialize()

        progress_bar = tqdm.tqdm(total=params.steps, desc='Num Steps')
        avg_loss = torch.tensor(0, device=self.accelerator.device)

        while self.global_step < self.params.steps:
            self.cache_calculator.run(self)
                
            # then go through the cache items
            for batch in self.dataloader_sampler:
                # in the case you need to start caching new elements
                if isinstance(batch, list) == False:
                    break
                if self.global_step % params.num_steps_per_validation == 0:
                    with torch.no_grad():
                        # ✅ Run reduction on ALL processes to sync EMA parameters
                        if self.ema_model != None:
                            for param in self.ema_model.shadow_params:
                                self.accelerator.reduce(param.data, reduction="mean")
                
                        # ✅ Ensure store() is called before restore()
                        if self.accelerator.is_main_process:
                            if self.ema_model != None:
                                self.ema_model.store(self.model.parameters())  # Store original model weights
                                self.ema_model.copy_to(self.model.parameters())
                
                            self.validate()
                            self.save_model()
                
                            if self.ema_model != None:
                                self.ema_model.restore(self.model.parameters())  # ✅ Now restore works!

                for elem in batch:
                    elem = elem.to(device=self.accelerator.device)
                    if self.params.bfloat16:
                        elem = elem.to(dtype=torch.bfloat16)
                
                with self.accelerator.accumulate(self.model):
                    loss = self.optimize(self.model, batch)
                    if self.preservation_model != None:
                        loss = loss + self.params.preservation_ratio * self.optimize(self.preservation_model, batch)
                    
                    # check if we are using repa
                    if self.params.use_repa:
                        self.repa_model.to(self.accelerator.device)
                        images = batch[0]
                        # extract the features with dino_v2 of the clean image in the spatial space
                        with torch.no_grad():
                            inputs = self.repa_processor(images=images.to(torch.float32), return_tensors="pt")
                            inputs = inputs['pixel_values'].to(self.accelerator.device, dtype=torch.bfloat16)
                            outputs = self.repa_model(inputs)
                        last_hidden_states = outputs.last_hidden_state
                        last_hidden_states = last_hidden_states.to(torch.bfloat16)

                        # https://github.com/sihyun-yu/REPA/blob/main/loss.py#L78
                        repa_projection = self.model.repa_proj
                        def mean_flat(x):
                            """
                            Take the mean over all non-batch dimensions.
                            """
                            return torch.mean(x, dim=list(range(1, len(x.size()))))
                        
                        proj_loss = 0
                        num_items = 0
                        for i, (z, z_tilde) in enumerate(zip(last_hidden_states, repa_projection)):
                            for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                                z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                                z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                                proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
                                num_items = num_items + 1

                        proj_loss = proj_loss / (len(last_hidden_states) * params.batch_size)
                        loss = loss + params.repa_lambda * proj_loss

                    avg_loss = avg_loss + loss
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()

                    if self.ema_model != None:
                        self.ema_model.step(self.model.parameters())

                    if self.lr_scheduler != None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                if self.accelerator.sync_gradients:
                    mean_loss = torch.mean(self.accelerator.gather(avg_loss))
                    avg_loss = torch.tensor(0, device=self.accelerator.device)

                    if self.logger != None:
                        self.logger.add_scalar('train/loss', mean_loss.item(), self.global_step)
                        if self.lr_scheduler != None:
                            last_lr = self.lr_scheduler.get_last_lr()
                            self.logger.add_scalar('train/lr', last_lr[0], self.global_step)
                    progress_bar.update(1)
                self.global_step = self.global_step + 1
            
            # delete the cache
            if self.accelerator.is_main_process:
                if os.path.exists('cache'):
                    shutil.rmtree('cache')
                os.makedirs('cache', exist_ok=True)
            
            if hasattr(self, 'repa_model'):
                self.repa_model.cpu()
            self.accelerator.wait_for_everyone()