from common.training_parameters_reader import TrainingParameters
from accelerate import Accelerator
import webdataset as wds
from torch.utils.data import DataLoader
from common.bucket_sampler import BucketSampler, BucketSamplerExtractFeatures, BucketSamplerDreambooth
from torch.utils.tensorboard import SummaryWriter
import torch
import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from peft import LoHaConfig, LoKrConfig, FourierFTConfig
import os
from diffusers.training_utils import EMAModel
from transformers import AutoImageProcessor, AutoModel
import random
from torch.optim.lr_scheduler import LambdaLR
import os
from diffusers import SanaTransformer2DModel

#from Sana.diffusion.utils.optimizer import CAME8BitWrapper

class Model:
    def __init__(self, params : TrainingParameters):
        os.environ['NCCL_P2P_DISABLE'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        self.accelerator = Accelerator(gradient_accumulation_steps=params.gradient_accumulation_steps)
        self.params = params

        self.process_index = self.accelerator.process_index
        self.num_processes = self.accelerator.num_processes

        # each gpu should have the same number of shards
        if self.params.dreambooth_dataset_folder == None:
            if self.params.num_shards >= self.num_processes:
                num_shards_per_gpu = self.params.num_shards // self.num_processes

                # allocate a range of shards for our process
                self.shard_index_begin = self.process_index * num_shards_per_gpu

                if self.process_index != self.num_processes - 1:
                    self.shard_index_end = self.shard_index_begin + num_shards_per_gpu
                else:
                    self.shard_index_end = self.params.num_shards
            else:
                # in the case of less shards than GPUs, repeat the images on all gpus
                self.shard_index_begin = 0
                self.shard_index_end = self.params.num_shards
        
        self.global_step = 0

    def extract_latents(self, images):
        raise NotImplemented
    
    def extract_embeddings(self, captions):
        raise NotImplemented
    
    def format_embeddings(self, embeds):
        pass

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
        # use flash attention
        # sana's transformer cannot use this
        if not isinstance(self.model,  SanaTransformer2DModel):
            self.pipe.enable_xformers_memory_efficient_attention()
        params = self.params
        if self.accelerator.is_main_process:
            self.logger = SummaryWriter()

            # also create a folder for the models under training
            os.makedirs('models', exist_ok=True)
        else:
            self.logger = None
        
        if params.dreambooth_dataset_folder != None:
            self.sampler = BucketSamplerDreambooth(
                params.dreambooth_dataset_folder,
                params.dreambooth_regularization_folder,
                params.dreambooth_instance,
                params.dreambooth_class,
                params.dreambooth_num_repeats,
                params.dreambooth_lambda,
                self.accelerator,
                params.batch_size,
                params.vae_max_batch_size,
                params.text_encoder_max_batch_size,
                self,
                params.dataset_seed,
                cache_size=4,
            )
        elif params.compute_features == False:
            shards = [f'shard-{shard_index:06d}.tar' for shard_index in range(self.shard_index_begin, self.shard_index_end)]
            self.sampler = BucketSampler(shards,
                            params.r2_dataset_folder,
                            self.accelerator,
                            params.batch_size,
                            params.r2_access_key,
                            params.r2_secret_key,
                            params.r2_endpoint,
                            params.r2_bucket_name,
                            params.dataset_seed,
                            cache_size=4)
        else:
            shards = [f'shard-{shard_index:06d}.tar' for shard_index in range(self.shard_index_begin, self.shard_index_end)]
            self.sampler = BucketSamplerExtractFeatures(shards,
                            params.r2_dataset_folder,
                            self.accelerator,
                            params.batch_size,
                            params.r2_access_key,
                            params.r2_secret_key,
                            params.r2_endpoint,
                            params.r2_bucket_name,
                            params.vae_max_batch_size,
                            params.text_encoder_max_batch_size,
                            self,
                            params.dataset_seed,
                            cache_size=4)
        #self.sampler = self.accelerator.prepare(self.sampler)
        
        # check for lora training
        if self.params.lora_rank != None:
            dtype = self.model.dtype
            device = self.model.device
            if self.params.lora_pretrained == None:
                if params.lora_algo == 'lora':
                    config = LoraConfig(r=params.lora_rank,
                                        lora_dropout=params.lora_dropout,
                                        target_modules=params.lora_target_modules,
                                        lora_alpha=params.lora_alpha,
                                        use_dora=params.lora_use_dora)
                elif params.lora_algo == 'loha':
                    config = LoHaConfig(r=params.lora_rank,
                                        module_dropout=params.lora_dropout,
                                        target_modules=params.lora_target_modules,
                                        alpha=params.lora_alpha)
                elif params.lora_algo == 'lokr':
                    config = LoKrConfig(r=params.lora_rank,
                                        module_dropout=params.lora_dropout,
                                        target_modules=params.lora_target_modules,
                                        alpha=params.lora_alpha)
                elif params.lora_algo == 'fourierft':
                    config = FourierFTConfig(target_modules=params.lora_target_modules, init_weights=True, alpha=0.01, scaling=1.0, ifft2_norm='ortho')
                self.model = get_peft_model(self.model, config).to(dtype=dtype)
            else:
                self.model = PeftModel.from_pretrained(self.model, params.lora_pretrained, is_trainable=True)
                self.model.print_trainable_parameters()

        params_to_optimizer = self.model.parameters()
        self.optimizer = torch.optim.AdamW(params_to_optimizer,
                                            lr=params.learning_rate,
                                            weight_decay=params.weight_decay)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        #self.model = self.accelerator.prepare(self.model)

        warmup_steps = params.warmup_steps
        self.lr_scheduler = None
        if warmup_steps != None:
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0  # full LR after warmup
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

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
            self.empty_embeddings = self.extract_embeddings([''])

    def validate(self):
        raise NotImplemented

    def optimize(self, latents, embeddings):
        raise NotImplemented

    def finalize(self):
        pass

    def save_model(self):
        self.model.save_pretrained(f'models/{self.global_step}')
    
    def run(self):
        params = self.params
        self.initialize()
        loss_fn = torch.nn.MSELoss()
        progress_bar = tqdm.tqdm(total=params.steps, desc='Num Steps')
        avg_loss = torch.tensor(0, device=self.accelerator.device)

        while self.global_step < self.params.steps:
            # then go through the cache items
            for ratio, latents, embeddings in self.sampler:
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
                                self.ema_model.restore(self.model.parameters())
                    self.accelerator.wait_for_everyone()
                
                with self.accelerator.accumulate(self.model):
                    # randomly train with the unconditional embedding
                    prob = random.random()
                    if prob < self.params.train_unconditional_prob:
                        # put the embeddings to the empty one
                        for idx in range(len(embeddings)):
                            embeddings[idx] = self.empty_embeddings[0]
                    loss = self.optimize(ratio, latents, embeddings)
                    
                    # # check if we are using repa
                    # if self.params.use_repa:
                    #     self.repa_model.to(self.accelerator.device)
                    #     images = batch[0]
                    #     # extract the features with dino_v2 of the clean image in the spatial space
                    #     with torch.no_grad():
                    #         inputs = self.repa_processor(images=images.to(torch.float32), return_tensors="pt")
                    #         inputs = inputs['pixel_values'].to(self.accelerator.device, dtype=torch.bfloat16)
                    #         outputs = self.repa_model(inputs)
                    #     last_hidden_states = outputs.last_hidden_state
                    #     last_hidden_states = last_hidden_states.to(torch.bfloat16)

                    #     # https://github.com/sihyun-yu/REPA/blob/main/loss.py#L78
                    #     repa_projection = self.model.repa_proj
                    #     def mean_flat(x):
                    #         """
                    #         Take the mean over all non-batch dimensions.
                    #         """
                    #         return torch.mean(x, dim=list(range(1, len(x.size()))))
                        
                    #     proj_loss = 0
                    #     num_items = 0
                    #     for i, (z, z_tilde) in enumerate(zip(last_hidden_states, repa_projection)):
                    #         for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                    #             z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    #             z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                    #             proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
                    #             num_items = num_items + 1

                    #     proj_loss = proj_loss / (len(last_hidden_states) * params.batch_size)
                    #     loss = loss + params.repa_lambda * proj_loss

                    avg_loss = avg_loss + loss
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
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
                        try:
                            self.logger.add_scalar('train/loss', mean_loss.item(), self.global_step)
                            if self.lr_scheduler != None:
                                last_lr = self.lr_scheduler.get_last_lr()
                                self.logger.add_scalar('train/lr', last_lr[0], self.global_step)
                        except OSError as e:
                            print(f"[Warning] TensorBoard logging failed: {e}")
                    progress_bar.update(1)
                self.global_step = self.global_step + 1
            # if hasattr(self, 'repa_model'):
            #     self.repa_model.cpu()