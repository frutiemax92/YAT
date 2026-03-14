from common.training_parameters_reader import TrainingParameters
from accelerate import Accelerator
from common.bucket_sampler import BucketSampler, BucketSamplerExtractFeatures, BucketSamplerDreambooth
from torch.utils.tensorboard import SummaryWriter
import torch
import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from peft import LoHaConfig, LoKrConfig, FourierFTConfig
import os
from diffusers.training_utils import EMAModel
from diffusers import BitsAndBytesConfig, SanaTransformer2DModel
import bitsandbytes
import random
from torch.optim.lr_scheduler import LambdaLR
import os
from diffusers.training_utils import compute_density_for_timestep_sampling
import math
from common.repa import RepaModel, RepaConfig
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from peft.helpers import rescale_adapter_scale

class Model:
    def __init__(self, params : TrainingParameters):
        os.environ['NCCL_P2P_DISABLE'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ["NCCL_TIMEOUT"] = "100000000"
        self.accelerator = Accelerator(
            gradient_accumulation_steps=params.gradient_accumulation_steps, 
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3600))])
        self.params = params

        self.process_index = self.accelerator.process_index
        self.num_processes = self.accelerator.num_processes

        self.timesteps = [int(timestep) for timestep in params.timesteps]

        # check if we have defined some training timesteps
        if len(self.timesteps) != 0:
            # override the select timesteps function
            def get_timesteps_from_list(batch_size):
                # the default behavior is to get the timesteps from a normal distribution
                indices = [
                    random.choice(self.timesteps)
                    for _ in range(batch_size)
                ]
                return self.scheduler.timesteps[indices].to(self.accelerator.device)
            
            self.get_timesteps = get_timesteps_from_list

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
        else:
            self.shard_index_begin = 0
            self.shard_index_end = self.params.num_shards
        
        self.global_step = 0

        # check if we use a quantization technique (for lora training)
        self.quantization_config = None
        if self.params.lora_base_model_8bit:
            self.quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    def extract_latents(self, images):
        raise NotImplemented
    
    def get_timesteps(self, batch_size):
        # the default behavior is to get the timesteps from a normal distribution
        u = compute_density_for_timestep_sampling('logit_normal', batch_size, logit_mean=0, logit_std=1.0, mode_scale=1.29)
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(self.accelerator.device)
        return timesteps
    
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
    
    # by default, we enable xformers
    def enable_efficient_attention(self):
        self.pipe.enable_xformers_memory_efficient_attention()
        
    def initialize(self):
        # use flash attention
        # sana's transformer cannot use this
        torch.backends.cuda.enable_flash_sdp(True)
        self.enable_efficient_attention()
        params = self.params
        if self.accelerator.is_main_process:
            self.logger = SummaryWriter()

            # also create a folder for the models under training
            os.makedirs('models', exist_ok=True)
        else:
            self.logger = None
        
        if params.dreambooth_dataset_folder != None:
            shards = [f'shard-{shard_index:06d}.tar' for shard_index in range(self.shard_index_begin, self.shard_index_end)]
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
                cache_size=4 * params.dreambooth_num_repeats * params.dreambooth_num_regularisation_passes,
                r2_access_key=self.params.r2_access_key,
                r2_bucket_name=self.params.r2_bucket_name,
                r2_endpoint=self.params.r2_endpoint,
                r2_secret_key=self.params.r2_secret_key,
                shards=shards,
                dreambooth_num_regularisation_passes=params.dreambooth_num_regularisation_passes
            )
        elif params.compute_features == False:
            shards = [f'shard-{shard_index:06d}.tar' for shard_index in range(self.shard_index_begin, self.shard_index_end)]
            print(self.params.local_shard_paths)
            self.sampler = BucketSampler(shards,
                            params.r2_dataset_folder,
                            self.accelerator,
                            params.batch_size,
                            params.r2_access_key,
                            params.r2_secret_key,
                            params.r2_endpoint,
                            params.r2_bucket_name,
                            params.dataset_seed,
                            cache_size=4,
                            local_paths=self.params.local_shard_paths)
        else:
            shards = [f'shard-{shard_index:06d}.tar' for shard_index in range(self.shard_index_begin, self.shard_index_end)]
            print(self.params.local_shard_paths)
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
                            cache_size=4,
                            use_repa=params.use_repa,
                            local_paths=self.params.local_shard_paths)
        #self.sampler = self.accelerator.prepare(self.sampler)

        # apply the dual gpu patch
        from common.bucket_sampler import dual_gpu_bucket_sampler_iter
        class PatchedSampler(type(self.sampler)):
            def __iter__(self):
                return dual_gpu_bucket_sampler_iter(self)

        if self.params.dual_gpu:
            self.sampler.__class__ = PatchedSampler
        
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
                    config = FourierFTConfig(
                        target_modules=params.lora_target_modules, 
                        init_weights=True, 
                        alpha=params.fourierft_alpha, 
                        scaling=1.0, 
                        ifft2_norm='ortho')
                self.model = get_peft_model(self.model, config).to(dtype=dtype)
            else:
                self.model = PeftModel.from_pretrained(self.model, params.lora_pretrained, is_trainable=True).to(dtype=dtype)
            self.model.print_trainable_parameters()

        params_to_optimizer = self.model.parameters()

        if self.params.use_adamw_8bit == False:
            self.optimizer = torch.optim.AdamW(params_to_optimizer,
                                                lr=params.learning_rate,
                                                weight_decay=params.weight_decay)
        else:
            self.optimizer = bitsandbytes.optim.Lion8bit(params=params_to_optimizer, lr=params.learning_rate, weight_decay=params.weight_decay)

        if self.params.dual_gpu == False:
            self.optimizer = self.accelerator.prepare(self.optimizer)
            self.model = self.accelerator.prepare(self.model)

        warmup_steps = params.warmup_steps
        self.lr_scheduler = None
        if warmup_steps != None:
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # apply EMA
        self.ema_model = None
        if self.params.use_ema:
            self.ema_model = EMAModel(self.model.parameters(), decay=0.999)
            self.ema_model.to(self.accelerator.device)

        # extract empty embedding
        with torch.no_grad():
            self.empty_embeddings = self.extract_embeddings([''])

        # check if we use repa
        if self.params.use_repa:
            repa_config = RepaConfig(target_modules=['transformer_blocks.0', 'transformer_blocks.1', 'transformer_blocks.2'], 
                                     hidden_shape=[1152, 1152, 1152])

            if self.params.repa_pretrained_model == None:
                self.model = RepaModel(self.model, repa_config)
            else:
                self.model = RepaModel.from_pretrained(self.model, self.params.repa_pretrained_model)
            self.model.to(torch.bfloat16)
        
        # check if we have timesteps, then apply a step callback
        # currently, we only support peft models at inference
        def step_callback(pipe, step, timestep, callback_kwargs):
            if timestep.item() in self.timesteps:
                rescale_adapter_scale(self.model, 1.0)
            else:
                rescale_adapter_scale(self.model, 0.0)

            return callback_kwargs
        self.validation_step_callback = None
        if len(self.timesteps) != 0:
            self.validation_step_callback =  step_callback

    def validate(self):
        raise NotImplemented

    def optimize(self, latents, embeddings):
        raise NotImplemented

    def finalize(self):
        pass

    def save_model(self):
        self.accelerator.unwrap_model(self.model).save_pretrained(f'models/{self.global_step}')
    
    def run(self):
        params = self.params
        self.initialize()
        loss_fn = torch.nn.MSELoss()
        progress_bar = tqdm.tqdm(total=params.steps, desc='Num Steps')
        avg_loss = torch.tensor(0, device=self.accelerator.device)

        while self.global_step < self.params.steps:
            # then go through the cache items
            for batch in self.sampler:
                ratio = batch.ratio
                latents = batch.vae_features
                embeddings = batch.embeddings
                with self.accelerator.accumulate(self.model):
                    # randomly train with the unconditional embedding
                    prob = random.random()
                    if prob < self.params.train_unconditional_prob:
                        # put the embeddings to the empty one
                        for idx in range(len(embeddings)):
                            embeddings[idx] = self.empty_embeddings[0]
                    loss = self.optimize(ratio, latents, embeddings)

                    if self.params.use_repa:
                        loss = loss + self.params.repa_lambda * self.model.calculate_loss(batch.repa_features)

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
                    if self.params.dual_gpu == False:
                        mean_loss = torch.mean(self.accelerator.gather(avg_loss))
                    mean_loss = avg_loss
                    avg_loss = torch.tensor(0, device=self.accelerator.device)

                    if self.logger != None:
                        try:
                            self.logger.add_scalar('train/loss', mean_loss.item(), self.global_step)
                            if self.lr_scheduler != None:
                                last_lr = self.lr_scheduler.get_last_lr()
                                self.logger.add_scalar('train/lr', last_lr[0], self.global_step)
                        except OSError as e:
                            print(f"[Warning] TensorBoard logging failed: {e}")

                    if self.global_step % params.num_steps_per_validation == 0:
                        with torch.no_grad():
                            # ✅ Run reduction on ALL processes to sync EMA parameters
                            if self.ema_model != None:
                                for param in self.ema_model.shadow_params:
                                    if self.params.dual_gpu == False:
                                        param.data = self.accelerator.reduce(param.data, reduction="mean")
                    
                            # ✅ Ensure store() is called before restore()
                            if self.accelerator.is_main_process:
                                if self.ema_model != None:
                                    self.ema_model.store(self.model.parameters())  # Store original model weights
                                    self.ema_model.copy_to(self.model.parameters())
                                
                                # we need to check if we temporarly disable the peft adapter for the first inference step
                                if len(self.timesteps) != 0:
                                    if 0 in self.timesteps:
                                        rescale_adapter_scale(self.model, 1.0)
                                    else:
                                        rescale_adapter_scale(self.model, 0.0)
                                self.validate()

                                if len(self.timesteps) != 0:
                                    rescale_adapter_scale(self.model, 1.0)
                                self.save_model()
                    
                                if self.ema_model != None:
                                    self.ema_model.restore(self.model.parameters())
                    progress_bar.update(1)
                    self.global_step = self.global_step + 1

                    #if self.params.dual_gpu == False:
                        #self.accelerator.wait_for_everyone()
            # if hasattr(self, 'repa_model'):
            #     self.repa_model.cpu()