from common.training_parameters_reader import TrainingParameters
from accelerate import Accelerator
from common.bucket_sampler import (BucketSampler, BucketSamplerExtractFeatures, BucketSamplerDreambooth,
                                  BucketSamplerLwControlNet)
from torch.utils.tensorboard import SummaryWriter
import torch
import tqdm
import gc
from peft import PeftConfig, get_peft_model, PeftModel
import os
from torch.optim.swa_utils import AveragedModel
from diffusers import BitsAndBytesConfig, SanaTransformer2DModel
import bitsandbytes
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from diffusers.training_utils import compute_density_for_timestep_sampling
import math
import copy
import contextlib
import importlib
import inspect
from common.repa import RepaModel, RepaConfig
from common.precompute import FeaturesPrecompute
from common.text_precompute import TextEmbeddingsPrecompute
from common.peft_bnb_patch import patch_lycoris_for_bnb_4bit
from common import lw_controlnet
from common.offload import offload_frozen_blocks, offload_activations
from accelerate.utils import InitProcessGroupKwargs, DistributedDataParallelKwargs
from datetime import timedelta
from peft.helpers import rescale_adapter_scale
from accelerate.utils import DeepSpeedPlugin
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN, ASPECT_RATIO_512_BIN

class Model:
    def __init__(self, params : TrainingParameters):
        os.environ['NCCL_P2P_DISABLE'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ["NCCL_TIMEOUT"] = "100000000"

        # a quantized model dequantizes a different weight on every layer of every step, which
        # fragments the allocator badly. Expandable segments give that memory back instead of
        # holding it reserved. Set before anything touches cuda, and overridable from the outside.
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

        self.accelerator = Accelerator(
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            kwargs_handlers=[
                InitProcessGroupKwargs(timeout=timedelta(seconds=3600)),
                DistributedDataParallelKwargs(find_unused_parameters=True),
            ],
        )

        if self.accelerator.state.deepspeed_plugin is not None:
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = params.batch_size
            self.accelerator.state.deepspeed_plugin.deepspeed_config["reduce_bucket_size"] = 50000000,
            self.accelerator.state.deepspeed_plugin.deepspeed_config["allgather_bucket_size"] =  50000000
            print('using deepspeed')
        self.params = params

        self.process_index = self.accelerator.process_index
        self.num_processes = self.accelerator.num_processes

        if self.num_processes > 1:
            seed_tensor = torch.tensor(params.dataset_seed, dtype=torch.int64, device=self.accelerator.device)
            torch.distributed.broadcast(seed_tensor, src=0)
            params.dataset_seed = int(seed_tensor.item())

        # draws the training timesteps and noise. Created once so its state advances from a step to
        # the next, and seeded apart on every gpu so they don't all train at the same timestep
        self.noise_generator = torch.Generator().manual_seed(params.dataset_seed + self.process_index)

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
        if self.params.dreambooth_instance_tar == None and self.params.num_shards != None:
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
            # a run reading local tars - dreambooth, or a lightweight controlnet with
            # lw_controlnet_tars - has no shard range to split
            self.shard_index_begin = 0
            self.shard_index_end = self.params.num_shards if self.params.num_shards != None else 0

        self.global_step = 0

        # set by initialize() when the features are precalculated instead of extracted every step
        self.features_precompute = None

        # lightweight controlnet: set by initialize() when train_lw_controlnet is on. The encoder
        # is built on first use and dropped again once the features are on disk.
        # reference_control is a FourierControl - the naming is historical, it owns the whole
        # fourierft control path and not a reference cross attention any more.
        self.reference_control = None
        self.reference_encoder = None
        self.reference_config = lw_controlnet.build_config(params) if params.train_lw_controlnet else None

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

    def encode_reference_images(self, images):
        """The ReferenceFeatures of a batch of reference images, as the samplers hand them over.

        dinov2 is frozen and its output is a plain dataclass, so the encoder is only needed while
        the features are being produced: a precompute run writes them to disk and drops it, and a
        live run keeps it resident, which is a couple hundred megabytes in bfloat16.
        """
        if self.reference_encoder == None:
            self.reference_encoder = lw_controlnet.build_encoder(self.reference_config,
                                                                 device=self.accelerator.device)
        else:
            self.reference_encoder.to(device=self.accelerator.device)
        return self.reference_encoder([lw_controlnet.to_reference_tensor(image) for image in images])

    def reference_context(self, reference):
        """Set the reference conditioning for the block, or do nothing on a plain lora run.

        Lets a validate() be written the same way whether or not the run has a control path.

        The reference needs no layout: the spectrum of a controlled layer is as long as that layer's
        output dimension, so what it writes is one offset per output channel and where the image
        tokens sit in the sequence never enters. That is also its limit - the offset is uniform over
        the frame - and it is why a validation pass cannot be misconfigured into reading the
        reference differently than training did.

        The two scalings are set here as well, rather than left at whatever the last caller wanted,
        so a validation image is always the model as it trains. Turning them into dials is what
        inference is for, not something a training curve should have to be read through.
        """
        if self.reference_control == None:
            return contextlib.nullcontext()
        stack = contextlib.ExitStack()
        stack.enter_context(
            self.reference_control.strength_scope(self.params.lw_controlnet_reference_scaling,
                                                 spectrum=self.params.lw_controlnet_spectrum_scaling))
        stack.enter_context(self.reference_control.reference(reference))
        return stack

    def free_reference_encoder(self):
        """Drop the reference encoder, once every feature it had to produce is on disk."""
        self.reference_encoder = None
        gc.collect()
        torch.cuda.empty_cache()

    def validation_references(self):
        """The reference features to validate with, one per validation prompt, or None.

        The configured reference images are cycled over, so a single reference image can be paired
        with every prompt. A model whose validate() ignores this simply validates its lora.
        """
        if self.reference_control == None or self.params.lw_controlnet_validation_references in (None, []):
            return None
        from PIL import Image
        paths = self.params.lw_controlnet_validation_references
        references = []
        for index in range(len(self.params.validation_prompts)):
            image = Image.open(paths[index % len(paths)]).convert('RGB')
            references.append(self.encode_reference_images([image]))
        return references

    def encode_validation_prompts(self):
        """Encode everything validate() will ask the text encoder for, and return True.

        The precompute pass memoizes the embeddings, so a model that implements this can have its
        text encoder freed before the training starts instead of after the first validation. The
        calls have to match the ones validate() makes, arguments included, for the memoized
        embeddings to be found again.
        """
        return False

    def format_embeddings(self, embeds):
        pass

    def resolution_buckets(self):
        """{resolution: {bucket key: height / width}} for the resolutions being trained at.

        With a single resolution the bucket keys are the keys of the aspect ratio table ('0.67'). With
        several they are '<resolution>:<key>' ('512:0.67'), so that a bucket only ever holds images
        of one aspect ratio and one resolution.
        """
        resolutions = getattr(self, 'resolutions', None)
        if not resolutions or len(resolutions) == 1:
            resolution = resolutions[0] if resolutions else None
            return {resolution: {key: float(key) for key in self.aspect_ratios.keys()}}
        return {resolution: {key: float(key.split(':')[-1])
                             for key in self.aspect_ratios.keys() if key.startswith(f'{resolution}:')}
                for resolution in resolutions}

    def find_closest_ratio(self, ratio, resolution=None):
        """The bucket key of the aspect ratio table closest to `ratio`, at `resolution`
        (the largest one being trained at by default)."""
        tables = self.resolution_buckets()
        candidates = tables[resolution] if resolution in tables else next(iter(tables.values()))
        return min(candidates, key=lambda key: abs(candidates[key] - ratio))

    def assign_bucket(self, ratio, rng=random):
        """The bucket of an image of aspect ratio `ratio`, at a resolution drawn at random."""
        resolution = rng.choice(list(self.resolution_buckets().keys()))
        return self.find_closest_ratio(ratio, resolution)

    # by default, we enable xformers
    def enable_efficient_attention(self):
        self.pipe.enable_xformers_memory_efficient_attention()
        
    def initialize(self):
        # override aspect ratios: one table per resolution to train at
        resolutions = self.params.aspect_ratios
        if resolutions != None:
            self.resolutions = list(resolutions)
            tables = {}
            for resolution in self.resolutions:
                if resolution == 512:
                    table = copy.deepcopy(ASPECT_RATIO_512_BIN)
                else:
                    scale = resolution / 1024
                    table = copy.deepcopy(ASPECT_RATIO_1024_BIN)
                    if resolution != 1024:
                        for key in table.keys():
                            table[key] = [float(math.floor(table[key][0] * scale)), float(math.floor(table[key][1] * scale))]
                tables[resolution] = table
            if len(tables) == 1:
                self.aspect_ratios = next(iter(tables.values()))
            else:
                self.aspect_ratios = {f'{resolution}:{key}': size
                                      for resolution, table in tables.items() for key, size in table.items()}
                
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
        
        if params.dreambooth_instance_tar != None:
            # the dual gpu iterator reads a single stream of shards, it cannot alternate the
            # instance and the class batches
            if params.dual_gpu:
                raise ValueError('dreambooth cannot be combined with dual_gpu')
            shards = [f'shard-{shard_index:06d}.tar' for shard_index in range(self.shard_index_begin, self.shard_index_end)]
            self.sampler = BucketSamplerDreambooth(
                params.dreambooth_instance_tar,
                params.dreambooth_regularization_folder,
                params.dreambooth_instance,
                params.dreambooth_class,
                self.accelerator,
                params.batch_size,
                params.vae_max_batch_size,
                params.text_encoder_max_batch_size,
                self,
                params.dataset_seed,
                cache_size=4,
                r2_access_key=self.params.r2_access_key,
                r2_bucket_name=self.params.r2_bucket_name,
                r2_endpoint=self.params.r2_endpoint,
                r2_secret_key=self.params.r2_secret_key,
                shards=shards,
                dreambooth_reg_tar=params.dreambooth_reg_tar,
                # the instance tar carries the references, the regularization one may or may not
                train_lw_controlnet=params.train_lw_controlnet,
            )
        elif params.train_lw_controlnet:
            # the dual gpu iterator builds its batches inline and ships them between processes,
            # bypassing the hook that attaches the reference features
            if params.dual_gpu:
                raise ValueError('train_lw_controlnet cannot be combined with dual_gpu')

            # the same shard plumbing as a regular run, with the src/ref pairing on top. Local
            # tars are the usual case here, since a reference dataset is built by hand.
            shards = [f'shard-{shard_index:06d}.tar' for shard_index in range(self.shard_index_begin, self.shard_index_end)]
            self.sampler = BucketSamplerLwControlNet(shards,
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
                            local_paths=self.params.local_shard_paths,
                            lw_controlnet_tars=params.lw_controlnet_tars)
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

        # run the vae and the text encoder once over a pool of samples, write the features to disk,
        # then train from those so both models stay out of the vram during the training steps
        if self.params.precompute_features:
            if self.params.dual_gpu:
                raise ValueError('precompute_features cannot be combined with dual_gpu')
            self.features_precompute = FeaturesPrecompute(self)
            self.sampler = self.features_precompute.run(self.sampler)
        elif self.params.precompute_text_embeddings:
            # only the text embeddings are cached, the vae stays loaded and encodes every step
            if self.params.dual_gpu:
                raise ValueError('precompute_text_embeddings cannot be combined with dual_gpu')
            self.features_precompute = TextEmbeddingsPrecompute(self)
            self.sampler = self.features_precompute.run(self.sampler)

        # check for lora training
        if self.params.peft is not None:
            patch_lycoris_for_bnb_4bit()
            if self.params.lora_pretrained == None:
                # the peft section of the yaml goes straight to peft's config loader
                config = PeftConfig.from_peft_type(**self.params.peft)
                self.model = get_peft_model(self.model, config,
                                            autocast_adapter_dtype=self.params.peft_autocast_adapter_dtype)
            else:
                self.model = PeftModel.from_pretrained(self.model, params.lora_pretrained, is_trainable=True,
                                                       autocast_adapter_dtype=self.params.peft_autocast_adapter_dtype)
            self.model.print_trainable_parameters()

        # the control path goes on after the adapter exists, so that the hook of a layer peft has
        # replaced lands on the lora layer and not on the Linear underneath it - the offset is meant
        # to be added to the layer's whole output, the adapter included
        if self.params.train_lw_controlnet:
            # a lightweight controlnet run trains the control path, so the model it is attached to
            # is frozen. A lora run is already there: peft froze the base weights and left the
            # adapter trainable, and which of the two learns on a given batch is then
            # lw_controlnet_freeze_path's call. A run with no adapter has a fully trainable model
            # underneath instead, and a full finetune with a control path bolted onto it is not the
            # cheap thing this is for - so it is frozen here, and the control path is the only thing
            # the optimizer is given.
            if self.params.peft is None:
                frozen = 0
                for parameter in self.model.parameters():
                    if parameter.requires_grad:
                        parameter.requires_grad_(False)
                        frozen += 1
                print(f'froze {frozen} base model tensors: the control path is the only thing training')
            self.reference_control = lw_controlnet.attach_control(self.model, self.reference_config,
                                                                  device=self.accelerator.device)
            if self.params.lw_controlnet_pretrained != None:
                lw_controlnet.load_control(self.reference_control, self.params.lw_controlnet_pretrained)
                print(f'resumed the control path from {self.params.lw_controlnet_pretrained}')
            print(f'hooked {len(self.reference_control.layers)} layers, '
                  f'{self.reference_control.num_trainable_parameters()} control path parameters')

        # only what actually learns. For a full finetune this is every parameter and changes
        # nothing; for a lora or a control path it keeps the frozen base weights out of the
        # optimizer, which is what stops an 8bit optimizer from being handed a few billion tensors
        # it will never step
        params_to_optimizer = [parameter for parameter in self.model.parameters()
                               if parameter.requires_grad]

        if self.params.optimizer is not None:
            self.optimizer = self._build_optimizer(params_to_optimizer)
        elif self.params.use_adamw_8bit == False:
            self.optimizer = torch.optim.AdamW(params_to_optimizer,
                                                lr=params.learning_rate,
                                                weight_decay=params.weight_decay)
        else:
            self.optimizer = bitsandbytes.optim.Lion8bit(params=params_to_optimizer, lr=params.learning_rate, weight_decay=params.weight_decay)

        if self.params.offload_to_cpu:
            if self.params.dual_gpu:
                raise ValueError('offload_to_cpu cannot be combined with dual_gpu')
            self.offload = offload_frozen_blocks(self.model, self.accelerator.device,
                                                 self.params.offload_percentage)

        if self.params.dual_gpu == False:
            if self.params.offload_to_cpu:
                # DistributedDataParallel refuses a module that spans the cpu and the gpu, and
                # prepare would move the whole model back to the gpu anyway. The model stays as it
                # is: only the adapter trains, so the processes start from the same weights and
                # average its gradients by hand, which is all DDP would have done here.
                self.optimizer, self.sampler = self.accelerator.prepare(self.optimizer, self.sampler)
                self.sync_trainable_parameters()
            else:
                self.optimizer, self.model, self.sampler = self.accelerator.prepare(
                    self.optimizer, self.model, self.sampler)

        self.load_optimizer_state()

        self.lr_schedulers = []
        self.loss_ema = None
        self.loss_ema_decay = self._resolve_loss_ema_decay()
        if self.params.lr_scheduler is not None:
            self.lr_schedulers = self._build_lr_schedulers()

        # apply AveragedModel (EMA / SWA)
        self.ema_model = None
        if self.params.averaged_model is not None:
            self.ema_model = self._build_averaged_model()
        
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

    def trainable_parameters(self):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def sync_trainable_parameters(self):
        """Give every process the weights of the first one, the way DDP does when it wraps a model."""
        if self.accelerator.num_processes == 1:
            return
        with torch.no_grad():
            for parameter in self.trainable_parameters():
                torch.distributed.broadcast(parameter.data, src=0)

    def average_gradients(self):
        """The all reduce DDP would do during the backward, for a model that is not wrapped in it."""
        if self.accelerator.num_processes == 1:
            return
        gradients = [parameter.grad for parameter in self.trainable_parameters() if parameter.grad is not None]
        for gradient in gradients:
            torch.distributed.all_reduce(gradient)
            gradient /= self.accelerator.num_processes

    def gpu_resident(self):
        """A context that has the whole model on the gpu, for the validation images."""
        offload = getattr(self, 'offload', None)
        return offload.on_device() if offload != None else contextlib.nullcontext()

    def validate(self):
        raise NotImplemented

    def optimize(self, ratio, latents, embeddings, repa_tokens, generator):
        raise NotImplemented

    def finalize(self):
        pass

    def repa_loss(self, batch):
        return 0.0

    _OPTIMIZER_MODULES = (
        'torch.optim',
        'bitsandbytes.optim',
        'torchao.optim',
    )

    @classmethod
    def _resolve_optimizer_class(cls, opt_type):
        """Resolves a bare optimizer name (searching torch.optim, bitsandbytes.optim, torchao.optim)
        or an explicit dotted class path to a `torch.optim.Optimizer` subclass."""
        if '.' in opt_type:
            module_name, cls_name = opt_type.rsplit('.', 1)
            return getattr(importlib.import_module(module_name), cls_name)
        for module_name in cls._OPTIMIZER_MODULES:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            candidate = getattr(module, opt_type, None)
            if (isinstance(candidate, type)
                    and issubclass(candidate, torch.optim.Optimizer)
                    and candidate is not torch.optim.Optimizer):
                return candidate
        raise ValueError(
            f'Unknown optimizer {opt_type!r}: not found in {", ".join(cls._OPTIMIZER_MODULES)}; '
            f'use a dotted class path such as torch.optim.AdamW'
        )

    def _build_optimizer(self, params_to_optimizer):
        """Instantiates the optimizer named in `params.optimizer` with its remaining keys as kwargs."""
        opt_cfg = dict(self.params.optimizer)
        opt_type = opt_cfg.pop('type')
        opt_cls = self._resolve_optimizer_class(opt_type)
        init_params = inspect.signature(opt_cls.__init__).parameters
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values())
        if 'lr' in init_params or accepts_kwargs:
            opt_cfg.setdefault('lr', self.params.learning_rate)
        if 'weight_decay' in init_params or accepts_kwargs:
            opt_cfg.setdefault('weight_decay', self.params.weight_decay)
        return opt_cls(params_to_optimizer, **opt_cfg)

    def _build_single_lr_scheduler(self, cfg):
        """Instantiates one LR scheduler dict, recursively building nested `schedulers` if present."""
        sched_cfg = dict(cfg)
        sched_type = sched_cfg.pop('type')
        if isinstance(sched_cfg.get('schedulers'), list):
            sched_cfg['schedulers'] = [
                self._build_single_lr_scheduler(child) for child in sched_cfg['schedulers']
            ]
        sched_path = sched_type if '.' in sched_type else f'torch.optim.lr_scheduler.{sched_type}'
        module_name, cls_name = sched_path.rsplit('.', 1)
        sched_cls = getattr(importlib.import_module(module_name), cls_name)
        if sched_cls is torch.optim.lr_scheduler.ChainedScheduler:
            return sched_cls(**sched_cfg)
        return sched_cls(self.optimizer, **sched_cfg)

    def _build_lr_schedulers(self):
        """Instantiates the list of LR schedulers from `params.lr_scheduler`."""
        configs = self.params.lr_scheduler if isinstance(self.params.lr_scheduler, list) else [self.params.lr_scheduler]
        return [self._build_single_lr_scheduler(cfg) for cfg in configs]

    def _resolve_loss_ema_decay(self):
        """Derives the loss EMA decay from `params.averaged_model` (falling back to 0.99)."""
        if isinstance(self.params.averaged_model, dict):
            for fn_key in ('multi_avg_fn', 'avg_fn'):
                fn_cfg = self.params.averaged_model.get(fn_key)
                if isinstance(fn_cfg, dict) and 'decay' in fn_cfg:
                    return float(fn_cfg['decay'])
        return 0.99

    def _build_averaged_model(self):
        """Instantiates `AveragedModel` from `params.averaged_model`."""
        cfg = dict(self.params.averaged_model)
        for fn_key in ('avg_fn', 'multi_avg_fn'):
            if isinstance(cfg.get(fn_key), dict):
                fn_cfg = dict(cfg[fn_key])
                fn_type = fn_cfg.pop('type')
                fn_path = fn_type if '.' in fn_type else f'torch.optim.swa_utils.{fn_type}'
                module_name, attr_name = fn_path.rsplit('.', 1)
                fn_factory = getattr(importlib.import_module(module_name), attr_name)
                cfg[fn_key] = fn_factory(**fn_cfg)
        cfg.setdefault('device', self.accelerator.device)
        return AveragedModel(self.accelerator.unwrap_model(self.model), **cfg)

    def _swap_in_ema_weights(self):
        """Stores current unwrapped model parameters on CPU and copies `ema_model` parameters in."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        saved = []
        with torch.no_grad():
            for p, ema_p in zip(unwrapped.parameters(), self.ema_model.module.parameters()):
                saved.append(p.data.detach().clone().cpu())
                p.data.copy_(ema_p.data.to(device=p.device, dtype=p.dtype))
        return saved

    def _restore_ema_weights(self, saved):
        """Restores the original unwrapped model parameters saved by `_swap_in_ema_weights`."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            for p, orig in zip(unwrapped.parameters(), saved):
                p.data.copy_(orig.to(device=p.device, dtype=p.dtype))

    def _sync_ema_parameters(self):
        """Averages trainable floating-point EMA parameters across processes, staging CPU tensors on the accelerator device."""
        for param in self.ema_model.module.parameters():
            if not param.requires_grad or not param.is_floating_point():
                continue
            orig_device = param.device
            tensor = param.data if orig_device == self.accelerator.device else param.data.to(self.accelerator.device)
            reduced = self.accelerator.reduce(tensor, reduction="mean")
            param.data.copy_(reduced.to(orig_device))

    def optimizer_state_path(self, directory):
        return os.path.join(directory, 'optimizer.pt')

    def save_optimizer_state(self):
        """Writes the optimizer state next to the model that save_model just wrote."""
        directory = f'models/{self.global_step}'
        os.makedirs(directory, exist_ok=True)
        torch.save(self.optimizer.state_dict(), self.optimizer_state_path(directory))

    def load_optimizer_state(self):
        """When resuming from a saved model, loads the optimizer state saved beside it, if any."""
        for source in (self.params.lora_pretrained, self.params.lw_controlnet_pretrained,
                       self.params.pretrained_model_path):
            if source is None or not os.path.isdir(source):
                continue
            path = self.optimizer_state_path(source)
            if not os.path.isfile(path):
                continue
            try:
                self.optimizer.load_state_dict(torch.load(path, map_location='cpu'))
            except Exception as e:
                print(f'[Warning] could not load the optimizer state from {path}: {e}')
                return
            # the saved groups carry the learning rate (and initial_lr written by any prior
            # scheduler) of the step they were saved at, which may be mid warmup or from a
            # different lr config: the current configuration decides
            for group in self.optimizer.param_groups:
                group['lr'] = self.params.learning_rate
                group['weight_decay'] = self.params.weight_decay
                group.pop('initial_lr', None)
            print(f'resumed the optimizer state from {path}')
            return

    def save_model(self):
        directory = f'models/{self.global_step}'
        model = self.accelerator.unwrap_model(self.model)
        if self.reference_control == None:
            model.save_pretrained(directory)
            return

        # the control path never replaced anything in the model, so the adapter is an ordinary peft
        # checkpoint with nothing to undo first. It is only detached for the save so that it cannot
        # end up inside the model's own state dict, and it lands next to the adapter as
        # fourier_control.pt plus its config. That is exactly the layout infer_krea2_controlnet.py
        # reads back with --control_pretrained, so a validation checkpoint can be used for inference
        # as it is.
        #
        # A run with no adapter has nothing else to write: the base model is frozen and byte for byte
        # what it was loaded from, so saving it would copy several gigabytes of unchanged weights
        # next to every checkpoint.
        if self.params.peft is not None:
            with lw_controlnet.detached_control(model):
                model.save_pretrained(directory)
        lw_controlnet.save_control(self.reference_control, directory)
    
    def batch_reference(self, batch):
        """The reference conditioning to run one batch under, as a context manager.

        A lightweight controlnet run sets the conditioning around the whole step: the hooks on the
        controlled layers read it from the shared context, so nothing has to be threaded through the
        model specific optimize(). A batch that carries no reference - one that came from a shard
        without a ref/ folder - leaves the reference block of every spectrum at zero, which is plain
        fourierft, and that is also what the reference dropout produces on the batches that do carry
        one.

        Under lw_controlnet_freeze_path the batches that do run with a reference train the control
        path alone: every other weight is frozen for the whole forward and backward, so the only
        thing that can account for the reference is the control path itself, rather than the lora
        absorbing it and the control path being left with nothing to learn. The batches without a
        reference - the plain shards, the class batches of a dreambooth run, and the ones the
        dropout turned into plain fourierft steps - keep training the diffusion path as before.
        """
        if self.reference_control == None:
            return contextlib.nullcontext()

        reference = batch.reference_features
        if reference != None and random.random() < self.params.lw_controlnet_dropout_prob:
            reference = None
        if reference != None:
            reference = lw_controlnet.features_to(reference, self.accelerator.device)

        # the freeze has to be entered before the reference context, so that requires_grad is
        # already off when the forward builds its graph, and lifted only once the backward that
        # runs inside both of them is done
        context = contextlib.ExitStack()
        if reference != None and self.params.lw_controlnet_freeze_path:
            context.enter_context(lw_controlnet.frozen_diffusion_path(self.model,
                                                                      self.reference_control))
        context.enter_context(self.reference_control.reference(reference))
        return context

    def batch_backward(self, batch, scale=1.0):
        """Forward and backward one batch, and return its loss detached.

        The backward pass runs inside the reference context on purpose. Gradient checkpointing
        replays the checkpointed blocks during backward, and the adapters read the conditioning
        from the shared context at that moment too: with the context already closed they would
        take the plain lora path and rebuild a smaller graph than the one the forward saved, which
        is exactly the tensor count mismatch torch.utils.checkpoint refuses.
        """
        self.apply_unconditional_dropout(batch)
        with self.batch_reference(batch):
            activation_offload = offload_activations() if self.params.offload_to_cpu else contextlib.nullcontext()
            with activation_offload:
                loss = self.diffusion_loss(batch)
            self.accelerator.backward(loss if scale == 1.0 else loss * scale)
        return loss.detach()

    def apply_unconditional_dropout(self, batch):
        """Replace the captions of a batch by the empty embedding, at train_unconditional_prob.

        This used to sit at the top of diffusion_loss. It runs before batch_reference now because
        the reference path reads the text length of the batch to find where the image tokens start
        in a joint sequence, and swapping the embeddings afterwards would leave it resolving
        against a length the forward pass no longer runs with.
        """
        if random.random() >= self.params.train_unconditional_prob:
            return
        for idx in range(len(batch.embeddings)):
            batch.embeddings[idx] = self.empty_embeddings[0]

    def diffusion_loss(self, batch):
        params = self.params
        ratio = batch.ratio
        latents = batch.vae_features
        embeddings = batch.embeddings
        repa_features = batch.repa_features

        # the timesteps and the noise are drawn from here. A generator made fresh on every batch starts
        # from the same default seed, which trained every step at the same timestep with the same noise
        generator = self.noise_generator
        if params.exploration_steps != None:
            with torch.no_grad():
                states = []
                losses = []
                for step in range(params.exploration_steps):
                    states.append(generator.get_state())
                    loss = self.optimize(ratio, latents, embeddings, repa_features, generator)
                    losses.append(loss)

            min_loss_idx = torch.argmin(torch.stack(losses))
            generator.set_state(states[min_loss_idx])
        return self.optimize(ratio, latents, embeddings, repa_features, generator)

    def run(self):
        params = self.params
        self.initialize()
        loss_fn = torch.nn.MSELoss()
        progress_bar = tqdm.tqdm(total=params.steps, desc='Num Steps')
        avg_loss = torch.tensor(0, device=self.accelerator.device)
        avg_instance_loss = torch.tensor(0.0, device=self.accelerator.device)
        avg_class_loss = torch.tensor(0.0, device=self.accelerator.device)

        # extract empty embedding
        # the precompute pass already did it, back when the text encoder was still loaded
        self.accelerator.wait_for_everyone()
        if self.features_precompute == None:
            with torch.no_grad():
                self.empty_embeddings = self.extract_embeddings([''])

        while self.global_step < self.params.steps:
            # then go through the cache items
            batches = iter(self.sampler)
            for batch in batches:
                with self.accelerator.accumulate(self.model):
                    if batch.is_regularization == None:
                        #if self.params.use_repa:
                            #loss = loss + self.params.repa_lambda * self.repa_loss(batch)

                        avg_loss = avg_loss + self.batch_backward(batch)
                    else:
                        # dreambooth: an instance batch and the class batch the sampler yields right
                        # after it make a single step, on loss = instance_loss + lambda * class_loss
                        class_batch = next(batches)
                        if batch.is_regularization or not class_batch.is_regularization:
                            raise RuntimeError('the dreambooth sampler must alternate instance and class batches')

                        # one backward pass per batch accumulates exactly the gradient of the sum,
                        # without holding both graphs in memory at once. The gpus only exchange the
                        # gradients once, after the second pass
                        with self.accelerator.no_sync(self.model):
                            instance_loss = self.batch_backward(batch)
                        class_loss = self.batch_backward(class_batch, self.params.dreambooth_lambda)

                        avg_loss = avg_loss + instance_loss + self.params.dreambooth_lambda * class_loss
                        avg_instance_loss = avg_instance_loss + instance_loss
                        avg_class_loss = avg_class_loss + class_loss

                    if self.accelerator.sync_gradients:
                        if self.params.offload_to_cpu:
                            self.average_gradients()
                        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                        if self.ema_model is not None:
                            self.ema_model.update_parameters(self.accelerator.unwrap_model(self.model))

                        for sched in self.lr_schedulers:
                            if not isinstance(sched, ReduceLROnPlateau):
                                sched.step()

                        self.optimizer.zero_grad()
                
                if self.accelerator.sync_gradients:
                    mean_loss = self.accelerator.gather(avg_loss).mean()
                    avg_loss = torch.tensor(0, device=self.accelerator.device)

                    loss_val = mean_loss.item()
                    if self.ema_model is not None:
                        if self.loss_ema is None:
                            self.loss_ema = loss_val
                        else:
                            self.loss_ema = self.loss_ema_decay * self.loss_ema + (1.0 - self.loss_ema_decay) * loss_val

                    metric_schedulers = [s for s in self.lr_schedulers if isinstance(s, ReduceLROnPlateau)]
                    if metric_schedulers:
                        metric = self.loss_ema if self.loss_ema is not None else loss_val
                        metric_name = 'loss_ema' if self.loss_ema is not None else 'loss'
                        prev_lr = self.optimizer.param_groups[0]['lr']
                        for sched in metric_schedulers:
                            sched.step(metric)
                        new_lr = self.optimizer.param_groups[0]['lr']
                        if new_lr < prev_lr:
                            if prev_lr > 0:
                                ratio = new_lr / prev_lr
                                for sched in self.lr_schedulers:
                                    if hasattr(sched, 'base_lrs'):
                                        sched.base_lrs = [lr * ratio for lr in sched.base_lrs]
                            if self.accelerator.is_main_process:
                                print(f'[LRScheduler] step {self.global_step}: lr {prev_lr:.3e} -> {new_lr:.3e} ({metric_name}={metric:.5f})')

                    # dreambooth: both halves of the loss, to see which one is not converging
                    stream_losses = {}
                    if self.params.dreambooth_instance_tar != None:
                        stream_losses = {
                            'train/instance_loss': self.accelerator.gather(avg_instance_loss).mean(),
                            'train/class_loss': self.accelerator.gather(avg_class_loss).mean(),
                        }
                        avg_instance_loss = torch.tensor(0.0, device=self.accelerator.device)
                        avg_class_loss = torch.tensor(0.0, device=self.accelerator.device)

                    if self.logger != None and self.accelerator.is_main_process:
                        try:
                            self.logger.add_scalar('train/loss', mean_loss.item(), self.global_step)
                            if self.loss_ema is not None:
                                self.logger.add_scalar('train/loss_ema', self.loss_ema, self.global_step)
                            for name, value in stream_losses.items():
                                self.logger.add_scalar(name, value.item(), self.global_step)
                            if self.lr_schedulers:
                                self.logger.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                        except Exception as e:
                            print(f"[Warning] TensorBoard logging failed: {e}")

                    if self.global_step % params.num_steps_per_validation == 0:
                        with torch.no_grad():
                            # ✅ Run reduction on ALL processes to sync EMA parameters
                            if self.ema_model is not None and self.params.dual_gpu == False and self.accelerator.num_processes > 1:
                                self._sync_ema_parameters()
                    
                            # ✅ Ensure store() is called before restore()
                            if self.accelerator.is_main_process:
                                saved_weights = None
                                if self.ema_model is not None:
                                    saved_weights = self._swap_in_ema_weights()
                                
                                # we need to check if we temporarly disable the peft adapter for the first inference step
                                if len(self.timesteps) != 0:
                                    if 0 in self.timesteps:
                                        rescale_adapter_scale(self.model, 1.0)
                                    else:
                                        rescale_adapter_scale(self.model, 0.0)

                                # for now, validating with this patch is disabled
                                if self.params.dual_gpu == False:
                                    # give the sampling all the room it can get: the gradients
                                    # have been applied already, and the allocator is holding on
                                    # to blocks too small for what the validation asks for
                                    self.optimizer.zero_grad(set_to_none=True)
                                    gc.collect()
                                    torch.cuda.empty_cache()

                                    # the vae is parked on the cpu between validations
                                    if self.features_precompute != None:
                                        self.features_precompute.before_validation()
                                    self.validate()
                                    if self.features_precompute != None:
                                        self.features_precompute.after_validation()
                                    gc.collect()
                                    torch.cuda.empty_cache()

                                if len(self.timesteps) != 0:
                                    rescale_adapter_scale(self.model, 1.0)
                                self.save_model()
                                self.save_optimizer_state()
                    
                                if saved_weights is not None:
                                    self._restore_ema_weights(saved_weights)
                    progress_bar.update(1)
                    self.global_step = self.global_step + 1

                    #if self.params.dual_gpu == False:
                        #self.accelerator.wait_for_everyone()
            # if hasattr(self, 'repa_model'):
            #     self.repa_model.cpu()