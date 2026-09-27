import json
import yaml
import huggingface_hub


def read_flag(yaml_root, key, default=False):
    """A yaml entry meant as a boolean.

    The configs are read with the BaseLoader, so every scalar comes back as a string: 'True' and
    'False' are both truthy on their own. Most of the flags of this file are presence based, but a
    parameter documented as a boolean has to honour 'false' as well.
    """
    if key not in yaml_root.keys():
        return default
    value = yaml_root[key]
    if isinstance(value, str):
        return value.strip().lower() not in ('false', 'no', 'off', '0', '')
    return bool(value)


def read_int(yaml_root, key, default=None):
    return int(yaml_root[key]) if key in yaml_root.keys() else default


def read_float(yaml_root, key, default=None):
    return float(yaml_root[key]) if key in yaml_root.keys() else default


class TrainingParameters:
    def __init__(self):
        self.r2_endpoint = None
        self.r2_access_key = None
        self.r2_secret_key = None
        self.r2_bucket_name = None
        self.r2_tar_files = None
        self.dataset_seed = 0

        self.batch_size = 4
        self.pretrained_pipe_path = None
        self.pretrained_model_path = None
        self.learning_rate = None
        self.steps = None
        self.num_steps_per_validation = None
        self.validation_prompts = None
        self.urls = None
        self.bfloat16 = None
        self.gradient_accumulation_steps = None
        self.output_repo = None
        self.use_preservation = None
        self.preservation_ratio = None
        self.url_probs = None
        self.lora_base_model_8bit = None
        self.lora_base_model_4bit = None
        self.use_adamw_8bit = False

        # lora
        self.lora_target_modules = None
        self.lora_rank = None
        self.lora_alpha = None
        self.lora_dropout = None
        self.lora_bias = None
        self.lora_use_rslora = None
        self.lora_pretrained = None

        # keep the lora/dora adapters in bfloat16 instead of the float32 peft casts them to
        self.lora_adapter_bf16 = False

        # lr scheduler
        self.cyclic_lr_max_lr = None
        self.cyclic_lr_step_size_up = None
        self.cyclic_lr_step_size_down = None
        self.cylic_lr_mode = None 

        # low vram
        self.low_vram = None
        self.pretrained_pipe_single_file = None

        self.dual_gpu = False

        # timesteps, this is used for a refiner model
        self.timesteps = []

        # aspect ratios override
        self.aspect_ratios = None

        # from the paper explorative modeling: unlocking a third pretraining axis and end-to-end generation
        self.exploration_steps = None

        # lightweight controlnet: a fourierft style half-spectrum per layer, part of it learned and
        # part of it written by a projection of the reference image features
        self.train_lw_controlnet = False
        self.lw_controlnet_tars = None
        self.lw_controlnet_pretrained = None
        self.lw_controlnet_adapter_name = 'default'
        self.lw_controlnet_reference_model = 'facebook/dinov2-base'
        self.lw_controlnet_reference_dim = None
        self.lw_controlnet_target_modules = None
        self.lw_controlnet_cond_dim = 256
        self.lw_controlnet_pool_grid = 2
        self.lw_controlnet_n_frequency = None
        self.lw_controlnet_frequency_ratio = 0.5
        self.lw_controlnet_n_reference = None
        self.lw_controlnet_reference_ratio = 0.25
        self.lw_controlnet_spectrum_scaling = 1.0
        self.lw_controlnet_reference_scaling = 1.0
        self.lw_controlnet_ifft_norm = 'ortho'
        self.lw_controlnet_random_loc_seed = 777
        self.lw_controlnet_dtype = 'float32'
        self.lw_controlnet_dropout_prob = 0.1
        self.lw_controlnet_freeze_path = False
        self.lw_controlnet_validation_references = None

        # precalculate the vae/text encoder features once at startup, then train without them
        self.precompute_features = False
        self.precompute_size = None
        self.precompute_cache_dir = 'precomputed_features'
        self.precompute_force = False
    
    def read_yaml(self, file):
        with open(file) as f:
            contents = f.read()
        yaml_root = yaml.load(contents, Loader=yaml.BaseLoader)
        if 'r2_endpoint' in yaml_root.keys():
            self.r2_endpoint = yaml_root['r2_endpoint']
            self.r2_access_key = yaml_root['r2_access_key']
            self.r2_secret_key = yaml_root['r2_secret_key']
            self.r2_bucket_name = yaml_root['r2_bucket_name']

            self.r2_tar_files = yaml_root['r2_tar_files'] if 'r2_tar_files' in yaml_root.keys() else None
        else:
            self.urls = yaml_root['urls']
        
        self.warmup_steps = int(yaml_root['warmup_steps']) if 'warmup_steps' in yaml_root.keys() else None
        
        self.compute_features = False
        if 'compute_features' in yaml_root.keys():
            self.compute_features = True
            self.vae_max_batch_size = int(yaml_root['vae_max_batch_size'])
            self.text_encoder_max_batch_size = int(yaml_root['text_encoder_max_batch_size'])
        
        self.num_shards = int(yaml_root['num_shards']) if 'num_shards' in yaml_root.keys() else None
        self.r2_dataset_folder = yaml_root['r2_dataset_folder'] if 'r2_dataset_folder' in yaml_root.keys() else None
        self.r2_upload_key = None
        if 'r2_upload_key' in yaml_root.keys():
            self.r2_upload_key = yaml_root['r2_upload_key']
            self.r2_upload_shard_size = yaml_root['r2_upload_shard_size']
        
        if 'url_probs' in yaml_root.keys():
            self.url_probs = [float(prob) for prob in yaml_root['url_probs']]
        
        if 'dataset_seed' in yaml_root.keys():
            self.dataset_seed = int(yaml_root['dataset_seed'])
        self.extract_features = 'extract_features' in yaml_root.keys()
        
        self.batch_size = int(yaml_root['batch_size'])
        
        if 'pretrained_model_path' in yaml_root.keys():
            self.pretrained_model_path = yaml_root['pretrained_model_path']
        
        # this is used for loading the sd1.5 compvis safetensors format
        if 'pretrained_pipe_single_file' in yaml_root.keys():
            self.pretrained_pipe_single_file = yaml_root['pretrained_pipe_single_file']
        if 'pretrained_pipe_path' in yaml_root.keys():
            self.pretrained_pipe_path = yaml_root['pretrained_pipe_path']
        
        self.learning_rate = float(yaml_root['learning_rate'])
        self.steps = int(yaml_root['steps'])
        self.num_steps_per_validation = int(yaml_root['num_steps_per_validation'])
        self.validation_prompts = yaml_root['validation_prompts']
        self.dual_gpu = 'dual_gpu' in yaml_root.keys()

        # training timesteps
        # this should be a list of integers
        if 'timesteps' in yaml_root.keys():
            self.timesteps = yaml_root['timesteps']

        # precalculate latents and embeddings to save swapping models to speed up training
        if 'cache_size' in yaml_root.keys():
            self.cache_size = int(yaml_root['cache_size'])
        else:
            self.cache_size = 1000
        
        if 'weight_decay' in yaml_root.keys():
            self.weight_decay = float(yaml_root['weight_decay'])
        else:
            self.weight_decay = 0.0
        
        self.bfloat16 =  'bfloat16' in yaml_root.keys()
        
        if 'gradient_accumulation_steps' in yaml_root.keys():
            self.gradient_accumulation_steps = yaml_root['gradient_accumulation_steps']
        else:
            self.gradient_accumulation_steps = 1
        
        self.use_ema = False
        if 'use_ema' in yaml_root.keys():
            self.use_ema = True
        
        self.lora_use_dora = 'lora_use_dora' in yaml_root.keys()
        
        # train using unconditional steps probability
        self.train_unconditional_prob = 0.0
        if 'train_unconditional_prob' in yaml_root.keys():
            self.train_unconditional_prob = float(yaml_root['train_unconditional_prob'])
        # learning rate scheduler
        if 'cyclic_lr_max_lr' in yaml_root.keys():
            self.cyclic_lr_max_lr = float(yaml_root['cyclic_lr_max_lr'])
            self.cyclic_lr_step_size_up = 2000
            self.cyclic_lr_step_size_down = 2000
            self.cylic_lr_mode = 'triangular'

            if 'cyclic_lr_step_size_up' in yaml_root.keys():
                self.cyclic_lr_step_size_up = int(yaml_root['cyclic_lr_step_size_up'])
            if 'cyclic_lr_step_size_down' in yaml_root.keys():
                self.cyclic_lr_step_size_down = int(yaml_root['cyclic_lr_step_size_down'])
            if 'cylic_lr_mode' in yaml_root.keys():
                self.cylic_lr_mode = yaml_root['cylic_lr_mode']
        
        self.huggingface_dataset_repo = None
        if 'huggingface_dataset_repo' in yaml_root.keys():
            self.huggingface_dataset_repo = yaml_root['huggingface_dataset_repo']
        
        # option with the shards download locally (the most stable)
        self.local_shard_paths = yaml_root['local_shard_paths'] if 'local_shard_paths' in yaml_root.keys() else None
        self.use_adamw_8bit  = 'use_adamw_8bit' in yaml_root.keys()
        
        # lora training
        if 'lora_rank' in yaml_root.keys():
            if 'lora_pretrained' in yaml_root.keys():
                self.lora_pretrained = yaml_root['lora_pretrained']
            self.lora_target_modules = yaml_root['lora_target_modules']
            self.lora_rank = int(yaml_root['lora_rank'])
            self.lora_alpha = int(yaml_root['lora_alpha'])

            # load the base model in 8bit
            self.lora_base_model_8bit = True if 'lora_base_model_8bit' in yaml_root.keys() else False

            # load the base model in 4bit (bitsandbytes QLoRA-style quantization)
            self.lora_base_model_4bit = True if 'lora_base_model_4bit' in yaml_root.keys() else False

            if 'lora_dropout' in yaml_root.keys():
                self.lora_dropout = float(yaml_root['lora_dropout'])
            else:
                self.lora_dropout = 0.0
            self.lora_bias = 'lora_bias' in yaml_root.keys()
            self.lora_algo = yaml_root['lora_algo'] # locon, lora, loha, lokr, dylora, glora, full, diag-oft, boft

            # peft casts the adapters to float32 for numerical stability. Keeping them in
            # bfloat16 halves what dora has to materialize to compute its weight norm, which is
            # what makes dora fit at all on a small card
            self.lora_adapter_bf16 = 'lora_adapter_bf16' in yaml_root.keys()

            self.lora_use_rslora = 'lora_use_rslora' in yaml_root.keys()
            self.lora_use_dora = 'lora_use_dora' in yaml_root.keys()

        self.fourierft_alpha = float(yaml_root['fourierft_alpha']) if 'fourierft_alpha' in yaml_root.keys() else 0.01
        self.low_vram = 'low_vram' in yaml_root.keys()
        self.use_calculated_features = 'use_calculated_features' in yaml_root.keys()

        # REPA parameters
        self.use_repa = 'use_repa' in yaml_root.keys()
        self.repa_lambda = 0.05
        if 'repa_lambda' in yaml_root.keys():
            self.repa_lambda = float(yaml_root['repa_lambda'])

        self.repa_pretrained_model = None
        if 'repa_pretrained_model' in yaml_root.keys():
            self.repa_pretrained_model = yaml_root['repa_pretrained_model']
        
        self.save_to_disk = False
        if 'save_to_disk' in yaml_root.keys():
            self.save_to_disk = True
        
        self.bucket_repeat = 1
        if 'bucket_repeat' in yaml_root.keys():
            self.bucket_repeat = int(yaml_root['bucket_repeat'])

        # aspect ratios override
        self.aspect_ratios = int(yaml_root['aspect_ratio']) if 'aspect_ratio' in yaml_root.keys() else None

        # for training simple dreambooth loras
        if 'dreambooth_dataset_folder' in yaml_root.keys():
            # ignoring it would silently turn the run into a regular training on the dataset shards
            raise ValueError('dreambooth_dataset_folder was renamed to dreambooth_instance_tar')
        self.dreambooth_instance_tar = yaml_root['dreambooth_instance_tar'] if 'dreambooth_instance_tar' in yaml_root.keys() else None
        # the weight of the loss of the class (regularization) batches, the instance batches are left unweighted
        self.dreambooth_lambda = float(yaml_root['dreambooth_lambda']) if 'dreambooth_lambda' in yaml_root.keys() else 1.0
        self.dreambooth_instance = yaml_root['dreambooth_instance'] if 'dreambooth_instance' in yaml_root.keys() else None
        self.dreambooth_class = yaml_root['dreambooth_class'] if 'dreambooth_class' in yaml_root.keys() else None
        self.dreambooth_regularization_folder = yaml_root['dreambooth_regularization_folder'] if 'dreambooth_regularization_folder' in yaml_root.keys() else None
        # a single local tar holding the regularization images, takes precedence over dreambooth_regularization_folder
        self.dreambooth_reg_tar = yaml_root['dreambooth_reg_tar'] if 'dreambooth_reg_tar' in yaml_root.keys() else None
        if self.dreambooth_instance_tar != None and self.dreambooth_reg_tar == None and self.dreambooth_regularization_folder == None:
            raise ValueError('a dreambooth training needs dreambooth_reg_tar or dreambooth_regularization_folder')

        # exploration steps
        if 'exploration_steps' in yaml_root.keys():
            self.exploration_steps = int(yaml_root['exploration_steps'])

        # lightweight controlnet: a reference image goes through a frozen dinov2, and every
        # controlled layer gets a fourierft spectrum as long as its output dimension - one block of
        # it learned, one written by a projection of those features - whose inverse fft is added to
        # that layer's output. The shards that carry no reference leave the reference block at zero,
        # which is plain fourierft, so a single run can mix both kinds of tar.
        self.train_lw_controlnet = read_flag(yaml_root, 'train_lw_controlnet')
        if self.train_lw_controlnet:
            # local tars, read in a loop. Without them the usual shard/r2 plumbing is used
            self.lw_controlnet_tars = yaml_root['lw_controlnet_tars'] if 'lw_controlnet_tars' in yaml_root.keys() else None
            if isinstance(self.lw_controlnet_tars, str):
                self.lw_controlnet_tars = [self.lw_controlnet_tars]

            # a directory written by a previous run, to resume from or to finetune
            self.lw_controlnet_pretrained = yaml_root['lw_controlnet_pretrained'] if 'lw_controlnet_pretrained' in yaml_root.keys() else None
            self.lw_controlnet_adapter_name = yaml_root['lw_controlnet_adapter_name'] if 'lw_controlnet_adapter_name' in yaml_root.keys() else 'default'
            if 'lw_controlnet_reference_model' in yaml_root.keys():
                self.lw_controlnet_reference_model = yaml_root['lw_controlnet_reference_model']
            self.lw_controlnet_reference_dim = read_int(yaml_root, 'lw_controlnet_reference_dim')

            # which layers get a control path. None means every Linear and Conv2d of the model,
            # which is what fourierft does; a list is matched on the module name, the way peft
            # matches its own target_modules. These do not have to be lora layers - the path is a
            # forward hook on the layer, so it lands on whatever that layer's output is, the lora
            # included
            self.lw_controlnet_target_modules = yaml_root['lw_controlnet_target_modules'] if 'lw_controlnet_target_modules' in yaml_root.keys() else None

            # width of the reference summary vector the shared conditioner produces once per step.
            # It is the whole channel the reference has into the model: every controlled layer reads
            # it through its own cond_dim x n_reference projection
            self.lw_controlnet_cond_dim = read_int(yaml_root, 'lw_controlnet_cond_dim', 256)

            # the reference patch field is average pooled onto this many cells per side before being
            # projected into that vector, so a coarse layout survives instead of only a global mean.
            # 1 is the plain mean. It does not make the path spatial - the offset it writes is one
            # number per output channel and lands uniformly over the frame - only more descriptive
            self.lw_controlnet_pool_grid = read_int(yaml_root, 'lw_controlnet_pool_grid', 2)

            # how many BINS of the half-spectrum each side fills, as a fraction of the bin count
            # of the layer. Each bin is one complex coefficient, so a fraction f of the bins costs
            # 2 * f * bins parameters, which is f * out_features - the same size these ratios always
            # meant, the difference being that every one of those parameters is now independent.
            # The two sets are disjoint, so the fractions cannot add up to more than 1.
            # lw_controlnet_n_frequency / lw_controlnet_n_reference override them with an absolute
            # bin count, which is then clamped per layer so a 320 channel conv stays in range
            self.lw_controlnet_frequency_ratio = read_float(yaml_root, 'lw_controlnet_frequency_ratio', 0.5)
            self.lw_controlnet_reference_ratio = read_float(yaml_root, 'lw_controlnet_reference_ratio', 0.25)
            self.lw_controlnet_n_frequency = read_int(yaml_root, 'lw_controlnet_n_frequency')
            self.lw_controlnet_n_reference = read_int(yaml_root, 'lw_controlnet_n_reference')

            # the gain each side is written into the spectrum at. 1.0, not the 150 fourierft uses:
            # with norm='ortho' on a 1d transform the 1/sqrt(N) is cancelled by the ~N/2 bins that
            # contribute, so the offset already comes out at the scale of the parameters themselves
            # and there is nothing to compensate for. Fourierft's 150 belongs to a 2d transform whose
            # fill ratio is ~1e-3; ours is 0.5 to 0.75. At 150 one step at lr 5e-4 moves a layer by
            # 69% of its own output, which over a few hundred hooked layers is noise.
            # Both are plain floats at inference too: 0 on the reference one is exactly the model
            # with no reference image, 0 on both is the base model
            self.lw_controlnet_spectrum_scaling = read_float(
                yaml_root, 'lw_controlnet_spectrum_scaling', 1.0)
            self.lw_controlnet_reference_scaling = read_float(
                yaml_root, 'lw_controlnet_reference_scaling', 1.0)

            # normalization of that inverse fft: ortho (1/sqrt(n)), backward (1/n, torch's default)
            # or forward (none). ortho is what lets one scaling cover every width in the model
            if 'lw_controlnet_ifft_norm' in yaml_root.keys():
                self.lw_controlnet_ifft_norm = yaml_root['lw_controlnet_ifft_norm']

            # seed the per-layer bin placement is drawn from. Both sets are scattered over individual
            # bins of one permutation, the way fourierft picks its frequencies, so each reaches every
            # scale of the spectrum rather than occupying one band of neighbouring frequencies
            self.lw_controlnet_random_loc_seed = read_int(
                yaml_root, 'lw_controlnet_random_loc_seed', 777)

            # dtype of the control parameters. The inverse fft itself always runs in float32 - there
            # is no half precision kernel for it - and these are a few thousand numbers per layer,
            # so float32 costs nothing worth saving here
            if 'lw_controlnet_dtype' in yaml_root.keys():
                self.lw_controlnet_dtype = yaml_root['lw_controlnet_dtype']

            # keys of the cross attention architecture this replaced. Silently ignoring them would
            # leave a config that reads as if it still said something about the run
            retired = [key for key in ('lw_controlnet_cond_in_dim', 'lw_controlnet_cond_out_dim',
                                       'lw_controlnet_attention_heads',
                                       'lw_controlnet_attention_head_dim',
                                       'lw_controlnet_reference_scale',
                                       'lw_controlnet_reference_tokens_per_side',
                                       'lw_controlnet_resampler_heads',
                                       'lw_controlnet_spatial_conditioning',
                                       'lw_controlnet_index_layout')
                       if key in yaml_root.keys()]
            if retired:
                print(f'[Warning] ignoring {", ".join(retired)}: lw_controlnet_index_layout is gone '
                      f'because the bins are always scattered now, and the reference no longer '
                      f'reaches the '
                      f'layers through a cross attention on the lora rank tensor, it is written into '
                      f'a fourierft spectrum that is added to the layer output. The size knobs are '
                      f'lw_controlnet_cond_dim, lw_controlnet_reference_ratio and '
                      f'lw_controlnet_reference_scaling')

            # probability of dropping the reference on a step that has one, so the model also
            # learns the plain fourierft path and the reference can be guided against at inference
            self.lw_controlnet_dropout_prob = read_float(yaml_root, 'lw_controlnet_dropout_prob', 0.1)

            # on a batch that carries a reference, freeze everything but the control path. The lora
            # then learns the subject from the samples that have no reference, and the control path is
            # the only thing that can account for the reference on the ones that do, instead of the
            # lora absorbing it. The dropped references, and the shards with no ref/ folder at all,
            # train the lora as before.
            #
            # Unlike under the architecture this replaced, it is safe from step 0: the control path
            # adds to the layer output directly rather than reaching it through peft's zero
            # initialized lora_B, so its gradient does not wait on any other weight
            self.lw_controlnet_freeze_path = read_flag(yaml_root, 'lw_controlnet_freeze_path', False)

            # reference images paired with validation_prompts, cycled over when there are fewer
            self.lw_controlnet_validation_references = yaml_root['lw_controlnet_validation_references'] if 'lw_controlnet_validation_references' in yaml_root.keys() else None
            if isinstance(self.lw_controlnet_validation_references, str):
                self.lw_controlnet_validation_references = [self.lw_controlnet_validation_references]

            # no adapter is needed: the control path hooks the layer, whatever that layer turned
            # out to be, and the trainer freezes the base model itself when there is no lora to do
            # it. Leaving lora_rank out is then a control only run - the cheapest form of this, and
            # the one where the reference is the only thing that can account for anything.

            # the reference images are read from the tars, so the sampler has to be the one that
            # decodes images and captions rather than precalculated features
            self.compute_features = True
            if 'vae_max_batch_size' not in yaml_root.keys():
                self.vae_max_batch_size = 1
            if 'text_encoder_max_batch_size' not in yaml_root.keys():
                self.text_encoder_max_batch_size = 1

        # precalculate the vae and text encoder features at startup, write them to disk, then free
        # those models so they don't take vram during the training steps
        self.precompute_features = 'precompute_features' in yaml_root.keys()
        if self.precompute_features:
            # the number of samples to precalculate, the training then loops over that pool
            self.precompute_size = int(yaml_root['precompute_size']) if 'precompute_size' in yaml_root.keys() else self.cache_size
            if 'precompute_cache_dir' in yaml_root.keys():
                self.precompute_cache_dir = yaml_root['precompute_cache_dir']

            # recalculate even if a usable cache is already on disk
            self.precompute_force = 'precompute_force' in yaml_root.keys()

            # the precompute pass reads images and captions, so it needs the extracting sampler
            self.compute_features = True
            if 'vae_max_batch_size' not in yaml_root.keys():
                self.vae_max_batch_size = 1
            if 'text_encoder_max_batch_size' not in yaml_root.keys():
                self.text_encoder_max_batch_size = 1

if __name__ == '__main__':
    params = TrainingParameters()
    params.read_yaml('tests/config_sana.yaml')
