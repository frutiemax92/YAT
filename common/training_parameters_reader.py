import json
import yaml
import huggingface_hub

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

        # lora
        self.lora_target_modules = None
        self.lora_rank = None
        self.lora_alpha = None
        self.lora_dropout = None
        self.lora_bias = None
        self.lora_use_rslora = None
        self.lora_pretrained = None

        # lr scheduler
        self.cyclic_lr_max_lr = None
        self.cyclic_lr_step_size_up = None
        self.cyclic_lr_step_size_down = None
        self.cylic_lr_mode = None 

        # low vram
        self.low_vram = None
    
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
        else:
            self.pretrained_pipe_path = yaml_root['pretrained_pipe_path']
        
        self.learning_rate = float(yaml_root['learning_rate'])
        self.steps = int(yaml_root['steps'])
        self.num_steps_per_validation = int(yaml_root['num_steps_per_validation'])
        self.validation_prompts = yaml_root['validation_prompts']

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
            
        # lora training
        if 'lora_rank' in yaml_root.keys():
            if 'lora_pretrained' in yaml_root.keys():
                self.lora_pretrained = yaml_root['lora_pretrained']
            self.lora_target_modules = yaml_root['lora_target_modules']
            self.lora_rank = int(yaml_root['lora_rank'])
            self.lora_alpha = int(yaml_root['lora_alpha'])

            if 'lora_dropout' in yaml_root.keys():
                self.lora_dropout = float(yaml_root['lora_dropout'])
            else:
                self.lora_dropout = 0.0
            self.lora_bias = 'lora_bias' in yaml_root.keys()
            self.lora_algo = yaml_root['lora_algo'] # locon, lora, loha, lokr, dylora, glora, full, diag-oft, boft

            self.lora_use_rslora = 'lora_use_rslora' in yaml_root.keys()
            self.lora_use_dora = 'lora_use_dora' in yaml_root.keys()
            self.dreambooth_lambda = float(yaml_root['dreambooth_lambda']) if 'dreambooth_lambda' in yaml_root.keys() else None
        
        self.low_vram = 'low_vram' in yaml_root.keys()
        self.use_calculated_features = 'use_calculated_features' in yaml_root.keys()

        # REPA parameters
        self.use_repa = 'use_repa' in yaml_root.keys()
        self.repa_lambda = 0.5
        if 'repa_lambda' in yaml_root.keys():
            self.repa_lambda = float(yaml_root['repa_lambda'])
        
        self.save_to_disk = False
        if 'save_to_disk' in yaml_root.keys():
            self.save_to_disk = True
        
        self.bucket_repeat = 1
        if 'bucket_repeat' in yaml_root.keys():
            self.bucket_repeat = int(yaml_root['bucket_repeat'])

if __name__ == '__main__':
    params = TrainingParameters()
    params.read_yaml('tests/config_sana.yaml')
