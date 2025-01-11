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

        # lora
        self.lora_target_modules = None
        self.lora_rank = None
        self.lora_alpha = None
        self.lora_dropout = None
        self.lora_bias = None
        self.lora_use_rslora = None
        self.lora_pretrained = None

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
            self.r2_tar_files = yaml_root['r2_tar_files']
        else:
            self.urls = yaml_root['urls']
        
        self.batch_size = int(yaml_root['batch_size'])
        self.pretrained_pipe_path = yaml_root['pretrained_pipe_path']
        
        if 'pretrained_model_path' in yaml_root.keys():
            self.pretrained_model_path = yaml_root['pretrained_model_path']
        
        self.learning_rate = float(yaml_root['learning_rate'])
        self.steps = int(yaml_root['steps'])
        self.num_steps_per_validation = int(yaml_root['num_steps_per_validation'])
        self.validation_prompts = yaml_root['validation_prompts']
        
        if 'weight_decay' in yaml_root.keys():
            self.weight_decay = yaml_root['weight_decay']
        else:
            self.weight_decay = 0.0
        
        self.bfloat16 =  'bfloat16' in yaml_root.keys()
        
        if 'gradient_accumulation_steps' in yaml_root.keys():
            self.gradient_accumulation_steps = yaml_root['gradient_accumulation_steps']
        else:
            self.gradient_accumulation_steps = 1
        
        if 'use_preservation' in yaml_root.keys():
            self.use_preservation = True
            self.preservation_ratio = float(yaml_root['preservation_ratio'])
        else:
            self.use_preservation = False
        
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
        
        self.low_vram = 'low_vram' in yaml_root.keys()

if __name__ == '__main__':
    params = TrainingParameters()
    params.read_yaml('tests/config_sana.yaml')
