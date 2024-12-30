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
        
        if 'bfloat16' in yaml_root.keys():
            self.bfloat16 = bool(yaml_root['bfloat16'])
        else:
            self.bfloat16 = False
        
        if 'gradient_accumulation_steps' in yaml_root.keys():
            self.gradient_accumulation_steps = yaml_root['gradient_accumulation_steps']
        else:
            self.gradient_accumulation_steps = 1

if __name__ == '__main__':
    params = TrainingParameters()
    params.read_yaml('tests/config_sana.yaml')
