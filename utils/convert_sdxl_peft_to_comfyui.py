from safetensors.torch import load_file, save_file
from peft import PeftConfig
import torch
import argparse

def convert_sdxl_peft_to_comfyui(peft_model_path : str, comfyui_model_path : str):
    config = PeftConfig.from_pretrained(peft_model_path)

    # we need the alpha value
    alpha = config.lora_alpha

    state_dict = load_file(peft_model_path + '/adapter_model.safetensors')
    new_state_dict = {}
    lora_keys = set()

    def get_resnet_index(indices):
        res = 0
        res = res + indices[1]
        res = res + indices[0] * 3
        return res

    def get_input_resnet_indices(indices):
        first_index = 1
        second_index = 0

        first_index = first_index + indices[1]
        first_index = first_index + indices[0] * 3
        return first_index, second_index

    def get_output_resnet_indices(indices):
        first_index = 1
        second_index = 0

        first_index = first_index + indices[1]
        first_index = first_index + indices[0] * 3
        return first_index, second_index

    def get_middle_resnet_indices(indices):
        return indices[0]*2

    def get_upsamplers_index(indices):
        first_index = 2
        second_index = 2

        first_index = first_index + (indices[0]) * 3
        return first_index, second_index

    def get_output_resnet_indices(indices):
        first_index = 0
        second_index = 0

        first_index = first_index + indices[1]
        first_index = first_index + (indices[0] * 3)
        return first_index, second_index


    def get_downsamplers_index(indices):
        res = 3
        res = res + indices[0] * 3
        return res

    def get_input_attention_indices(indices):
        first_index = 4
        first_index = first_index + (indices[0] - 1) * 3
        first_index = first_index + indices[1]

        second_index = 1
        return first_index, second_index

    def get_output_attention_indices(indices):
        first_index = 0
        first_index = first_index + (indices[0]) * 3
        first_index = first_index + indices[1]

        second_index = 1
        return first_index, second_index

    def replace_resnet_labels(new_key):
        new_key = new_key.replace('norm1', 'in_layers.0')
        new_key = new_key.replace('conv1', 'in_layers.2')
        new_key = new_key.replace('time_emb_proj', 'emb_layers.1')
        new_key = new_key.replace('norm2', 'out_layers.0')
        new_key = new_key.replace('conv2', 'out_layers.3')
        new_key = new_key.replace('conv_shortcut', 'skip_connection')
        return new_key

    unique_keys = []
    for key, value in state_dict.items():
        # first, replace base_model with lora_unet
        new_key = key.replace('base_model', 'lora_unet')

        # remove model
        new_key = new_key.replace('model.', '')

        # replace down_blocks with input_blocks
        new_key = new_key.replace('down_blocks', 'input_blocks')

        # replace up_blocks with output_blocks
        new_key = new_key.replace('up_blocks', 'output_blocks')

        # replace mid_block with middle_block
        new_key = new_key.replace('mid_block', 'middle_block')

        # get the integer indices
        tokens = new_key.split('.')
        indices = []

        for t in tokens:
            if t.isdigit():
                indices.append(int(t))


        if 'add_embedding' in new_key:
            new_key = new_key.replace('add_embedding', 'label_emb')
            new_key = new_key.replace('linear_1', '0.0')
            new_key = new_key.replace('linear_2', '0.2')
        
        if 'conv_in.weight' in new_key:
            new_key = new_key.replace('conv_in', 'input_blocks.0.0')
        
        if 'time_embedding' in new_key:
            if 'linear_1' in new_key:
                new_key = new_key.replace('time_embedding.linear_1', 'time_embed.0')
            elif 'linear_2' in new_key:
                new_key = new_key.replace('time_embedding.linear_2', 'time_embed.2')
        
        if 'add_embedding' in new_key:
            if 'linear_1' in new_key:
                new_key = new_key.replace('add_embedding.linear_1', 'label_emb.0.0')
            elif 'linear_2' in new_key:
                new_key = new_key.replace('add_embedding.linear_2', 'label_emb.0.2')
        
        if 'input_blocks' in new_key:
            if 'downsamplers' in new_key:
                downsamplers_index = get_downsamplers_index(indices)
                new_key = new_key.replace(f'downsamplers.{indices[1]}.', '')
                new_key = new_key.replace(f'{indices[0]}', f'{downsamplers_index}.0')
                new_key = new_key.replace('conv', 'op')
            
            if 'attentions' in new_key:
                new_key = new_key.replace('attentions.', '')

                first_index, second_index = get_input_attention_indices(indices)
                new_key = new_key.replace(f'{indices[0]}.{indices[1]}', f'{first_index}.{second_index}')
            
            if 'resnets' in new_key:
                first_index, second_index = get_input_resnet_indices(indices)
                new_key = new_key.replace('resnets.', '')
                new_key = new_key.replace(f'{indices[0]}.{indices[1]}', f'{first_index}.{second_index}')
                new_key = replace_resnet_labels(new_key)


        if 'output_blocks' in new_key:
            if 'attentions' in new_key:
                new_key = new_key.replace('attentions.', '')

                first_index, second_index = get_output_attention_indices(indices)
                new_key = new_key.replace(f'{indices[0]}.{indices[1]}', f'{first_index}.{second_index}')
            
            elif 'upsamplers' in new_key:
                first_index, second_index = get_upsamplers_index(indices)
                new_key = new_key.replace(f'upsamplers.{indices[1]}.', '')
                new_key = new_key.replace(f'{indices[0]}', f'{first_index}.{second_index}')
            
            elif 'resnets' in new_key:
                first_index, second_index = get_output_resnet_indices(indices)
                new_key = new_key.replace('resnets.', '')
                new_key = new_key.replace(f'{indices[0]}.{indices[1]}', f'{first_index}.{second_index}')
                new_key = replace_resnet_labels(new_key)

        
        if 'middle_block' in new_key:
            if 'attentions' in new_key:
                new_key = new_key.replace(f'attentions.{indices[0]}.', '')
                new_key = new_key.replace('middle_block', 'middle_block.1')
            #new_key = new_key.replace(f'1.{indices[1]}', f'1')

            elif 'resnet' in new_key:
                index = get_middle_resnet_indices(indices)
                new_key = new_key.replace('resnets.', '')
                new_key = new_key.replace(f'{indices[0]}', f'{index}')
                new_key = replace_resnet_labels(new_key)


        
        # convert lora_A to lora_down
        new_key = new_key.replace('lora_A', 'lora_down')
        new_key = new_key.replace('lora_B', 'lora_up')

        if 'lora_down' in new_key:
            value = value.flatten(1)
        elif 'lora_up' in new_key:
            value = value.reshape((value.shape[0], -1))

        # finally, replace the dots with _ before lora
        pos = new_key.find('.lora')
        substring = new_key[:pos]
        substring = substring.replace('.', '_')
        new_key = substring + new_key[pos:]
        new_state_dict[new_key] = value

        # finally, add an alpha value if it doesn't exist
        if not substring in unique_keys:
            new_key = substring + '.alpha'
            new_state_dict[new_key] = torch.tensor(alpha)
            unique_keys.append(substring)

    save_file(new_state_dict, comfyui_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--peft_model_path', required=True, type=str)
    parser.add_argument('--comfyui_model_path', required=True, type=str)
    args = parser.parse_args()
    convert_sdxl_peft_to_comfyui(args.peft_model_path, args.comfyui_model_path)