# YAT : Yet Another Trainer for diffusion models



This is Yet Another Trainer for diffusion models, currently supports those models:
1. SANA
2. Pixart-Sigma
3. Stable Diffusion 3.5

It supports loading TAR folders in the WebDataset format (https://github.com/webdataset/webdataset) either from a secure Cloudflare R2 Bucket, public urls or from local folders. For public urls and local TAR folder, use the `urls` parameter. For the secure urls, see the Cloudflare R2 parameters in the next section.

The trainer allows for multi-gpu training with `Accelerate` and dynamic aspect ratio bucketing; you don't need to train with square images only!

For generating WebDatasets, it is highly suggested to use Img2Dataset (https://github.com/rom1504/img2dataset) for downloading the images and generating at the same time the TAR folders.

The code also allows for Lycoris Finetuning (not Dreambooth) as it can give faster convergence due to a lower parameter count/different reparametrization. Tested to work correctly with Lora, Locon and Loha, the other algorithms give an error as of writing.

This is currently tested under Ubuntu 24.04.1 LTS.

## How to train

First, create a Python virtual environnement.

    python3 -m venv .venv
    source .venv/bin/activate

Then, install the required packages.

    pip install --upgrade -r requirements.txt

After that, configure your accelerate configuration.

    accelerate config

It will prompt you with an interactive menu, the most important options are your number of graphic cards and the mixed precision setting.
After that, prepare a config.yaml file, you can check the example in tests. 
Finally execute the script.

    accelerate launch train_sana.py --config config.yaml

You can also run tensorboard in the current directory for validation images and logs for losses.

    tensorboard --logdir . --bind_all

The model will be saved for each validation, which is defined by the `num_steps_per_validation` parameter.

## List of parameters in the config file

Here is the current list of parameters in the config file.

- `r2_endpoint` : the endpoint for Cloudflare r2
- `r2_access_key` : the access key for Cloudflare r2
- `r2_secret_key` : the secret key for Cloudflare r2
- `r2_bucket_name` : the bucket name containing your TAR files in Cloudflare r2
- `r2_tar_files` : a list of TAR files that will form your dataset. There will be randomly sampled with equal weights.
- `batch_size` : the batch size for the training. The effective batch size will be `num_gpus * batch_size`
- `learning_rate` : the learning rate for the training.
- `steps` : the total number of steps for the training. Since the dataset is an iterative one, deducing automatically the number of images in the dataset is expensive, therefore it's better to manually set it. The number of `epochs` will be `steps / (number of images in your dataset)`
- `bfloat16` : do the training entirely in bfloat16. This is highly suggested for saving VRAM; it takes a boolean value (true of false).
- `gradient_accumulation_steps` : the gradient accumulation steps, which will increase the effective batch size at the cost of slower training.
- `validation_prompts` : a list of validation prompts for your validation.
- `pretrained_pipe_path` : a path to the diffusers pipeline, either hosted locally or on HuggingFace
- `urls` : contains the public urls that point to the TAR files in the form of WebDataset
- `pretrained_model_path` : a path to the model that will get trained that is part of the pipeline. This is used when you want to start from a finetuned model and use the default pipeline.
- `lora_rank` : the rank of the lora (see https://arxiv.org/abs/2106.09685)
- `lora_alpha` : the alpha parameter for lora training. A correct value is `lora_alpha=lora_rank`.
- `lora_dropout` : dropout probability for lora training
- `lora_algo` : the algorithm to use for lora training
  - `lora`
  - `locon`
  - `loha`
  - `lokr`
  - `dylora`
  - `glora`
  - `full`
  - `diag-oft`
  - `boft`
- `lora_target_modules` : the names of the targeted modules for the reparametrization. For SANA, a good value is `ff`.
- `low_vram` : use this when low on VRAM. For SANA, it is possible with this option to train with a `batch size=4`, `lora_rank=8`, `lora_algo=locon` under 12 GB VRAM (tested on dual RTX4070s)
- `use_preservation` : the original model under training is cloned in a frozen copy. The training loss is then `loss_tot=loss_noise + preservation_ratio*loss_reconstruction`. Use this if you want to preserve some of the original model behaviour.
- `preservation_ratio` : the ratio as explained just above

## About this repository

This is a personal project for experimenting with training diffusion models as I like to know what is going on under the hood and apply some modifications to my personal preferences. I do not garantee the best results out of this project, use it at your own risk!