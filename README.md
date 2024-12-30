# YAT : Yet Another Trainer for diffusion models



This is Yet Another Trainer for diffusion models, currently supports those models:
1. SANA
2. Pixart-Sigma

It supports loading TAR folders in the WebDataset format (https://github.com/webdataset/webdataset) either from a secure Cloudflare R2 Bucket, public urls or from local folders. For public urls and local TAR folder, use the `urls` parameter. For the secure urls, see the Cloudflare R2 parameters in the next section.

The trainer allows for multi-gpu training with `Accelerate` and dynamic aspect ratio bucketing; you don't need to train with square images only!

For generating WebDatasets, it is highly suggested to use Img2Dataset (https://github.com/rom1504/img2dataset) for downloading the images and generating at the same time the TAR folders.

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

## About this repository

This is a personal project for experimenting with training diffusion models as I like to know what is going on under the hood and apply some modifications to my personal preferences. I do not garantee the best results out of this project, use it at your own risk!
