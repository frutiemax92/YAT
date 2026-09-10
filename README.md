# YAT : Yet Another Trainer for diffusion models



This is Yet Another Trainer for diffusion models, currently supports those models:
1. SANA
2. Pixart-Sigma
3. Stable Diffusion 3.5

It supports loading TAR folders in the WebDataset format (https://github.com/webdataset/webdataset) either from a secure Cloudflare R2 Bucket, public urls or from local folders. For public urls and local TAR folder, use the `urls` parameter. For the secure urls, see the Cloudflare R2 parameters in the next section.

The trainer allows for multi-gpu training with `Accelerate` and dynamic aspect ratio bucketing; you don't need to train with square images only!

For generating WebDatasets, it is highly suggested to use Img2Dataset (https://github.com/rom1504/img2dataset) for downloading the images and generating at the same time the TAR folders.

The code also allows for LoRA Finetuning (not Dreambooth) as it can give faster convergence due to a lower parameter count/different reparametrization. Lokr and Loha algorithms are also supported. There is currently a bug in PEFT with the 'conv_depth' module, so do not add that to the target modules.

The trainer creates a `cache` folder, which contains latents and embeddings. This is done to speed up the training process and reducing the VRAM requirements. By default, the number of cache elements is 1000, but you can increase it by specifying `cache_size` in the config file.

This is currently tested under Ubuntu 24.04.1 LTS.

## How to train

First, create a Python virtual environnement.

    python3 -m venv .venv
    source .venv/bin/activate

Then, install the required packages.

    pip install --upgrade -r requirements.txt

The current diffusers library version has a bug with the SanaPipeline, you need to install it from its github repository or you won't get good validation images.

    pip install --upgrade git+https://github.com/huggingface/diffusers.git

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

Here is the current list of parameters in the config file. Every option is optional unless stated
otherwise; a boolean option is enabled by simply being present in the file (`use_ema: True`), and
removing the line is what disables it.

### Dataset

- `r2_endpoint` : the endpoint for Cloudflare r2
- `r2_access_key` : the access key for Cloudflare r2
- `r2_secret_key` : the secret key for Cloudflare r2
- `r2_bucket_name` : the bucket name containing your TAR files in Cloudflare r2
- `r2_dataset_folder` : the folder inside the bucket that holds the dataset shards. The trainer builds the shard names itself as `shard-000000.tar`, `shard-000001.tar`, and so on.
- `num_shards` : how many shards that folder contains. The range is split evenly between the GPUs, so each process downloads from its own slice of the dataset.
- `local_shard_paths` : a list of TAR files already downloaded on the machine. When it is set, the shards are picked from that list instead of being downloaded from r2, which is the most stable option.
- `dataset_seed` : the seed used to pick the shards and to shuffle the samples. It also decides the order the precalculated features are replayed in.
- `r2_tar_files` : a list of TAR files that will form your dataset. There will be randomly sampled with equal weights. It comes from the older dataset layout and is superseded by `r2_dataset_folder` and `num_shards`, which is what the trainer reads now.
- `urls` : contains the public urls that point to the TAR files in the form of WebDataset. Same thing, it is read from the config but no longer used.

### Training

- `batch_size` : the batch size for the training. The effective batch size will be `num_gpus * batch_size`
- `learning_rate` : the learning rate for the training.
- `steps` : the total number of steps for the training. Since the dataset is an iterative one, deducing automatically the number of images in the dataset is expensive, therefore it's better to manually set it. The number of `epochs` will be `steps / (number of images in your dataset)`
- `num_steps_per_validation` : the number of steps between two validations. The model is also saved under `models/<step>` at every validation.
- `warmup_steps` : the learning rate ramps up linearly from zero over that many steps, then stays at `learning_rate`.
- `weight_decay` : the weight decay of the optimizer, defaults to `0.0`.
- `gradient_accumulation_steps` : the gradient accumulation steps, which will increase the effective batch size at the cost of slower training.
- `bfloat16` : do the training entirely in bfloat16. This is highly suggested for saving VRAM; it takes a boolean value (true of false).
- `use_adamw_8bit` : use the 8 bit Lion optimizer from bitsandbytes instead of AdamW, which saves VRAM on the optimizer states.
- `use_ema` : keep an exponential moving average of the weights (decay `0.999`) and validate and save from it instead of the raw weights.
- `train_unconditional_prob` : the probability that a batch is trained with the empty prompt embedding instead of its own captions, which trains the unconditional branch used by classifier-free guidance.
- `timesteps` : a list of timestep indices to restrict the training to, instead of sampling them from the usual distribution. This is how a refiner model is trained: the lora is also disabled at validation for the timesteps that are not in the list.
- `aspect_ratio` : overrides the aspect ratio bucket table. `512` and `1024` select the standard tables, any other value scales the 1024 table by `aspect_ratio / 1024`.
- `exploration_steps` : from the paper *explorative modeling: unlocking a third pretraining axis and end-to-end generation*. Each step draws that many noise samples, and the one with the lowest loss is the one that gets trained on.
- `dual_gpu` : dedicate the second GPU to the feature extraction and send the features to the first GPU, which only trains. It cannot be combined with `precompute_features`.
- `low_vram` : use this when low on VRAM. For SANA, it is possible with this option to train with a `batch size=4`, `lora_rank=8`, `lora_algo=lora` under 12 GB VRAM (tested on dual RTX4070s). Only the SD3.5 trainer acts on it at the moment.
- `validation_prompts` : a list of validation prompts for your validation.

### Model

- `pretrained_pipe_path` : a path to the diffusers pipeline, either hosted locally or on HuggingFace
- `pretrained_model_path` : a path to the model that will get trained that is part of the pipeline. This is used when you want to start from a finetuned model and use the default pipeline.
- `pretrained_pipe_single_file` : a path or url to a single safetensors checkpoint in the original (CompVis) format, which is how the finetunes from civitai are distributed.

### Features

The trainer can either run the VAE and the text encoder on every batch, precalculate them once at
startup, or read features that were extracted in a separate pass.

- `compute_features` : the dataset shards contain images and captions, so the VAE and the text encoder run on every batch to produce the latents and the embeddings. Without it, the shards are expected to already contain `latent.pt` and `emb.pt` entries.
- `vae_max_batch_size` : how many images the VAE encodes at once. Extracting features takes more VRAM than the training itself, so this is usually smaller than `batch_size`.
- `text_encoder_max_batch_size` : the same thing for the text encoder.
- `precompute_features` : run the VAE and the text encoder once at startup over a pool of samples, write the latents and the embeddings to disk, then free both models so they don't take any VRAM during the training steps. The training then loops over that pool of precalculated features.
- `precompute_size` : the number of samples to precalculate. It defaults to `cache_size`. Note that the training only ever sees those samples, so this is a trade-off between VRAM/speed and dataset diversity.
- `precompute_cache_dir` : where the precalculated features are written, defaults to `precomputed_features`. A cache that matches the current `batch_size` and is big enough is reused on the next run instead of being recalculated.
- `precompute_force` : recalculate the features even when a usable cache is already on disk.
- `extract_features` : instead of training, encode the whole dataset and upload the resulting latents and embeddings as WebDataset shards to r2. A later run can then train on them without `compute_features`.
- `r2_upload_key` : the folder in the bucket the extracted shards are uploaded to.
- `r2_upload_shard_size` : the number of samples per uploaded shard.
- `cache_size` : the default value for `precompute_size`.

The validation prompts are still encoded by the text encoder, so it can only be freed once its
embeddings have been memoized. A model that implements `encode_validation_prompts` (SANA and Krea 2
do) encodes them at the end of the precompute pass, and its text encoder is then freed before the
lora is even built, which is what makes a big text encoder fit next to the model on a small card.
Any other model keeps its text encoder until the first validation has run. The VAE is kept on the
CPU either way, and moved back to the GPU only while the validation images are decoded.

Those embeddings are written next to the features, so on a later run with the same
`validation_prompts` the text encoder is not even loaded: SANA and Krea 2 build their pipeline
without it. Changing `validation_prompts` means it has to be loaded again to encode the new ones.
The VRAM in use is printed at the end of the pass, before and after the VAE and the text encoder are
let go, which is the quickest way to see what is actually taking the memory.

`precompute_features` also works for a dreambooth training, where the cache is split in two pools:

    precomputed_features/instance/rank0, rank1, ...
    precomputed_features/regularization/rank0, rank1, ...

The instance images are encoded **once**, whatever `dreambooth_num_repeats` says: repeating them is
the training's job, and it replays the same cached features. The regularization pool is filled from
the shards of `r2_dataset_folder` until `precompute_size` samples have been written. An epoch is then
the instance pool repeated `dreambooth_num_repeats` times, shuffled at every repeat, followed by the
regularization pool, which is the order the live dreambooth sampler produces.

Because the repeats happen at replay, `dreambooth_num_repeats` and
`dreambooth_num_regularisation_passes` are not part of what makes a cache valid: changing them
re-uses the same features instead of calculating them again. Changing the datasets, the prompts, the
batch size or the aspect ratio does invalidate it.

### LoRA

- `lora_rank` : the rank of the lora (see https://arxiv.org/abs/2106.09685)
- `lora_alpha` : the alpha parameter for lora training. A correct value is `lora_alpha=lora_rank`.
- `lora_dropout` : dropout probability for lora training
- `lora_algo` : the algorithm to use for lora training
  - `lora`
  - `loha`
  - `lokr`
  - `fourierft`
- `lora_target_modules` : the names of the targeted modules for the reparametrization. For SANA, a good value is `conv_inverted conv_point to_q to_k to_v to_out.0 linear_1 linear_2 proj`.
- `lora_pretrained` : if you want to resume training from a lora model, specify it there
- `lora_use_dora` : use DoRA (weight decomposed low rank adaptation) instead of plain lora.
- `lora_base_model_8bit` : load the frozen base model in 8 bit with bitsandbytes.
- `lora_base_model_4bit` : load the frozen base model in 4 bit (nf4, QLoRA style). For the models with a big text encoder, such as Krea 2, it also quantizes the text encoder.
- `fourierft_alpha` : the alpha of the `fourierft` algorithm, defaults to `0.01`.
- `lora_adapter_bf16` : keep the adapters in bfloat16 instead of the float32 peft casts them to. It roughly halves what DoRA has to materialize on every forward, at the cost of some numerical stability, so watch the loss when turning it on. On a 12 GB card it is what makes `lora_use_dora` fit with a wide `lora_target_modules` list.

### Dreambooth

Setting `dreambooth_dataset_folder` switches the sampler to the dreambooth one, which alternates
between the instance images and the regularization images.

- `dreambooth_dataset_folder` : the folder or TAR file holding the instance images.
- `dreambooth_regularization_folder` : the folder holding the regularization images. When `r2_bucket_name` is set, the regularization images are taken from the dataset shards instead.
- `dreambooth_instance` : the instance prompt, used as the caption of the instance images that have no caption of their own.
- `dreambooth_class` : the class prompt, used the same way for the regularization images.
- `dreambooth_num_repeats` : how many times the instance images are repeated between two regularization passes. With `precompute_features` the repeats cost nothing: the features are calculated once and replayed.
- `dreambooth_num_regularisation_passes` : how many regularization shards are consumed in between. With `1`, the instance images and one regularization shard strictly alternate.
- `dreambooth_lambda` : the weight of the regularization loss. It is read and passed to the sampler, but it is not applied to the loss at the moment.

### REPA

- `use_repa` : extract DINOv2 features next to the latents, for representation alignment (see https://arxiv.org/abs/2410.06940).
- `repa_lambda` : the weight of the alignment loss, defaults to `0.05`. The loss itself is currently commented out in the trainer, so only the feature extraction is active.
- `repa_pretrained_model` : read from the config but not used yet.

### Read but currently unused

These keys are still parsed, so an old config file keeps working, but nothing reads them anymore.

- `use_preservation` : the original model under training is cloned in a frozen copy. The training loss is then `loss_tot=loss_noise + preservation_ratio*loss_reconstruction`. Use this if you want to preserve some of the original model behaviour.
- `preservation_ratio` : the ratio as explained just above
- `url_probs` : the sampling weights that went with `urls`.
- `huggingface_dataset_repo` : a dataset repository on HuggingFace.
- `use_calculated_features`
- `lora_bias`, `lora_use_rslora`
- `cyclic_lr_max_lr`, `cyclic_lr_step_size_up`, `cyclic_lr_step_size_down`, `cylic_lr_mode` : the cyclic learning rate scheduler, replaced by `warmup_steps`.
- `save_to_disk`, `bucket_repeat` : only used by the older caching sampler in `common/cache.py`.

## About this repository

This is a personal project for experimenting with training diffusion models as I like to know what is going on under the hood and apply some modifications to my personal preferences. I do not garantee the best results out of this project, use it at your own risk!