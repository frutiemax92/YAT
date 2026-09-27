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

The instance images are encoded **once**: cycling over them is the training's job, and it replays
the same cached features. The regularization pool is filled from `dreambooth_reg_tar`, or from the
shards of `dreambooth_regularization_folder`, until `precompute_size` samples have been written; a
`dreambooth_reg_tar` holding fewer images than that is encoded once, whole. The training then takes
one batch of each pool in
turn, an instance batch and then a class batch, which is the order the live dreambooth sampler
produces. Each pool is reshuffled every time it runs out, so the small instance pool simply comes
around more often.

`dreambooth_lambda` is not part of what makes a cache valid: changing it re-uses the same features
instead of calculating them again. Changing the datasets, the prompts, the batch size or the aspect
ratio does invalidate it.

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

Setting `dreambooth_instance_tar` switches the sampler to the dreambooth one, which strictly
alternates an instance batch and a class batch. Every training step takes one of each, and
optimizes the loss of the dreambooth paper:

    loss = instance_loss + dreambooth_lambda * class_loss

Each batch gets its own backward pass, which accumulates exactly the gradient of that sum without
holding both graphs in memory at once, and the optimizer steps once. `gradient_accumulation_steps`
counts those pairs: with `2`, a step is two instance batches and two class batches. Besides
`train/loss`, the instance and the class losses are logged apart as `train/instance_loss` and
`train/class_loss`.

The instance images are cycled over for as long as the training runs, and the two streams are
bucketed apart, so a batch never mixes instance and class images.
Dreambooth cannot be combined with `dual_gpu`.

- `dreambooth_instance_tar` : the local TAR file holding the instance images. It replaces `dreambooth_dataset_folder`, which is now refused.
- `dreambooth_reg_tar` : a single local TAR file holding the regularization (class) images, cycled over like the instance images. When set, it is used instead of `dreambooth_regularization_folder`, even if `r2_bucket_name` is set.
- `dreambooth_regularization_folder` : without `dreambooth_reg_tar`, where the regularization images come from: the shards of that folder in the bucket when `r2_bucket_name` is set, otherwise a local folder. One of the two has to be given.
- `dreambooth_instance` : the instance prompt, used as the caption of the instance images that have no caption of their own.
- `dreambooth_class` : the class prompt, used the same way for the regularization images.
- `dreambooth_lambda` : the weight of the prior preservation loss, i.e. the loss of the class (regularization) batches, defaults to `1.0`. The loss of the instance batches is not weighted. This is the λ of the dreambooth paper.

### Lightweight controlnet

A reference image conditions the generation without a second copy of the transformer, using
[FourierFT](https://arxiv.org/abs/2405.03003) as the parameterization. Every controlled layer is given
a complex half-spectrum with `out_features // 2 + 1` bins, built out of zeros with two disjoint sets of
its bins written, and the inverse real FFT of it is added to that layer's output:

    z  = zeros(bins) complex, scattered bins set from spectrum * spectrum_scaling
                              and from reference * reference_scaling
    z' = irfft(z, n=out_features)
    y  = layer(x) + z'

`spectrum` is a few hundred learned complex coefficients, zero initialized as in FourierFT.
`reference` is a projection of a summary vector describing the reference image, which a shared
conditioner computes once per step from the frozen DINOv2 output and every layer reads through its own
small projection. Both sets are scattered over individual bins of one per-layer permutation - the way
FourierFT picks its frequencies, so each reaches every scale of the spectrum rather than occupying one
band - and they never overlap, since a step with no reference image is exactly the reference bins left
at zero and that must not disturb the learned ones. The mechanism lives in the
`lightweight_controlnet` submodule.

An inverse *real* FFT rather than the real part of a full one, which is what FourierFT itself takes.
`Real(ifft(z))` of a real `z` is exactly even-symmetric - `z'[c] == z'[N-c]` to the last bit - so that
map has rank `⌊N/2⌋+1` instead of `N`: entries written past that rank cost parameters and optimizer
state without reaching any offset the others could not, and each one ties channel `c` to channel `N-c`,
a pair with nothing to do with each other. `irfft` over a complex half-spectrum has `2·bins − 2 = N`
free reals, so it is full rank with nothing redundant. The parameter count is unchanged: a fraction
`f` of the bins is `2·f·bins = f·N` reals, exactly what a fraction `f` of the entries used to cost.

"Output dimension" is `out_features` for a Linear and `out_channels` for a Conv2d, so `z'` is one
number per output channel. Two things follow from that, and they are the whole character of this
design.

**The reference contribution is per sample for free.** Every image of the batch writes its own block
and gets its own `z'`, at the cost of one 1d FFT per layer. The weight-space form of FourierFT cannot
do that - a per-sample delta there is a per-sample `out x in` matrix and the layer becomes a batched
matmul over it.

**The offset is uniform over the frame.** One number per channel is added to every token of a sequence
and over the whole extent of a feature map, so the path has no per-position channel: it cannot draw a
mask or hold an edge. What it is instead is a global instruction, the same category as SDXL's own
`crop_coords_top_left` micro-conditioning, which is also a spatially uniform per-channel bias and
which does move the framing of an image. So coarse placement and size are reachable; the edge accuracy
of a real ControlNet is not.

Whether *where* reaches the model at all is `lw_controlnet_pool_grid`'s job. Measured on four 160px
blobs at the four corners of a 512px frame, through DINOv2-base and the conditioner: at `pool_grid 1`
the four summary vectors sit at cosine 0.996 of each other - a 6-9% relative change, less than a size
change produces - so position is all but absent. At `pool_grid 2`, the default, the corners separate to
cosine 0.95 and a 24-31% relative change, as strong as the size signal. Raise it further for
references whose entire content is layout, and note it costs one shared `cond_dim * g^2 -> cond_dim`
projection rather than anything per layer.

Nothing of the model is rewritten. The path is attached as a `register_forward_hook` per layer, so the
module tree, the parameters and the state dict are exactly what they were: the PEFT adapter of the run
saves as an ordinary PEFT adapter with nothing to unpatch first, the base weights stay frozen, and a
4bit base model stays 4bit. The layers may be PEFT LoRA layers - the hook then fires after the adapter,
so the offset lands on the layer's whole output - but they do not have to be.

Both scalings default to **1.0**, not to the 150 FourierFT uses. Under `lw_controlnet_ifft_norm:
ortho` on a 1-D transform there is nothing to compensate for: the `1/√N` the transform applies is
cancelled by the `~N/2` bins that contribute, so `std(z')` lands at roughly `std(parameter)` whatever
`N` is - which is also what lets one gain cover a 320-channel convolution and a 3072-wide linear at
once. FourierFT's 150 belongs to its own setting, a 2-D transform spreading ~1000 entries over
`out × in ≈ 1e7`, a fill ratio near `1e-3` that does attenuate hugely; ours is 0.5 to 0.75 and
attenuates nothing. A gain of 150 here is 150× too hot: at `lr 5e-4` one optimizer step already moves
a layer by 69% of its own output, and over a few hundred hooked layers that is noise rather than an
image. The reference
projection additionally carries a `1/sqrt(cond_dim)` correction that puts one optimizer step on either
block at the same size - without it the reference block reached the scale of the layer's own output
after a single step at `lr=1e-4` while the learned block was two orders of magnitude below it.

This replaced two earlier designs, both of which conditioned the LoRA rank tensor: one that made both
projections bilinear in the activation and the reference, and one that put a per-layer cross attention
from the rank tensor to a resampled memory of the reference, with 2d position embeddings on both sides.
Neither produced a model that followed its reference. This design gives up spatial placement rather
than the reference, and buys directness with it: the offset is added to the layer's output instead of
being routed through PEFT's zero initialized `lora_B`, so no part of it can be invisible at
initialization, and there is no sequence layout to declare or get wrong. The checkpoints of the three
are not interchangeable.

A step with no reference image leaves the reference block at zero, which is plain FourierFT: that is
the unconditional path, it trains on the dropped steps, and it is what `lw_controlnet_dropout_prob`
exercises. It is identical to `--control_strength 0` and not merely similar to it - the same entries,
the same zeros.

A dataset is a TAR built by `create_lw_controlnet_tar.py` from a folder holding a `src` and a
`ref` subfolder:

    dataset/src/<name>.jpg   the image the diffusion path denoises
    dataset/src/<name>.txt   its caption
    dataset/ref/<name>.jpg   the reference image

The three are paired by the name they share. An ordinary TAR of images and captions can be listed
next to those: its samples have no reference and train the LoRA alone, which is how a run mixes
reference conditioned data with a plain dataset. The two kinds are bucketed apart, so a batch is
never a mix of the two.

    python create_lw_controlnet_tar.py my_dataset my_dataset.tar
    accelerate launch train_krea2.py --config config_krea2_lw_controlnet.yaml

Every validation writes `models/<step>/`, holding an ordinary PEFT adapter next to
`fourier_control.pt` and `fourier_control_config.json` - the layout `infer_krea2_controlnet.py`
reads back with `--control_pretrained`. The adapter needs no special handling, since the model was
never patched: it is the LoRA as any loader reads it, and it is what the model computes with both
strengths at 0. `fourier_control.pt` holds the control path, which is not a LoRA and cannot be folded
into one. The per-layer bin placements are saved with it as buffers, so a checkpoint does not depend
on `lw_controlnet_random_loc_seed` staying the same in a later run.

A directory holding `reference_control.pt` is a checkpoint of the cross attention architecture this
replaced; loading one is refused by name rather than half succeeding.

**The LoRA is optional.** The control path never touches an adapter - it hooks the layer, whatever that
layer turned out to be - and the trainer freezes the base model itself when there is no LoRA to do it,
so leaving `lora_rank` out gives a control only run: the control path is the only thing the optimizer
is handed, the reference is the only thing in the model that can account for anything, and a checkpoint
is a few megabytes instead of a copy of the model. That is the run to start with when the question is
whether the reference works at all. `infer_krea2_controlnet.py` reads such a checkpoint as it is; its
`--lora_rank` defaults to `0`, meaning no adapter.

With a `lora_rank` set, both halves train: PEFT froze the base weights and left the adapter trainable,
and which of the two learns on a given batch is `lw_controlnet_freeze_path`'s call. Rank has nothing to
do with the reference's capacity any more - under the architecture this replaced everything the
reference contributed had to pass through those `r` numbers, and now it is added to the layer's output
beside the LoRA. Raise rank for the subject; raise `lw_controlnet_cond_dim` and
`lw_controlnet_reference_ratio` for the control.

Either way it cannot be combined with `dual_gpu`, which raises rather than silently ignoring the
reference images, since that iterator builds its batches inline and never sees the reference.

It does combine with dreambooth. `dreambooth_instance_tar` is then a TAR in the `src`/`ref`
layout, so the reference path is trained on the instance images, and `dreambooth_reg_tar` can be
either kind: a plain TAR of images and captions preserves the prior of the LoRA alone, a `src`/`ref`
one also regularizes the reference path. Each stream pairs its own halves, so the two TARs can
number their samples the same way. An instance TAR built with `--allow_missing_captions` holds no
captions at all, and `dreambooth_instance`/`dreambooth_class` supply them as usual.

- `train_lw_controlnet` : a boolean, turns the control path on.
- `lw_controlnet_tars` : the local TAR files to read, cycled over. Without them the usual shard/R2 plumbing is used.
- `lw_controlnet_pretrained` : a `models/<step>` directory to resume the control path from.
- `lw_controlnet_dropout_prob` : probability of dropping the reference on a step that has one, defaults to `0.1`, so the plain FourierFT path keeps working and the reference can be guided against at inference.
- `lw_controlnet_freeze_path` : a boolean, off by default. On a batch that carries a reference, freezes everything but the control path, so the LoRA cannot absorb what the reference was supposed to account for. Unlike under the architecture this replaced it is safe from step 0: the control path is added to the layer's output directly rather than routed through PEFT's zero initialized `lora_B`, so its gradient never waits on another weight.
- `lw_controlnet_validation_references` : reference images paired with `validation_prompts`, cycled over when there are fewer. Without them the validation images show what the LoRA alone does.
- `lw_controlnet_reference_model` : the frozen encoder, defaults to `facebook/dinov2-base`. Its hidden size is deduced from the name, or given with `lw_controlnet_reference_dim`.
- `lw_controlnet_target_modules` : which layers get a control path, matched on the module name the way PEFT matches its own. Left out, **every** Linear and Conv2d of the model is hooked, which is what FourierFT itself does. They do not have to be the LoRA layers.
- `lw_controlnet_cond_dim` : the width of the reference summary vector, i.e. what the whole model sees of the reference, defaults to `256`. Every controlled layer reads it through its own `cond_dim x n_reference` projection, which is where the per-layer parameter count goes, so this is the size knob of the whole thing.
- `lw_controlnet_pool_grid` : the reference patch field is average pooled onto this many cells per side before being projected into that vector, `2` by default; `1` is the plain global mean. This is what carries *where* anything is in the reference - at `1` four blobs at the four corners of a frame come out at cosine 0.996 of each other, at `2` at cosine 0.95 - so raise it when the reference's content is mostly layout. It costs one shared projection, not a per-layer one.
- `lw_controlnet_frequency_ratio`, `lw_controlnet_reference_ratio` : how many *bins* of each layer's half-spectrum the learned coefficients and the reference fill, as a fraction of that layer's bin count, `0.5` and `0.25`. Each bin is one complex coefficient, so a fraction `f` of the bins costs `2·f·bins = f·out_features` parameters - the same size these ratios always meant. The two sets are disjoint, so they cannot add up to more than `1`; at exactly `1` the whole half-spectrum is written, which is `N` free reals and the most this parameterization reaches. `lw_controlnet_n_frequency` and `lw_controlnet_n_reference` override them with an absolute bin count, clamped per layer so a narrow layer stays in range.
- `lw_controlnet_spectrum_scaling`, `lw_controlnet_reference_scaling` : the gain each side is written into the spectrum at, both `1.0`, and the two values validation images are generated at. Not FourierFT's 150 - see above; at 150 the first validation image of a 454-layer run comes out as pure noise. Plain floats rather than parameters, so they never reach a checkpoint: `infer_krea2_controlnet.py --spectrum_strength` and `--control_strength` are the same two dials at inference, and `--control_strength 0` is exactly the model with no reference image.
- `lw_controlnet_ifft_norm` : normalization of the inverse FFT, `ortho` (1/sqrt(n)) by default, or `backward` (1/n, torch's own) or `forward` (none). `ortho` is what lets one scaling cover every width in the model.
- `lw_controlnet_random_loc_seed` : the seed the per-layer bin placement is drawn from, `777`. Both sets are scattered over individual bins of one permutation, the way FourierFT picks its frequencies, so each reaches every scale of the spectrum rather than occupying one band of neighbouring frequencies. A checkpoint saves its own placements, so changing this does not invalidate one.
- `lw_controlnet_dtype` : dtype of the control parameters, `float32`. The inverse FFT itself always runs in float32 - there is no half precision kernel for it - and these are a few thousand numbers per layer.

`precompute_features` works here too: the reference features are written to the cache next to the
latents and the embeddings, and DINOv2 is dropped for the rest of the training.

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