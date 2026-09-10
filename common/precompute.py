import gc
import json
import os
import random
import shutil

import torch
from tqdm import tqdm

from common.bucket_sampler import Batch


def _map_tensors(obj, fn):
    """Recursively apply fn to every tensor found in obj, preserving the container types."""
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_map_tensors(o, fn) for o in obj)
    if isinstance(obj, list):
        return [_map_tensors(o, fn) for o in obj]
    if isinstance(obj, dict):
        return {k: _map_tensors(v, fn) for k, v in obj.items()}
    return obj


def to_cpu(obj):
    return _map_tensors(obj, lambda t: t.detach().to('cpu'))


def to_device(obj, device):
    return _map_tensors(obj, lambda t: t.to(device))


def contains_tensor(obj):
    if torch.is_tensor(obj):
        return True
    if isinstance(obj, (list, tuple)):
        return any(contains_tensor(o) for o in obj)
    if isinstance(obj, dict):
        return any(contains_tensor(o) for o in obj.values())
    return False


def is_prompt(value):
    if isinstance(value, str):
        return True
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return all(isinstance(v, str) for v in value)
    return False


def release_module(module):
    """Drop the weights of a module, whoever still holds a reference to it.

    Setting the attribute to None is not enough for the big text encoders: the pipeline and the
    quantizer plumbing keep the module alive, so the vram stays taken. Emptying every parameter and
    buffer releases the storage no matter who is still pointing at the module.
    """
    empty = torch.empty(0)
    for param in module.parameters(recurse=True):
        param.grad = None
        # a bitsandbytes parameter also carries the quantization state on the gpu
        quant_state = getattr(param, 'quant_state', None)
        if quant_state is not None:
            for name in ('absmax', 'code', 'offset'):
                if torch.is_tensor(getattr(quant_state, name, None)):
                    setattr(quant_state, name, empty)
        param.data = empty
    for module_name, buffer in list(module.named_buffers(recurse=True)):
        parent = module
        path = module_name.split('.')
        for part in path[:-1]:
            parent = getattr(parent, part)
        if torch.is_tensor(buffer):
            setattr(parent, path[-1], empty)


class FreedTextEncoder(torch.nn.Module):
    """Stands in for the text encoder the precompute pass freed.

    The validation code moves the text encoder around and the pipelines read its dtype, so those
    keep working, while an actual encoding attempt raises instead of returning something wrong.
    """

    def __init__(self, dtype):
        super().__init__()
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def forward(self, *args, **kwargs):
        raise RuntimeError('the text encoder was freed by the features precompute pass')


class PromptEmbeddingCache:
    """Memoizes text embeddings by prompt so validation keeps working once the text encoder is gone.

    The wrapper stays transparent while the text encoder is alive: every call goes through and its
    result is remembered. Once the encoder has been freed, a hit replays the stored tensors and a
    miss raises instead of silently producing garbage.
    """

    def __init__(self, device='cpu'):
        self.entries = {}
        self.frozen = False
        self.device = device

        # off during the precompute pass, otherwise every training caption would be kept in memory
        self.recording = False

    @staticmethod
    def get_prompt(args, kwargs):
        if 'prompt' in kwargs:
            return kwargs['prompt']
        if len(args) > 0:
            return args[0]
        return None

    # these never change the embedding values, and they come in as a string in one call site and
    # as an object in the next, which would split the entry in two
    ignored_kwargs = ('device', 'generator')

    @classmethod
    def make_key(cls, name, args, kwargs):
        # only the arguments that can change the embedding values are part of the key, so the
        # device/generator/dtype objects that differ between calls don't split cache entries
        def keep(key, value):
            if key in cls.ignored_kwargs:
                return False
            return isinstance(value, (str, int, float, bool, type(None), list, tuple))

        key_args = tuple(repr(a) for a in args if keep(None, a))
        key_kwargs = tuple(sorted((k, repr(v)) for k, v in kwargs.items() if keep(k, v)))
        return (name, key_args, key_kwargs)

    def is_cacheable(self, args, kwargs):
        # a pipeline hands the embeddings it was given straight back to encode_prompt, with the
        # prompt set to None. Those calls don't depend on any prompt, so they must never share a
        # cache entry: the tensors that tell them apart cannot be part of the key.
        if not is_prompt(self.get_prompt(args, kwargs)):
            return False
        return not (contains_tensor(args) or contains_tensor(kwargs))

    def wrap(self, name, fn):
        def wrapper(*args, **kwargs):
            if not self.is_cacheable(args, kwargs):
                return fn(*args, **kwargs)

            key = self.make_key(name, args, kwargs)
            if key in self.entries:
                return to_device(self.entries[key], self.device)
            if self.frozen:
                raise RuntimeError(
                    f'the text encoder was freed by the features precompute pass, but {name} was '
                    f'called with an uncached prompt: {key}. Either add the prompt to the config '
                    f'before training, or disable precompute_features.')
            result = fn(*args, **kwargs)
            if self.recording:
                self.entries[key] = to_cpu(result)
            return result
        return wrapper

    def freeze(self):
        self.frozen = True


class PrecomputedBucketSampler:
    """Yields batches of latents/embeddings read back from the precompute cache.

    Batches keep the index they were written with, and every process shuffles those indices with
    the same seed. Since the writer produced batch i with the same aspect ratio on every process,
    the ratios stay in lockstep across processes without any collective communication.
    """

    def __init__(self, cache_dir, accelerator, seed, use_repa=False, model=None,
                 pools=None, num_repeats=1):
        self.cache_dir = cache_dir
        self.accelerator = accelerator
        self.seed = seed
        self.use_repa = use_repa
        self.model = model
        self.num_repeats = max(1, num_repeats)

        # a plain run has a single pool, a dreambooth one has the instance images and the
        # regularization images kept apart so each is only calculated once
        self.pools = pools if pools != None else [None]
        self.directories = {}
        self.pool_batches = {}
        for pool in self.pools:
            directory = rank_dir(cache_dir, accelerator.process_index, pool)
            manifest = read_manifest(directory)
            if manifest is None:
                raise RuntimeError(f'no precomputed features found in {directory}')
            self.directories[pool] = directory
            self.pool_batches[pool] = manifest['batches']

        self.rank_dir = self.directories[self.pools[0]]
        self.batches = self.pool_batches[self.pools[0]]

    def __len__(self):
        return sum(len(batches) for batches in self.pool_batches.values())

    def load_batch(self, batch_index, pool=None):
        if pool is None:
            pool = self.pools[0]
        directory = self.directories[pool]
        entry = self.pool_batches[pool][batch_index]
        latents = []
        embeddings = []
        repa_features = []
        for filename in entry['files']:
            sample = torch.load(os.path.join(directory, filename), map_location='cpu', weights_only=False)
            latents.append(sample['latent'])
            if entry['emb_per_sample']:
                embeddings.append(sample['embedding'])
            if sample.get('repa') is not None:
                repa_features.append(sample['repa'])

        if not entry['emb_per_sample']:
            embeddings = torch.load(os.path.join(directory, entry['emb_file']),
                                    map_location='cpu', weights_only=False)

        batch = Batch()
        batch.ratio = entry['ratio']
        batch.vae_features = torch.stack(latents)
        batch.embeddings = embeddings
        if self.use_repa and len(repa_features) > 0:
            batch.repa_features = torch.stack(repa_features)
            # same convention as BucketSampler: DINO on 224x224 always gives a 16x16 token grid
            batch.repa_spatial_dims = (16, 16)
            aspect_ratio = self.model.aspect_ratios[str(batch.ratio)]
            patch_size = getattr(self.model, 'patch_size', 16)
            batch.proj_spatial_dims = (int(aspect_ratio[0]) // patch_size, int(aspect_ratio[1]) // patch_size)
        return batch

    def epoch_order(self, epoch):
        """The (pool, index) pairs one epoch is made of.

        A dreambooth epoch is the instance images repeated dreambooth_num_repeats times, then the
        regularization pool, which is the order the dreambooth sampler produces live. The repeats
        cost nothing here: the same cached features are replayed.
        """
        # the same seed on every process keeps the ratio of batch i identical across processes
        rng = random.Random(self.seed + epoch)
        order = []
        for pool in self.pools:
            indices = list(range(len(self.pool_batches[pool])))
            repeats = self.num_repeats if pool == INSTANCE_POOL else 1
            for _ in range(repeats):
                rng.shuffle(indices)
                order.extend((pool, index) for index in indices)
        return order

    def __iter__(self):
        epoch = 0
        while True:
            for pool, batch_index in self.epoch_order(epoch):
                yield self.load_batch(batch_index, pool)
            epoch = epoch + 1


PROMPT_EMBEDDINGS_FILE = 'prompt_embeddings.pt'


def prompt_embeddings_path(cache_dir, process_index):
    return os.path.join(rank_dir(cache_dir, process_index), PROMPT_EMBEDDINGS_FILE)


def precomputed_prompts_available(params, process_index):
    """True when a previous run left everything the text encoder would be asked for on disk.

    A model can check this before building its pipeline and skip loading the text encoder
    altogether, which is the difference between fitting on a small card and not.
    """
    if not getattr(params, 'precompute_features', False):
        return False
    if getattr(params, 'precompute_force', False):
        return False

    wanted = max(1, params.precompute_size // params.batch_size)
    for pool in pools_for(params):
        manifest = read_manifest(rank_dir(params.precompute_cache_dir, process_index, pool))
        if manifest is None:
            return False
        # the instance pool holds a whole small dataset, not precompute_size samples
        if len(manifest.get('batches', [])) < (1 if pool == INSTANCE_POOL else wanted):
            return False

    path = prompt_embeddings_path(params.precompute_cache_dir, process_index)
    if not os.path.exists(path):
        return False
    try:
        stored = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return False
    # the prompts have to be the ones the validation will ask for
    return list(stored.get('validation_prompts') or []) == list(params.validation_prompts or [])


INSTANCE_POOL = 'instance'
REGULARIZATION_POOL = 'regularization'


def rank_dir(cache_dir, process_index, pool=None):
    """Where one process keeps its features. A dreambooth run keeps two pools side by side."""
    if pool:
        return os.path.join(cache_dir, pool, f'rank{process_index}')
    return os.path.join(cache_dir, f'rank{process_index}')


def pools_for(params):
    """The pools a run is made of: one, or the instance/regularization pair for dreambooth."""
    if getattr(params, 'dreambooth_dataset_folder', None) != None:
        return [INSTANCE_POOL, REGULARIZATION_POOL]
    return [None]


def read_manifest(directory):
    manifest_path = os.path.join(directory, 'manifest.json')
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path) as f:
            return json.load(f)
    except Exception:
        return None


class FeaturesPrecompute:
    """Runs the VAE/text encoder once at startup, then keeps them out of the training steps.

    The pass consumes the regular feature extracting sampler until enough batches are on disk, so
    it reuses the existing shard download, bucketing and extraction code. Afterwards the VAE is
    parked on the cpu (it is still needed to decode validation images) and the text encoder is
    dropped after the first validation, once the validation prompts have been memoized.
    """

    def __init__(self, model):
        self.model = model
        self.params = model.params
        self.accelerator = model.accelerator
        self.cache_dir = self.params.precompute_cache_dir
        self.pools = pools_for(self.params)
        self.rank_dir = rank_dir(self.cache_dir, self.accelerator.process_index)
        self.pool_dirs = {pool: rank_dir(self.cache_dir, self.accelerator.process_index, pool)
                          for pool in self.pools}
        self.batch_size = self.params.batch_size
        self.num_batches = max(1, self.params.precompute_size // self.batch_size)
        self.prompt_cache = PromptEmbeddingCache(self.accelerator.device)
        self.text_encoder_freed = False

    def cache_signature(self):
        """Everything that changes what ends up in the pool, so a stale cache is not reused."""
        params = self.params
        signature = {
            'batch_size': self.batch_size,
            'use_repa': bool(params.use_repa),
            'dataset_seed': params.dataset_seed,
            # this one decides the resolution of the cached latents
            'aspect_ratios': getattr(params, 'aspect_ratios', None),
        }
        if getattr(params, 'dreambooth_dataset_folder', None) != None:
            # dreambooth_num_repeats and dreambooth_num_regularisation_passes are deliberately
            # not part of this: they decide how the cached features are replayed, not what gets
            # calculated, so changing them must not throw the cache away
            signature['dreambooth'] = {
                'dataset_folder': params.dreambooth_dataset_folder,
                'regularization_folder': params.dreambooth_regularization_folder,
                'instance': params.dreambooth_instance,
                'class': params.dreambooth_class,
            }
        return signature

    def prompt_embeddings_path(self):
        return prompt_embeddings_path(self.cache_dir, self.accelerator.process_index)

    def save_prompt_embeddings(self):
        # they sit next to the pools, not inside one of them
        os.makedirs(os.path.dirname(self.prompt_embeddings_path()), exist_ok=True)
        torch.save({
            'entries': self.prompt_cache.entries,
            'validation_prompts': list(self.params.validation_prompts or []),
        }, self.prompt_embeddings_path())

    def load_prompt_embeddings(self):
        path = self.prompt_embeddings_path()
        if not os.path.exists(path):
            return False
        try:
            stored = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as error:
            print(f'[Warning] could not read the precomputed prompt embeddings: {error}')
            return False
        self.prompt_cache.entries.update(stored.get('entries') or {})
        return True

    def text_encoder_missing(self):
        """True when the model was built without a text encoder at all."""
        pipe = getattr(self.model, 'pipe', None)
        if pipe is None:
            return False
        return hasattr(pipe, 'text_encoder') and getattr(pipe, 'text_encoder') is None

    def install_prompt_cache(self):
        """Memoize every text embedding call so validation survives the text encoder being freed."""
        self.model.extract_embeddings = self.prompt_cache.wrap('extract_embeddings', self.model.extract_embeddings)
        pipe = getattr(self.model, 'pipe', None)
        if pipe is not None and hasattr(pipe, 'encode_prompt'):
            pipe.encode_prompt = self.prompt_cache.wrap('encode_prompt', pipe.encode_prompt)

    def pool_is_complete(self, pool):
        manifest = read_manifest(self.pool_dirs[pool])
        if manifest is None:
            return False
        if manifest.get('signature') != self.cache_signature():
            return False
        # the instance pool holds the whole (small) dataset, so any of it is the whole of it
        wanted = 1 if pool == INSTANCE_POOL else self.num_batches
        return len(manifest.get('batches', [])) >= wanted

    def cache_is_complete(self):
        if self.params.precompute_force:
            return False
        return all(self.pool_is_complete(pool) for pool in self.pools)

    def all_processes_have_cache(self):
        local = torch.tensor([1 if self.cache_is_complete() else 0],
                             dtype=torch.int64, device=self.accelerator.device)
        return bool(torch.min(self.accelerator.gather(local)).item() == 1)

    def write_batch(self, manifest_batches, batch_index, batch, pool=None, regularization=None):
        latents = batch.vae_features
        embeddings = batch.embeddings
        repa_features = batch.repa_features

        # embeddings are usually one entry per sample, but some models return a
        # (embeddings, masks) pair for the whole batch instead: keep those in a single file
        emb_per_sample = isinstance(embeddings, list) and len(embeddings) == len(latents)

        files = []
        for i in range(len(latents)):
            sample = {
                'latent': to_cpu(latents[i]),
                'embedding': to_cpu(embeddings[i]) if emb_per_sample else None,
                'repa': to_cpu(repa_features[i]) if repa_features is not None else None,
            }
            filename = f'{batch_index:07d}_{i}.pt'
            torch.save(sample, os.path.join(self.pool_dirs[pool], filename))
            files.append(filename)

        entry = {
            'ratio': float(batch.ratio),
            'emb_per_sample': emb_per_sample,
            'files': files,
        }
        if regularization is not None:
            entry['regularization'] = bool(regularization)
        if not emb_per_sample:
            emb_file = f'{batch_index:07d}_emb.pt'
            torch.save(to_cpu(embeddings), os.path.join(self.pool_dirs[pool], emb_file))
            entry['emb_file'] = emb_file
        manifest_batches.append(entry)

    def write_manifest(self, manifest_batches, pool=None):
        manifest = {
            'batch_size': self.batch_size,
            'use_repa': bool(self.params.use_repa),
            'signature': self.cache_signature(),
            'batches': manifest_batches,
        }
        with open(os.path.join(self.pool_dirs[pool], 'manifest.json'), 'w') as f:
            json.dump(manifest, f)

    def run(self, sampler):
        """Fill the cache from sampler, unless a usable one is already on disk."""
        self.install_prompt_cache()

        # the model being trained is not needed until the features are on disk, so its vram goes
        # to the vae and the text encoder for the duration
        cache_ready = self.all_processes_have_cache()
        moved_to_cpu = False
        if not cache_ready:
            moved_to_cpu = self.move_model_to_cpu()
            self.report_memory('with the model parked on the cpu')

        if cache_ready:
            samples = sum(len(read_manifest(self.pool_dirs[pool])['batches'])
                          for pool in self.pools) * self.batch_size
            print(f'skipping the extraction, reusing the {samples} samples already precomputed in '
                  f'{self.cache_dir} (delete that folder or set precompute_force to extract again)')
            self.load_prompt_embeddings()

            # nothing can be encoded any more, so a missing prompt has to say so clearly instead
            # of failing somewhere deep in the pipeline
            if self.text_encoder_missing():
                self.prompt_cache.freeze()
                self.text_encoder_freed = True
                self.install_text_encoder_stub()
        else:
            for pool in self.pools:
                self.extract_pool(sampler, pool)

        # from here on the text encoder is on its way out, so remember every embedding it produces
        self.prompt_cache.recording = True

        # the empty embedding is used for unconditional training steps, so it has to be cached
        # before the text encoder goes away
        with torch.no_grad():
            self.model.empty_embeddings = self.model.extract_embeddings([''])

        # a model that can encode its validation prompts up front lets us drop the text encoder
        # right now, instead of carrying it through the lora setup and the first training steps
        validation_ready = False
        if self.accelerator.is_main_process:
            encode_validation_prompts = getattr(self.model, 'encode_validation_prompts', None)
            with torch.no_grad():
                validation_ready = bool(encode_validation_prompts and encode_validation_prompts())
            if not validation_ready:
                print('this model cannot encode its validation prompts up front, so the text '
                      'encoder stays loaded until the first validation has run')

        # keep them for the next run, which can then skip loading the text encoder entirely
        self.save_prompt_embeddings()

        self.accelerator.wait_for_everyone()
        self.report_memory('before freeing the vae and the text encoder')
        self.free_vae()

        # only the main process validates, so every other process can drop the text encoder now
        if validation_ready or not self.accelerator.is_main_process:
            self.free_text_encoder()

        # now that they are gone there is room for the model again
        if moved_to_cpu:
            self.model.model.to(self.accelerator.device)
        self.report_memory('after freeing them, model back on the gpu')

        return PrecomputedBucketSampler(self.cache_dir,
                                        self.accelerator,
                                        self.params.dataset_seed,
                                        use_repa=self.params.use_repa,
                                        model=self.model,
                                        pools=self.pools,
                                        num_repeats=getattr(self.params, 'dreambooth_num_repeats', 1))

    def extract_pool(self, sampler, pool):
        """Fill one pool from the sampler.

        The instance pool is read once, to the end of the dataset: repeating those images is the
        replay's job, not something worth encoding again. The regularization pool is read until
        precompute_size samples are on disk.
        """
        directory = self.pool_dirs[pool]
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

        # whatever the previous phase left half filled must not leak into this pool
        for key in getattr(sampler, 'buckets', {}):
            sampler.buckets[key].clear()

        instance = pool == INSTANCE_POOL
        if pool != None:
            # the dreambooth sampler queues one kind of shard or the other depending on this
            sampler.mode = pool
        limit = None if instance else self.num_batches

        label = f'Precomputing {pool} features' if pool else 'Precomputing latents and embeddings'
        manifest_batches = []
        pbar = tqdm(total=limit, desc=label, disable=not self.accelerator.is_main_process)
        for batch in sampler:
            self.write_batch(manifest_batches, len(manifest_batches), batch, pool=pool,
                             regularization=getattr(sampler, 'reg_shard', None))
            pbar.update(1)
            if limit != None and len(manifest_batches) >= limit:
                break
        pbar.close()
        self.write_manifest(manifest_batches, pool=pool)
        stop_sampler(sampler)

        samples = len(manifest_batches) * self.batch_size
        label = f'{pool} ' if pool else ''
        print(f'precomputed {samples} {label}samples in {directory}')

    def move_model_to_cpu(self):
        """Park the model being trained on the cpu, so the vae and the text encoder get the vram.

        This is measured to be safe for a 4 bit, device_map dispatched model: the weights come back
        bit identical and nothing is left behind on the gpu. Older bitsandbytes versions refuse the
        move outright, which is what the except is for.
        """
        try:
            self.model.model.to('cpu')
            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as error:
            print(f'[Warning] could not move the model to the cpu for the precompute pass: {error}')
            return False

    def report_memory(self, stage):
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated(self.accelerator.device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(self.accelerator.device) / (1024 ** 3)
        print(f'[rank{self.accelerator.process_index}] vram {stage}: '
              f'{allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved')

    def vae(self):
        pipe = getattr(self.model, 'pipe', None)
        vae = getattr(pipe, 'vae', None) if pipe is not None else None
        if vae is None:
            vae = getattr(self.model, 'vae', None)
        return vae

    def text_encoders(self):
        pipe = getattr(self.model, 'pipe', None)
        holders = [holder for holder in (pipe, self.model) if holder is not None]
        found = []
        for holder in holders:
            for name in ('text_encoder', 'text_encoder_2', 'text_encoder_3'):
                encoder = getattr(holder, name, None)
                if encoder is not None and not isinstance(encoder, FreedTextEncoder):
                    found.append((holder, name))
        return found

    def free_vae(self):
        vae = self.vae()
        if vae is None:
            return
        try:
            vae.to('cpu')
        except Exception as error:
            print(f'[Warning] could not move the vae to the cpu: {error}')
        gc.collect()
        torch.cuda.empty_cache()

    def vae_to_device(self):
        vae = self.vae()
        if vae is None:
            return
        try:
            vae.to(self.accelerator.device)
        except Exception as error:
            print(f'[Warning] could not move the vae back to the gpu: {error}')

    @staticmethod
    def encoder_dtype(encoder):
        dtype = getattr(encoder, 'dtype', None)
        if dtype is not None:
            return dtype
        for parameter in encoder.parameters():
            return parameter.dtype
        return torch.bfloat16

    def install_text_encoder_stub(self):
        """Put the stub where a text encoder that was never loaded would have been."""
        pipe = getattr(self.model, 'pipe', None)
        if pipe is None:
            return
        dtype = getattr(pipe, 'dtype', None) or torch.bfloat16
        for name in ('text_encoder', 'text_encoder_2', 'text_encoder_3'):
            if hasattr(pipe, name) and getattr(pipe, name) is None:
                setattr(pipe, name, FreedTextEncoder(dtype))

    def free_text_encoder(self):
        if self.text_encoder_freed:
            return
        for holder, name in self.text_encoders():
            encoder = getattr(holder, name)
            dtype = self.encoder_dtype(encoder)
            release_module(encoder)
            # the validation code moves the text encoder to the gpu and reads its dtype before it
            # gets to the memoized embeddings, so a stub takes its place instead of None
            setattr(holder, name, FreedTextEncoder(dtype))
        self.text_encoder_freed = True
        self.prompt_cache.freeze()
        gc.collect()
        torch.cuda.empty_cache()

    def before_validation(self):
        # the vae decodes the validation images, so it goes back to the gpu for the duration
        self.vae_to_device()

    def after_validation(self):
        self.free_vae()
        # the validation prompts are memoized by now, so the text encoder is no longer needed
        self.free_text_encoder()


def stop_sampler(sampler):
    """Stop the shard download process the sampler started for its iteration."""
    process = getattr(sampler, 'download_process', None)
    if process is None:
        return
    if process.is_alive():
        process.terminate()
        process.join()
    sampler.download_process = None
