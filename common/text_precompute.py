"""Precompute the text encoder features of a whole dataset, and leave the vae alone.

Unlike FeaturesPrecompute (common/precompute.py), which stores latents and embeddings of a pool of
samples, this only caches one embedding per caption. The training keeps using the regular feature
extracting sampler, so the images are still encoded by the vae at every step, but the sampler asks
the cache for the embeddings instead of the text encoder, which is freed.

Layout of text_embeddings_cache_dir:
    meta.json               the model the embeddings were made with
    prompt_embeddings.pt    the validation prompts, memoized for pipe.encode_prompt
    ab/abcdef....pt         the embedding of one caption, named after the sha1 of the caption
"""
import functools
import hashlib
import json
import os
import tarfile

import torch
from tqdm import tqdm

from common.precompute import FeaturesPrecompute, PromptEmbeddingCache

PROMPT_EMBEDDINGS_FILE = 'prompt_embeddings.pt'
META_FILE = 'meta.json'


@functools.lru_cache(maxsize=4)
def read_captions(shard_paths):
    """Every distinct caption of the shards, sorted. Only the .txt members of the tars are read."""
    captions = set()
    for path in shard_paths:
        if not os.path.exists(path):
            print(f'[Warning] {path} does not exist, its captions are skipped')
            continue
        with tarfile.open(path) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = os.path.basename(member.name)
                # webdataset: everything after the first dot of the file name is the extension
                if '.' not in name or name.split('.', 1)[1] != 'txt':
                    continue
                captions.add(tar.extractfile(member).read().decode('utf-8'))
    return sorted(captions)


def dataset_captions(params):
    # the empty caption is the unconditional embedding
    return read_captions(tuple(params.local_shard_paths)) + ['']


def caption_path(cache_dir, caption):
    digest = hashlib.sha1(caption.encode('utf-8')).hexdigest()
    return os.path.join(cache_dir, digest[:2], digest + '.pt')


def missing_captions(params):
    return [caption for caption in dataset_captions(params)
            if not os.path.exists(caption_path(params.text_embeddings_cache_dir, caption))]


def check_meta(params):
    """Embeddings of another model are useless: refuse them instead of training on noise."""
    path = os.path.join(params.text_embeddings_cache_dir, META_FILE)
    if not os.path.exists(path):
        return
    with open(path) as f:
        meta = json.load(f)
    if meta.get('pretrained_pipe_path') != params.pretrained_pipe_path:
        raise RuntimeError(
            f'{params.text_embeddings_cache_dir} holds embeddings of {meta.get("pretrained_pipe_path")}, '
            f'not of {params.pretrained_pipe_path}: delete it or set another text_embeddings_cache_dir')


def text_embeddings_available(params):
    """True when every caption and the validation prompts are on disk, so no text encoder is needed."""
    if not getattr(params, 'text_embeddings_cache_dir', None) or not params.precompute_text_embeddings:
        return False
    if getattr(params, 'text_embeddings_force', False):
        return False
    check_meta(params)
    path = os.path.join(params.text_embeddings_cache_dir, PROMPT_EMBEDDINGS_FILE)
    if not os.path.exists(path):
        return False
    try:
        stored = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return False
    if list(stored.get('validation_prompts') or []) != list(params.validation_prompts or []):
        return False
    return len(missing_captions(params)) == 0


class TextEmbeddingsPrecompute(FeaturesPrecompute):
    """Encodes every caption once, then swaps the text encoder for a lookup on disk."""

    def __init__(self, model):
        # FeaturesPrecompute.__init__ is about the pool of latents, none of it applies here
        self.model = model
        self.params = model.params
        self.accelerator = model.accelerator
        self.cache_dir = self.params.text_embeddings_cache_dir
        self.prompt_cache = PromptEmbeddingCache(self.accelerator.device)
        self.text_encoder_freed = False

    def prompt_embeddings_path(self):
        return os.path.join(self.cache_dir, PROMPT_EMBEDDINGS_FILE)

    def save_prompt_embeddings(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        torch.save({
            'entries': self.prompt_cache.entries,
            'validation_prompts': list(self.params.validation_prompts or []),
        }, self.prompt_embeddings_path())

    def install_lookup(self):
        """The sampler asks model.extract_embeddings for its captions: answer from the disk."""
        device = self.accelerator.device
        cache_dir = self.cache_dir

        def lookup(captions):
            embeddings = []
            for caption in captions:
                path = caption_path(cache_dir, caption)
                if not os.path.exists(path):
                    raise KeyError(f'no precomputed text embedding for the caption {caption!r} in {cache_dir}: '
                                   f'the dataset changed, set text_embeddings_force to encode it again')
                embeddings.append(torch.load(path, map_location='cpu', weights_only=True).to(device))
            return embeddings

        self.model.extract_embeddings = lookup

    def encode_captions(self, captions, force=False):
        """This process encodes its share of the captions that are not on disk yet.

        captions has to be the same list on every process (the full sorted caption list): it is
        split first and only then filtered against the disk. Filtering first gives each process a
        different list, since the other one is already writing files, and then the shares overlap
        and leave captions to nobody.
        """
        rank = self.accelerator.process_index
        mine = captions[rank::self.accelerator.num_processes]
        if not force:
            mine = [caption for caption in mine if not os.path.exists(caption_path(self.cache_dir, caption))]
        step = max(1, self.params.text_encoder_max_batch_size)
        extract = self.model.extract_embeddings
        pbar = tqdm(total=len(mine), desc='Precomputing text embeddings',
                    disable=not self.accelerator.is_main_process)
        with torch.no_grad():
            for i in range(0, len(mine), step):
                chunk = mine[i:i + step]
                embeddings = extract(chunk)
                for caption, embedding in zip(chunk, embeddings):
                    path = caption_path(self.cache_dir, caption)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    # written under another name first, so a run stopped midway never leaves a torn file
                    temporary = f'{path}.{rank}.tmp'
                    torch.save(embedding.detach().to('cpu'), temporary)
                    os.replace(temporary, path)
                pbar.update(len(chunk))
        pbar.close()

    def run(self, sampler):
        check_meta(self.params)
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(os.path.join(self.cache_dir, META_FILE), 'w') as f:
            json.dump({'pretrained_pipe_path': self.params.pretrained_pipe_path}, f)

        # validation encodes its prompts through the pipeline, which the memo has to see
        pipe = getattr(self.model, 'pipe', None)
        if pipe is not None and hasattr(pipe, 'encode_prompt'):
            pipe.encode_prompt = self.prompt_cache.wrap('encode_prompt', pipe.encode_prompt)

        if self.params.text_embeddings_force:
            missing = dataset_captions(self.params)
        else:
            missing = missing_captions(self.params)

        if self.text_encoder_missing():
            # the model was built without the text encoder because everything was on disk
            if missing:
                raise RuntimeError(f'{len(missing)} caption embeddings are missing but the text encoder was '
                                   f'not loaded, set text_embeddings_force to encode them')
            stored = torch.load(self.prompt_embeddings_path(), map_location='cpu', weights_only=False)
            self.prompt_cache.entries.update(stored.get('entries') or {})
            self.prompt_cache.freeze()
            self.text_encoder_freed = True
            self.install_text_encoder_stub()
            print(f'using the {len(dataset_captions(self.params))} precomputed text embeddings of '
                  f'{self.cache_dir}, the text encoder is not loaded')
        else:
            if missing:
                print(f'encoding {len(missing)} captions')
                self.encode_captions(dataset_captions(self.params), force=self.params.text_embeddings_force)
            else:
                print(f'all the text embeddings are already in {self.cache_dir}')
            self.accelerator.wait_for_everyone()

            # from here the validation prompts are memoized so validation survives the encoder being freed
            self.prompt_cache.recording = True
            validation_ready = False
            if self.accelerator.is_main_process:
                with torch.no_grad():
                    validation_ready = bool(self.model.encode_validation_prompts())
                self.save_prompt_embeddings()
                if not validation_ready:
                    print('this model cannot encode its validation prompts up front, so the text encoder '
                          'stays loaded until the first validation has run')
            self.accelerator.wait_for_everyone()
            if validation_ready or not self.accelerator.is_main_process:
                self.free_text_encoder()
            self.report_memory('after freeing the text encoder, the vae stays loaded')

        self.install_lookup()
        # the unconditional embedding comes from the same files
        self.model.empty_embeddings = self.model.extract_embeddings([''])
        return sampler

    def before_validation(self):
        pass

    def after_validation(self):
        # the vae is still needed by the training, only the text encoder goes
        self.free_text_encoder()
