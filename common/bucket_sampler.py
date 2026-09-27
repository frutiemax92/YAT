from tqdm import tqdm
from torch.utils.data import BatchSampler, IterableDataset
import random
import itertools
from copy import copy
import torch
from accelerate import Accelerator
import gc
from torchvision import transforms
import os
from tarfile import ReadError
from common.cloudflare import get_secured_urls, download_tar
import webdataset as wds
import argparse
from common.training_parameters_reader import TrainingParameters
from collections import deque
import io
from transformers import AutoImageProcessor, AutoModel
import multiprocessing as mp
import time
import random
import torch.distributed as dist
from PIL import Image
from PIL import ImageFile
from queue import Empty

ImageFile.LOAD_TRUNCATED_IMAGES = True

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def decode_image(elem):
    """The image of a webdataset sample, whatever extension it was stored under."""
    for extension in ('jpg', 'jpeg', 'png', 'webp'):
        if extension in elem.keys():
            return elem[extension]
    raise KeyError(f'no image in the sample {elem.get("__key__")}, only {sorted(elem.keys())}')

class Batch:
    def __init__(self):
        self.embeddings = None
        self.vae_features = None
        self.repa_features = None
        self.ratio = None
        self.repa_spatial_dims = None  # (height, width) of repa token grid
        self.proj_spatial_dims = None  # (height, width) of proj token grid
        # dreambooth only: True for a class batch, False for an instance batch
        self.is_regularization = None
        # lightweight controlnet only: the ReferenceFeatures of the reference images of this
        # batch, None for a batch that came from a shard without a ref/ folder
        self.reference_features = None


class BucketSampler:
    def __init__(self,
                 shards : list[str],
                 features_path : str,
                 accelerator : Accelerator,
                 batch_size : int,
                 r2_access_key : str,
                 r2_secret_key : str,
                 r2_endpoint : str,
                 r2_bucket_name : str,
                 cache_size : int,
                 seed : int,
                 model,
                 use_repa : bool = False,
                 local_temp_dir : str = 'temp',
                 local_paths: list[str] = None,
                 ):
        self.process_index = accelerator.process_index
        self.num_processes = accelerator.num_processes
        self.device = accelerator.device
        self.accelerator = accelerator
        self.shards = shards
        self.model = model
        self.buckets = {float(key) : [] for key in model.aspect_ratios.keys()}
        self.valid_buckets = set()
        self.ready_bucket = -1.0
        self.seed = seed
        self.use_repa = use_repa
        self.batch_size = batch_size
        self.r2_access_key = r2_access_key
        self.r2_secret_key = r2_secret_key
        self.r2_endpoint = r2_endpoint
        self.r2_bucket_name = r2_bucket_name
        self.local_temp_dir = local_temp_dir
        self.features_path = features_path
        self.cache_size = cache_size
        self.local_paths = local_paths
        self.bucket_to_yield = torch.tensor([-1])

        # set while iterating, so the shard downloader can be stopped when we stop consuming batches
        self.download_process = None

        def local_file_getter(process_index : int, to_train : mp.Queue, to_remove : mp.Queue):
            print('using local file getter')
            
            local_path = random.choice(self.local_paths)
            while os.path.exists(local_path) == False:
                local_path = random.choice(self.local_paths)
            to_train.put(local_path)

            # the simplest solution is to wait for to_remove
            r = to_remove.get()

        def download_shard_worker(process_index : int,
            to_train : mp.Queue, 
            to_remove : mp.Queue):
            current_item = 0
            num_shards = 0
            random.seed(self.process_index + self.seed)
            while True:
                if to_train.qsize() < 4:
                    current_shard_index = self.get_next_shard_index()
                    dataset_url = get_secured_urls(self.r2_access_key,
                                        self.r2_secret_key,
                                        self.r2_endpoint,
                                        self.r2_bucket_name,
                                        [self.features_path + '/' + self.shards[current_shard_index]]
                                        )[0]
                    local_shard_path = self.local_temp_dir + f'/shard_{process_index}_{current_item}.tar'
                    try:
                        download_tar(dataset_url, local_shard_path)
                    except Exception as error:
                        print(error)
                        current_shard_index = self.get_next_shard_index()
                        continue
                    to_train.put(local_shard_path)
                    current_item = current_item + 1

                try:
                    r = to_remove.get(timeout=10)
                    self.cleanup_shard(r)
                except:
                    pass
        
        if self.local_paths == None:
            self.download_shard_proc = download_shard_worker
        else:
            self.download_shard_proc = local_file_getter
        
        os.makedirs(self.local_temp_dir, exist_ok=True)

        if use_repa:
            import timm
            self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.accelerator.device)
            #self.dino = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True).to(self.accelerator.device)
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad = False

    def pt_decoder(self, key, value):
        if key.endswith("pt"):
            return torch.load(io.BytesIO(value), map_location="cpu")
        elif key.endswith("txt"):
            return value.decode("utf-8")
        elif key in (".jpg", ".jpeg", ".png"):
            return Image.open(io.BytesIO(value)).convert("RGB")
        else:
            return value
    
    def add_bucket(self, ratio):
        self.buckets[ratio] = deque(maxlen=self.batch_size)

    def attach_extras(self, batch, samples, ratio):
        """Hook for the samplers whose samples carry more than an image and a caption."""
        pass

    def begin_shard(self):
        """Hook called before a shard is read, for the state that does not outlive one."""
        pass

    def reset_buckets(self):
        """Drop everything half accumulated, so one phase does not leak into the next."""
        for key in self.buckets:
            self.buckets[key].clear()

    def process_element(self, elem):
        ratio, latent, emb = elem['ratio'], elem['latent.pt'], elem['emb.pt']
        ratio = float(ratio)
        if not ratio in self.buckets.keys():
            self.add_bucket(ratio)
        self.buckets[ratio].append((latent, emb))
    
    def get_next_shard_index(self):
        return random.randint(0, len(self.shards) - 1)
    
    def extract_features(self, batch, ratio):
        return torch.stack(batch[0]), batch[1]
    
    def cleanup_shard(self, local_shard_path):
        # Clean up at end of shard
        if os.path.exists(local_shard_path):
            os.remove(local_shard_path)
    
    def get_invalid_bucket(self):
        return [-1.0]
    
    def get_unique_ratios(self, dist_valid_buckets):
        unique_ratios, counts = torch.unique(dist_valid_buckets, return_counts=True)
        ratio_counts = {float(r): int(c) for r, c in zip(unique_ratios, counts) 
                        if r != -1.0}
        return ratio_counts
    
    def get_ratio_from_key(self, key):
        return key
    
    def find_closest_ratio(self, ratio):
        pass

    def find_closest_key(self, ratio):
        if ratio == -1.0:
            return -1.0
        
        error = 1000
        res = -1.0
        for r in self.buckets.keys():
            new_error = abs(ratio - r)
            if new_error < error:
                error = new_error
                res = r
        return res
    
    def get_local_shard_path(self, to_train : mp.Queue):
        return to_train.get()
    
    def remove_shard(self, to_remove : mp.Queue, local_shard_path):
        to_remove.put(local_shard_path)
    
    def ready_batches(self, buckets, keep_leftovers=False):
        """Yield a batch for every bucket that holds batch_size samples on every process.

        The bucket sizes are gathered, so every process has to call this in lockstep. A bucket that
        filled up past batch_size while waiting for the other processes is emptied once its batch
        is out, unless keep_leftovers is set: then only the samples of the batch are taken out.
        """
        # this should be deterministic across all processes...
        for key in buckets.keys():
            num_samples = torch.tensor(len(buckets[key]), dtype=torch.int64, device=self.accelerator.device)
            num_samples = self.accelerator.gather(num_samples)

            is_ok = torch.min(num_samples) >= self.batch_size
            is_ok = bool(is_ok.item())

            #self.accelerator.wait_for_everyone()
            if is_ok:
                # consume batch_size samples for the current bucket
                closest_ratio = key
                # a sample is (image, caption), plus whatever else the sampler put after it - a
                # reference image, for the lightweight controlnet - transposed into one list per
                # field. The first two are what every sampler below reads.
                samples = [buckets[closest_ratio][i] for i in range(self.batch_size)]
                batch = tuple(list(column) for column in zip(*samples))

                # now we must extract the text embeddings and vae features, and making sure we don't exceed the vae max batch size
                # as extracting features tends to use more VRAM than actually training the model
                # we freeze the vae and text encoder model
                ratio = self.get_ratio_from_key(closest_ratio)

                gc.collect()
                torch.cuda.empty_cache()
                vae_features, embeddings, repa_features = self.extract_features(batch, ratio)

                batch = Batch()
                batch.ratio = ratio
                batch.embeddings = embeddings
                batch.vae_features = vae_features
                if self.use_repa:
                    batch.repa_features = repa_features
                    # REPA tokens from DINO on 224x224 are always 16x16 (256 tokens)
                    batch.repa_spatial_dims = (16, 16)
                    # Proj token spatial dims will be set by the model based on image size
                    # For now store aspect ratio dimensions
                    aspect_ratio = self.model.aspect_ratios[str(ratio)]
                    target_height = int(aspect_ratio[0])
                    target_width = int(aspect_ratio[1])
                    # Assuming 32x32 patch size, so divide by 32
                    patch_size = getattr(self.model, 'patch_size', 16)
                    batch.proj_spatial_dims = (target_height // patch_size, target_width // patch_size)

                self.attach_extras(batch, samples, ratio)

                yield batch
                if keep_leftovers:
                    del buckets[closest_ratio][:self.batch_size]
                else:
                    buckets[closest_ratio].clear()

                del batch
                del vae_features
                del embeddings
                gc.collect()

    def __iter__(self):
        sync_counter = 0
        
        # start the download process
        to_train = mp.Queue()
        to_remove = mp.Queue()
        p = mp.Process(target=self.download_shard_proc, args=(self.process_index, to_train, to_remove))
        p.start()
        self.download_process = p

        cache_index = 0
        while True:
            local_shard_path = self.get_local_shard_path(to_train)

            # the worker signals that it has nothing more to hand out
            if local_shard_path is None:
                break

            dataset = (
                wds.WebDataset(local_shard_path, shardshuffle=True, nodesplitter=None, workersplitter=None)
                .shuffle(1000)
                .decode(self.pt_decoder)
            )
            self.begin_shard()

            for elem in dataset:
                self.process_element(elem)

                self.accelerator.wait_for_everyone()

                # consume all the accumulated samples in the buckets
                yield from self.ready_batches(self.buckets)

            self.remove_shard(to_remove, local_shard_path)

class BucketSamplerExtractFeatures(BucketSampler):
    def __init__(self,
                 shards : list[str],
                 features_path : str,
                 accelerator : Accelerator,
                 batch_size : int,
                 r2_access_key : str,
                 r2_secret_key : str,
                 r2_endpoint : str,
                 r2_bucket_name : str,
                 vae_max_batch_size : int,
                 text_encoder_max_batch_size : int,
                 model,
                 seed,
                 cache_size : int,
                 use_repa : bool = False,
                 local_temp_dir : str = 'temp',
                 local_paths=None,
                 ):
        super().__init__(shards,
                         features_path,
                         accelerator,
                         batch_size, 
                         r2_access_key,
                         r2_secret_key,
                         r2_endpoint,
                         r2_bucket_name,
                         local_temp_dir,
                         seed,
                         model,
                         use_repa,
                        local_paths=local_paths)
        self.vae_max_batch_size = vae_max_batch_size
        self.text_encoder_max_batch_size = text_encoder_max_batch_size
        self.model = model
        self.cache_size = cache_size
        self.valid_shards = []

    def find_closest_ratio(self, ratio):
        error = 1000
        res = -1
        for r in self.model.aspect_ratios.keys():
            new_error = abs(float(r) - ratio)
            if new_error < error:
                error = new_error
                res = float(r)
        return res
    
    def process_element(self, elem):
        img = decode_image(elem)
        caption = elem['txt']
        ratio = img.height / img.width
        ratio = float(ratio)
        ratio = self.find_closest_ratio(ratio)
        self.buckets[ratio].append((img, caption))
    
    def extract_features(self, batch, ratio, low_vram=False):
        # now we must extract the text embeddings and vae features, and making sure we don't exceed the vae max batch size
        # as extracting features tends to use more VRAM than actually training the model
        # we freeze the vae and text encoder model
        vae_features = []
        embeddings = []
        repa_features = []

        transform = self.get_transform(ratio)
        with torch.no_grad():
            imgs = []
            for i in range(0, len(batch[0]), self.vae_max_batch_size):
                end_index = min(len(batch[0]), i+self.vae_max_batch_size)
                images = batch[0][i:end_index]
                images = torch.stack([transform(x) for x in images]).to(dtype=torch.bfloat16, device=self.accelerator.device)
                imgs.extend(images)
                features = self.model.extract_latents(images)
                vae_features.extend(features)
            
            for i in range(0, len(batch[1]), self.text_encoder_max_batch_size):
                end_index = min(len(batch[1]), i+self.text_encoder_max_batch_size)
                captions = batch[1][i:i+self.text_encoder_max_batch_size]
                embeds = self.model.extract_embeddings(captions)
                embeddings.extend(embeds)
        
            # check if we need to do repa
            # copied from pixeldit
            repa_tokens = None
            if self.use_repa:
                from torchvision.transforms.functional import pil_to_tensor
                #images = torch.stack([pil_to_tensor(img) for img in images])

                import torch.nn.functional as F
                imgs = torch.stack(imgs)
                imgs01 = (imgs.float() + 1.0) / 2.0
                imgs224 = F.interpolate(imgs01, size=(224, 224), mode="bicubic", align_corners=False)

                from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN, device=imgs224.device, dtype=imgs224.dtype).view(1, 3, 1, 1)
                std = torch.as_tensor(IMAGENET_DEFAULT_STD, device=imgs224.device, dtype=imgs224.dtype).view(1, 3, 1, 1)
                imgs224 = (imgs224 - mean) / std
                feats = self.dino.forward_features(imgs224)
                if isinstance(feats, dict):
                    if "x_norm_patchtokens" in feats:
                        repa_tokens = feats["x_norm_patchtokens"]  # [B, 256, 768]
                    elif "tokens" in feats:
                        repa_tokens = feats["tokens"][:, 1:]
                elif isinstance(feats, torch.Tensor):
                    repa_tokens = feats[:, 1:]

        return torch.stack(vae_features), embeddings, repa_tokens
    
    def get_transform(self, bucket):
        aspect_ratio = self.model.aspect_ratios[str(bucket)]
        target_height = int(aspect_ratio[0])
        target_width = int(aspect_ratio[1])
        return transforms.Compose([
            transforms.Resize((target_height, target_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

class ReferencePairing:
    """Pairs the src/<name> and ref/<name> halves of a lightweight controlnet shard back together.

    webdataset keys a sample by its path minus the extension, so the image and its reference reach
    the sampler as two separate samples, in whatever order the tar and the shuffle buffer produce
    them. What is still unmatched when a shard ends was a src without a ref or the other way
    around, and is dropped by begin_shard.

    Each stream of shards needs a pairing of its own: the instance and the regularization tars of
    a dreambooth run number their samples the same way, so a shared one would pair the image of
    one with the reference of the other.
    """
    def __init__(self):
        self.src = {}
        self.ref = {}

    def clear(self):
        self.src.clear()
        self.ref.clear()

    def pair(self, elem):
        """(image, caption, reference) once a sample is complete, None while it waits for its half.

        A sample outside a src/ or ref/ folder is an ordinary image and caption: it is complete on
        its own and carries no reference, so it trains the lora alone. The caption comes back as
        None when the shard holds none, which is what a dreambooth instance tar looks like: the
        caller fills in the instance prompt.
        """
        key = elem.get('__key__', '')
        caption = elem['txt'] if 'txt' in elem.keys() else None

        if key.startswith('src/'):
            name = key[len('src/'):]
            reference = self.ref.pop(name, None)
            if reference is None:
                self.src[name] = (decode_image(elem), caption)
                return None
            return decode_image(elem), caption, reference

        if key.startswith('ref/'):
            name = key[len('ref/'):]
            source = self.src.pop(name, None)
            if source is None:
                self.ref[name] = decode_image(elem)
                return None
            return source[0], source[1], decode_image(elem)

        return decode_image(elem), caption, None


class ReferencePairedSamples:
    """The bucket layout and the reference encoding of a lightweight controlnet run.

    Mixed into a sampler, it stays inert until setup_reference_pairing is told the run wants a
    reference path: the bucket keys, the ratio they are read back with and the features attached
    to a batch all fall back to what the sampler did before.

    With it on, referenced and plain samples are bucketed apart, the bucket key being
    (ratio, has_reference), so a batch never mixes the two and the reference path is used either
    by every sample of a step or by none of them.
    """
    def setup_reference_pairing(self, train_lw_controlnet):
        self.train_lw_controlnet = train_lw_controlnet
        if not train_lw_controlnet:
            return
        # ready_batches walks the keys in a fixed order and gathers their counts, so every process
        # has to hold the same key set
        self.buckets = {(float(key), has_reference): []
                        for key in self.model.aspect_ratios.keys()
                        for has_reference in (0, 1)}

    def new_pairing(self):
        return ReferencePairing() if self.train_lw_controlnet else None

    def get_ratio_from_key(self, key):
        return key[0] if self.train_lw_controlnet else key

    def add_sample(self, buckets, image, caption, reference):
        ratio = self.find_closest_ratio(float(image.height) / image.width)
        if not self.train_lw_controlnet:
            buckets[ratio].append((image, caption))
        elif reference is None:
            buckets[(ratio, 0)].append((image, caption))
        else:
            buckets[(ratio, 1)].append((image, caption, reference))

    def attach_extras(self, batch, samples, ratio):
        if len(samples[0]) < 3:
            return
        batch.reference_features = self.model.encode_reference_images(
            [sample[2] for sample in samples])


class BucketSamplerLwControlNet(ReferencePairedSamples, BucketSamplerExtractFeatures):
    """Reads the tars of a lightweight controlnet run, with or without a reference image.

    A shard made by create_lw_controlnet_tar.py holds src/<name>.jpg, src/<name>.txt and
    ref/<name>.jpg, which ReferencePairing puts back together. An ordinary shard, with images and
    captions at its root and no ref/ folder, is read exactly as it always was and its batches
    carry no reference - those steps train the lora alone.
    """
    def __init__(self, *args, lw_controlnet_tars=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_reference_pairing(True)
        self.pairing = self.new_pairing()

        self.lw_controlnet_tars = lw_controlnet_tars
        if lw_controlnet_tars:
            def local_tar_worker(process_index : int, to_train : mp.Queue, to_remove : mp.Queue):
                """Hands out the local tars in a loop, and never deletes them.

                The inherited local file getter yields a single shard and then exits, which is
                what the precompute pass wants but would hang a training that keeps asking for
                more. These tars are the whole dataset, so they are cycled over instead.
                """
                rng = random.Random(self.process_index + self.seed)
                order = list(self.lw_controlnet_tars)
                while True:
                    rng.shuffle(order)
                    for path in order:
                        if not os.path.exists(path):
                            print(f'[Warning] {path} does not exist, skipping it')
                            continue
                        to_train.put(path)
                        # the trainer is done with the shard before it asks for the next one
                        to_remove.get()

            self.download_shard_proc = local_tar_worker

    def reset_buckets(self):
        super().reset_buckets()
        self.pairing.clear()

    def begin_shard(self):
        self.pairing.clear()

    def process_element(self, elem):
        sample = self.pairing.pair(elem)
        if sample is None:
            # the other half of the pair has not come out of the shard yet
            return
        image, caption, reference = sample
        if caption is None:
            raise KeyError(f'no caption for the sample {elem.get("__key__")}')
        self.add_sample(self.buckets, image, caption, reference)


class BucketSamplerDreambooth(ReferencePairedSamples, BucketSamplerExtractFeatures):
    """Alternates an instance batch and a class batch, as in the dreambooth paper.

    The two streams are bucketed apart, so a batch never mixes instance and class images. The
    instance images are cycled over for as long as the training runs, the class images come from
    dreambooth_reg_tar when it is set, otherwise from the regularization folder or from the dataset
    shards. Every batch carries is_regularization, which the trainer uses to weight the class
    loss by dreambooth_lambda.

    With train_lw_controlnet, both streams also read the src/ref layout: an instance tar built by
    create_lw_controlnet_tar.py trains the reference path on the instance images, and the
    regularization tar can be either kind - a plain tar of images and captions keeps the prior of
    the lora alone, a src/ref one also regularizes the reference path. Each stream pairs with its
    own ReferencePairing, since the two tars number their samples the same way.
    """
    def __init__(self,
                 dreambooth_instance_tar : str,
                 dreambooth_regularization_folder : str,
                 dreambooth_instance : str,
                 dreambooth_class : str,
                 accelerator : Accelerator,
                 batch_size : int,
                 vae_max_batch_size : int,
                 text_encoder_max_batch_size : int,
                 model,
                 seed,
                 cache_size : int,
                 local_temp_dir : str = 'temp',
                 r2_access_key : str = None,
                 r2_secret_key : str = None,
                 r2_endpoint : str = None,
                 r2_bucket_name : str = None,
                 shards : list[str] = None,
                 dreambooth_reg_tar : str = None,
                 train_lw_controlnet : bool = False,
        ):
        super().__init__(
            shards,
            dreambooth_regularization_folder,
            accelerator,
            batch_size,
            r2_access_key,
            r2_secret_key,
            r2_endpoint,
            r2_bucket_name,
            vae_max_batch_size,
            text_encoder_max_batch_size,
            model,
            seed,
            cache_size,
            # keywords: local_temp_dir used to land in the use_repa slot, which loaded dinov2 and
            # extracted repa features on every dreambooth run
            use_repa=False,
            local_temp_dir=local_temp_dir,
        )
        self.setup_reference_pairing(train_lw_controlnet)
        self.dreambooth_instance_tar = dreambooth_instance_tar
        self.dreambooth_regularization_folder = dreambooth_regularization_folder
        self.dreambooth_reg_tar = dreambooth_reg_tar
        self.dreambooth_instance = dreambooth_instance
        self.dreambooth_class = dreambooth_class

        # 'alternate' is the training behaviour: an instance batch, then a class batch, on repeat.
        # The precompute pass asks for one stream at a time instead, and reads the instance images
        # only once: the replay cycles over them from the cache.
        self.mode = 'alternate'

        def download_shard_worker(process_index : int, to_train : mp.Queue, to_remove : mp.Queue):
            # only the class images go through here, the instance images and dreambooth_reg_tar are
            # read directly
            random.seed(self.process_index + self.seed)
            queued = 0
            while True:
                if to_train.qsize() < 4:
                    # we get the regularization images either from a local folder or a bucket on the cloud
                    if self.r2_bucket_name == None:
                        to_train.put(self.dreambooth_regularization_folder)
                        continue

                    dataset_url = get_secured_urls(self.r2_access_key,
                                self.r2_secret_key,
                                self.r2_endpoint,
                                self.r2_bucket_name,
                                [self.features_path + '/' + self.shards[self.get_next_shard_index()]]
                                )[0]
                    local_shard_path = self.local_temp_dir + f'/shard_{process_index}_{queued}.tar'
                    try:
                        download_tar(dataset_url, local_shard_path)
                    except Exception as error:
                        print(error)
                        continue
                    to_train.put(local_shard_path)
                    queued = queued + 1
                    continue

                # don't run ahead of the training, and clean up the shards it is done with. Only the
                # shards we downloaded ourselves get deleted, never a local folder the user pointed us at
                try:
                    local_shard_path = to_remove.get(timeout=1)
                    if self.r2_bucket_name != None:
                        self.cleanup_shard(local_shard_path)
                except Empty:
                    pass

        self.download_shard_proc = download_shard_worker

    def add_element(self, buckets, pairing, elem, is_regularization):
        """Put one webdataset sample in its bucket, once it is complete.

        Without a pairing this is the image and the caption of the sample. With one, a src/ half
        waits for its ref/ half and the sample only lands in a bucket when both are in.
        """
        if pairing is None:
            img, caption, reference = decode_image(elem), elem.get('txt'), None
        else:
            sample = pairing.pair(elem)
            if sample is None:
                return
            img, caption, reference = sample

        # a tar of bare images takes its caption from the prompts of the config. Resolved here and
        # not before, so a run that captions every image never has to have them set at all.
        if caption is None:
            caption = (self.dreambooth_class if is_regularization
                       else self.dreambooth_instance + ' ' + self.dreambooth_class)

        self.add_sample(buckets, img, caption, reference)

    def shard_batches(self, shard_paths, is_regularization, keep_leftovers=False):
        """Yield the batches of one stream, with buckets of its own.

        keep_leftovers is for the small local tars: every image of them has to make it into a batch,
        instead of being dropped when a bucket fills up faster on one process than on the others.
        The downloaded shards are big and differ between processes, so a bucket could grow without
        end there: they keep the regular behaviour.
        """
        buckets = {key : [] for key in self.buckets.keys()}
        # the pairing belongs to this stream, and nothing of it survives one shard
        pairing = self.new_pairing()
        for local_shard_path in shard_paths:
            dataset = (
                wds.WebDataset(local_shard_path, shardshuffle=True, nodesplitter=None, workersplitter=None)
                .shuffle(1000)
                .decode(self.pt_decoder)
            )
            if pairing != None:
                pairing.clear()

            num_elements = 0
            for elem in dataset:
                self.add_element(buckets, pairing, elem, is_regularization)
                num_elements = num_elements + 1

                self.accelerator.wait_for_everyone()
                for batch in self.ready_batches(buckets, keep_leftovers):
                    batch.is_regularization = is_regularization
                    yield batch

            if keep_leftovers:
                # what is left was waiting on the other processes: hand it out before the next pass
                while True:
                    ready = False
                    for batch in self.ready_batches(buckets, keep_leftovers):
                        ready = True
                        batch.is_regularization = is_regularization
                        yield batch
                    if not ready:
                        break

            # the local tars are cycled over forever, so an empty one would hang the training
            if num_elements == 0:
                kind = 'regularization' if is_regularization else 'instance'
                raise RuntimeError(f'no {kind} images found in {local_shard_path}')

    def regularization_shards(self, to_train : mp.Queue, to_remove : mp.Queue):
        while True:
            local_shard_path = to_train.get()
            yield local_shard_path
            # shard_batches only asks for the next shard once it is done with this one
            to_remove.put(local_shard_path)

    def __iter__(self):
        if self.mode == 'instance':
            # one pass over the instance images: the features only need to be calculated once
            yield from self.shard_batches([self.dreambooth_instance_tar], False, keep_leftovers=True)
            return

        if self.dreambooth_reg_tar != None:
            # a single local tar, nothing to download: the precompute pass reads it once at most,
            # the training cycles over it like the instance images
            if self.mode == 'regularization':
                regularization = self.shard_batches([self.dreambooth_reg_tar], True, keep_leftovers=True)
            else:
                regularization = self.shard_batches(itertools.repeat(self.dreambooth_reg_tar), True, keep_leftovers=True)
        else:
            to_train = mp.Queue()
            to_remove = mp.Queue()
            p = mp.Process(target=self.download_shard_proc, args=(self.process_index, to_train, to_remove))
            p.start()
            self.download_process = p
            regularization = self.shard_batches(self.regularization_shards(to_train, to_remove), True)

        if self.mode == 'regularization':
            yield from regularization
            return

        instance = self.shard_batches(itertools.repeat(self.dreambooth_instance_tar), False, keep_leftovers=True)
        while True:
            yield next(instance)
            yield next(regularization)

    def find_closest_key(self, ratio):
        error = 1000
        res = -1
        for r in self.buckets.keys():
            if r[0] != ratio[0]:
                continue
            new_error = abs(ratio[1] - r[1])
            if new_error < error:
                error = new_error
                res = r
        return res
    
    def get_transform(self, bucket):
        aspect_ratio = self.model.aspect_ratios[str(bucket)]
        target_height = int(aspect_ratio[0])
        target_width = int(aspect_ratio[1])
        return transforms.Compose([
            transforms.Resize((target_height, target_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    def get_unique_ratios(self, dist_valid_buckets):
        # Get unique (is_reg_pass, ratio) pairs row-wise
        unique_pairs, counts = torch.unique(dist_valid_buckets, return_counts=True, dim=0)

        # Convert to dictionary: {(is_reg_pass, ratio): count}
        ratio_counts = {
            (int(pair[0].item()), float(pair[1].item())): int(c.item())
            for pair, c in zip(unique_pairs, counts)
            if pair[1].item() != -1.0  # filter invalid ratio
        }
        return ratio_counts

## dual-gpu BucketSampler __iter__ patch

def dual_gpu_bucket_sampler_iter(self):
    random.seed(self.process_index + self.seed)
    print('using dual_gpu_bucket_sampler_iter')

    # start the download process
    to_train = mp.Queue()
    to_remove = mp.Queue()
    p = mp.Process(target=self.download_shard_proc, args=(self.process_index, to_train, to_remove))

    # download/process the elements
    if self.accelerator.process_index == 1:
        p.start()

    def features_yielding_process(self):
        while True:
            obj = [None]
            dist.recv_object_list(obj, src=1)
            yield obj[0]


    def extract_features_process(self):
        while True:
            # this will wait for a valid shard path
            local_shard_path = self.get_local_shard_path(to_train)

            dataset = (
                wds.WebDataset(local_shard_path, shardshuffle=True, nodesplitter=None, workersplitter=None)
                .shuffle(1000)
                .decode(self.pt_decoder)
            )
            
            for elem in dataset:
                self.process_element(elem)
                
                # Check for valid buckets before synchronization
                for r in list(self.buckets.keys()):
                    if len(self.buckets[r]) >= self.batch_size:
                        closest_ratio = r
                        batch = ([self.buckets[closest_ratio][i][0] for i in range(self.batch_size)],
                                [self.buckets[closest_ratio][i][1] for i in range(self.batch_size)])
                        
                        # now we must extract the text embeddings and vae features, and making sure we don't exceed the vae max batch size
                        # as extracting features tends to use more VRAM than actually training the model
                        # we freeze the vae and text encoder model
                        ratio = self.get_ratio_from_key(closest_ratio)
                        vae_features, embeddings, repa_features = self.extract_features(batch, ratio, low_vram=True)

                        batch = Batch()
                        batch.ratio = ratio
                        batch.embeddings = embeddings
                        batch.vae_features = vae_features
                        if self.use_repa:
                            batch.repa_features = repa_features

                        # send those to gpu0
                        dist.send_object_list([batch], 0)
                        
                        # Clean up after yielding
                        self.buckets[closest_ratio].clear()
                        break
            self.remove_shard(to_remove, local_shard_path)
    
    if self.accelerator.process_index == 0:
        yield from features_yielding_process(self)

    else:
        # dont yield anything
        extract_features_process(self)
