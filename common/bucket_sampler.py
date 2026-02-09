from tqdm import tqdm
from torch.utils.data import BatchSampler, IterableDataset
import random
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
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def flush():
    gc.collect()
    torch.cuda.empty_cache()

class Batch:
    def __init__(self):
        self.embeddings = None
        self.vae_features = None
        self.repa_features = None
        self.ratio = None

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
                 use_repa : bool = False,
                 local_temp_dir : str = 'temp',
                 local_paths: list[str] = None  
                 ):
        self.process_index = accelerator.process_index
        self.num_processes = accelerator.num_processes
        self.device = accelerator.device
        self.accelerator = accelerator
        self.shards = shards
        self.buckets = {}
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

        def local_file_getter(self, process_index : int, to_train : mp.Queue, to_remove : mp.Queue):
            print('using local file getter')
            
            local_path = random.choice(self.local_paths)
            while os.path.exists(local_path) == False:
                local_path = random.choice(self.local_paths)
            to_train.put(local_path)

            # the simplest solution is to wait for to_remove
            r = to_remove.get()

        def download_shard_worker(self, process_index : int, to_train : mp.Queue, to_remove : mp.Queue):
            current_item = 0
            local_shard_paths = []
            while True:
                current_shard_index = self.get_next_shard_index()
                dataset_url = get_secured_urls(self.r2_access_key,
                                    self.r2_secret_key,
                                    self.r2_endpoint,
                                    self.r2_bucket_name,
                                    [self.features_path + '/' + self.shards[current_shard_index]]
                                    )[0]
                if len(local_shard_paths) < 10:
                    local_shard_path = self.local_temp_dir + f'/shard_{process_index}_{current_item}.tar'
                    try:
                        download_tar(dataset_url, local_shard_path)
                    except Exception as error:
                        print(error)
                        current_shard_index = self.get_next_shard_index()
                        continue
                    local_shard_paths.append(local_shard_path)
                else:
                    # this will wait here
                    r = to_remove.get()

                    # strangely, this can happen?!
                    if r in local_shard_paths:
                        local_shard_paths.remove(r)

                        # cover the case for duplicates
                        if r in local_shard_paths == False:
                            self.cleanup_shard(r)
                        local_shard_path = local_shard_paths[-1]
                
                to_train.put(local_shard_path)
                current_item = current_item + 1
        
        if self.local_paths == None:
            self.download_shard_proc = download_shard_worker
        else:
            self.download_shard_proc = local_file_getter
        
        os.makedirs(self.local_temp_dir, exist_ok=True)

        if use_repa:
            self.repa_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
            self.repa_model = AutoModel.from_pretrained('facebook/dinov2-base').to(torch.bfloat16)
            self.repa_model.to(self.accelerator.device)

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
        error = 1000
        res = -1
        for r in self.buckets.keys():
            new_error = abs(ratio - r)
            if new_error < error:
                error = new_error
                res = r
        return res
    
    def __iter__(self):
        sync_counter = 0
        random.seed(self.process_index + self.seed)
        
        # start the download process
        to_train = mp.Queue()
        to_remove = mp.Queue()
        p = mp.Process(target=self.download_shard_proc, args=(self, self.process_index, to_train, to_remove))
        p.start()
        while True:
            # this will wait for a valid shard path
            local_shard_path = to_train.get()

            dataset = (
                wds.WebDataset(local_shard_path, shardshuffle=True, nodesplitter=None, workersplitter=None)
                .shuffle(1000)
                .decode(self.pt_decoder)
            )
            
            self.accelerator.wait_for_everyone()
            for elem in dataset:
                self.process_element(elem)
                
                sync_counter += 1
                if sync_counter % (self.batch_size * 2) != 0:
                    continue
                
                # Check for valid buckets before synchronization
                for r in list(self.buckets.keys()):
                    if len(self.buckets[r]) >= self.batch_size:
                        self.valid_buckets.add(r)
                
                bucket_list = list(self.valid_buckets)
                max_buckets = 100

                # Convert to list of [is_reg_pass, ratio]
                bucket_arr = [
                    [float(r[0]), float(r[1])] if isinstance(r, tuple) else float(r)
                    for r in bucket_list
                ]

                # Pad to fixed size
                bucket_arr = bucket_arr[:max_buckets] + self.get_invalid_bucket() * (max_buckets - len(bucket_arr))
                bucket_tensor = torch.tensor(bucket_arr, device=self.device, dtype=torch.float32)
                
                #try:
                # Gather valid buckets across processes
                self.accelerator.wait_for_everyone()
                dist_valid_buckets = self.accelerator.gather(bucket_tensor)
                ratio_counts = self.get_unique_ratios(dist_valid_buckets)
                
                for ratio, count in ratio_counts.items():
                    if count >= self.num_processes:
                        closest_ratio = self.find_closest_key(ratio)
                        batch = ([self.buckets[closest_ratio][i][0] for i in range(self.batch_size)],
                                [self.buckets[closest_ratio][i][1] for i in range(self.batch_size)])
                        
                        # now we must extract the text embeddings and vae features, and making sure we don't exceed the vae max batch size
                        # as extracting features tends to use more VRAM than actually training the model
                        # we freeze the vae and text encoder model
                        ratio = self.get_ratio_from_key(closest_ratio)
                        vae_features, embeddings, repa_features = self.extract_features(batch, ratio)

                        batch = Batch()
                        batch.ratio = ratio
                        batch.embeddings = embeddings
                        batch.vae_features = vae_features
                        if self.use_repa:
                            batch.repa_features = repa_features

                        yield batch
                        
                        # Clean up after yielding
                        self.buckets[closest_ratio].clear()
                        self.valid_buckets.remove(closest_ratio)
                        self.accelerator.wait_for_everyone()
                        break
                            
                #except Exception as e:
                    #print(f"Process {self.process_index} gather error: {e}")
                    #continue
            to_remove.put(local_shard_path)

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
        if 'jpg' in elem.keys():
            img = elem['jpg']
        elif 'jpeg' in elem.keys():
            img = elem['jpeg']
        else:
            img = elem['png']

        caption = elem['txt']
        ratio = img.height / img.width
        ratio = float(ratio)
        ratio = self.find_closest_ratio(ratio)
        if not ratio in self.buckets.keys():
            self.add_bucket(ratio)

        self.buckets[ratio].append((img, caption))
    
    def extract_features(self, batch, ratio):
        # now we must extract the text embeddings and vae features, and making sure we don't exceed the vae max batch size
        # as extracting features tends to use more VRAM than actually training the model
        # we freeze the vae and text encoder model
        vae_features = []
        embeddings = []
        repa_features = []

        transform = self.get_transform(ratio)
        with torch.no_grad():
            flush()
            for i in range(0, len(batch[0]), self.vae_max_batch_size):
                end_index = min(len(batch[0]), i+self.vae_max_batch_size)
                images = batch[0][i:end_index]
                images = torch.stack([transform(x) for x in images]).to(dtype=torch.bfloat16, device=self.accelerator.device)
                features = self.model.extract_latents(images)
                vae_features.extend(features)
            
            flush()
            for i in range(0, len(batch[1]), self.text_encoder_max_batch_size):
                end_index = min(len(batch[1]), i+self.text_encoder_max_batch_size)
                captions = batch[1][i:i+self.text_encoder_max_batch_size]
                embeds = self.model.extract_embeddings(captions)
                embeddings.extend(embeds)
        
            # check if we need to do repa
            if self.use_repa:
                images = batch[0]
                inputs = self.repa_processor(images=images, return_tensors="pt")
                inputs = inputs['pixel_values'].to(self.accelerator.device, dtype=torch.bfloat16)
                outputs = self.repa_model(inputs)

                repa_features = outputs.pooler_output
        return torch.stack(vae_features), embeddings, repa_features
    
    def get_transform(self, bucket):
        aspect_ratio = self.model.aspect_ratios[str(bucket)]
        target_height = int(aspect_ratio[0])
        target_width = int(aspect_ratio[1])
        return transforms.Compose([
            transforms.Resize((target_height, target_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

class BucketSamplerDreambooth(BucketSamplerExtractFeatures):
    def __init__(self,
                 dreambooth_dataset_folder : str,
                 dreambooth_regularization_folder : str,
                 dreambooth_instance : str,
                 dreambooth_class : str,
                 dreambooth_num_repeats : int,
                 dreambooth_lambda : float,
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
                 dreambooth_num_regularisation_passes = 1,
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
            local_temp_dir
        )
        self.dreambooth_dataset_folder = dreambooth_dataset_folder
        self.dreambooth_regularization_folder = dreambooth_regularization_folder
        self.dreambooth_instance = dreambooth_instance
        self.dreambooth_class = dreambooth_class
        self.dreambooth_lambda = dreambooth_lambda
        self.dreambooth_num_repeats = dreambooth_num_repeats
        self.dreambooth_num_regularisation_passes = dreambooth_num_regularisation_passes
        self.reg_shard = False

        # this is a hack to reuse the existing code!
        def download_shard_worker(self, q : mp.Queue, num_tars = mp.Value):
            current_item = 0
            while True:
                with num_tars.get_lock():
                    n = num_tars.value
                if n >= self.cache_size:
                    time.sleep(1.0)
                    continue

                if current_item % 2 == 0:
                    for r in range(self.dreambooth_num_repeats):
                        q.put((False, self.dreambooth_dataset_folder))
                        with num_tars.get_lock():
                            num_tars.value = num_tars.value + 1
                else:
                    # we get the regularization images either from a local folder or a bucket on the cloud
                    for r in range(self.dreambooth_num_regularisation_passes):
                        if self.r2_bucket_name == None:
                            q.put((True, self.dreambooth_regularization_folder))
                        else:
                            current_shard_index = self.get_next_shard_index()
                            dataset_url = get_secured_urls(self.r2_access_key,
                                        self.r2_secret_key,
                                        self.r2_endpoint,
                                        self.r2_bucket_name,
                                        [self.features_path + '/' + self.shards[current_shard_index]]
                                        )[0]
                            local_shard_path = self.local_temp_dir + f'/shard_{self.process_index}_{current_item}.tar'
                            try:
                                download_tar(dataset_url, local_shard_path)
                            except:
                                current_shard_index = self.get_next_shard_index()
                                continue
                            q.put((True, local_shard_path))
                        with num_tars.get_lock():
                            num_tars.value = num_tars.value + 1
                current_item = current_item + 1
        self.download_shard_proc = download_shard_worker
    
    def cleanup_shard(self, local_shard_path):
        pass

    def get_local_shard_path(self, q : mp.Queue):
        is_reg, path = q.get()
        self.reg_shard = is_reg
        return path

    def get_invalid_bucket(self):
        return [(-1.0, -1.0)]

    def process_element(self, elem):
        if 'jpg' in elem.keys():
            img = elem['jpg']
        elif 'jpeg' in elem.keys():
            img = elem['jpeg']
        else:
            img = elem['png']

        if 'txt' in elem.keys():
            caption = elem['txt']
        elif self.reg_shard == True:
            caption = self.dreambooth_class
        else:
            caption = self.dreambooth_instance + ' ' + self.dreambooth_class
        
        ratio = img.height / img.width
        ratio = float(ratio)
        ratio = self.find_closest_ratio(ratio)

        ratio = (float(self.reg_shard), ratio)
        if not ratio in self.buckets.keys():
            self.add_bucket(ratio)
        self.buckets[ratio].append((img, caption))

    def get_ratio_from_key(self, key):
        if bool(key[0]) == True:
            new_lr = self.model.params.learning_rate * self.dreambooth_lambda
        else:
            new_lr = self.model.params.learning_rate
        for param_group in self.model.optimizer.param_groups:
            param_group["lr"] = new_lr
        return key[1]

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
