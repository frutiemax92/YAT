import multiprocessing
from tqdm import tqdm
from torch.utils.data import BatchSampler, IterableDataset
import random
from copy import copy
from torchvision.transforms import Resize, Compose, RandomCrop, RandomHorizontalFlip, CenterCrop, PILToTensor
import torch
from accelerate import Accelerator
from diffusers import SanaPAGPipeline
import gc
from tarfile import ReadError

def flush():
    gc.collect()
    torch.cuda.empty_cache()

class RoundRobinMix(torch.utils.data.IterableDataset):
    def __init__(self, datasets, seed=0, accelerator=None):
        self.datasets = datasets
        self.num_datasets = len(datasets)
        self.curr_dataset = 0
        self.iterators = [iter(dataset) for dataset in self.datasets]
        self.buffer_size = 100
        self.buffer = []
        self.accelerator = accelerator
        random.seed(seed)

    def __iter__(self):
        while True:
            for i in tqdm(range(self.buffer_size), desc='Downloading images'):
                try:
                    item = next(self.iterators[self.curr_dataset])
                except ReadError as e:
                    print('IOError!')
                    continue
                except StopIteration as e:
                    print(f'StopIteration exception!')
                    self.iterators[self.curr_dataset] = iter(self.datasets[self.curr_dataset])
                    item = next(self.iterators[self.curr_dataset])
                
                self.buffer.append(item)
            self.curr_dataset = self.curr_dataset + 1
            if self.curr_dataset >= self.num_datasets:
                random.shuffle(self.buffer)
                if self.accelerator != None:
                    print(f'process_index = {self.accelerator.process_index}')
                for item in self.buffer:
                    yield item
                self.buffer.clear()
                self.curr_dataset = 0

            # try:
            #     yield next(self.iterators[self.curr_dataset])
            # except:
            #     self.iterators[self.curr_dataset] = iter(self.datasets[self.curr_dataset])
            # self.curr_dataset = (self.curr_dataset + 1) % self.num_datasets

class BucketDataset(IterableDataset):
    def __init__(self,
                 dataset,
                 batch_size,
                 aspect_ratios,
                 accelerator : Accelerator,
                 pipe,
                 extract_latents_handler,
                 extract_embeddings_handler,
                 discard_low_res=True):
        super().__init__()
        self.dataset = dataset
        self.total_batch_size = batch_size * accelerator.num_processes
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.discard_low_res = discard_low_res
        self.accelerator = accelerator
        self.pipe = pipe
        self.extract_embeddings_handler = extract_embeddings_handler
        self.extract_latents_handler = extract_latents_handler
    
    def extract_embeddings(self, captions : list[str]):
        return self.extract_embeddings_handler(captions)

    def extract_latents(self, images : torch.Tensor):
        return self.extract_latents_handler(images)

    def find_closest_ratio(self, img):
        width = img.shape[-1]
        height = img.shape[-2]
        ratio = height / width

        # find the closest ratio from the aspect ratios table
        min_distance = 100
        target_ratio = 0.6
        for r in self.aspect_ratios.keys():
            distance = abs(float(r) - ratio)
            if min_distance > distance:
                target_ratio = r
                min_distance = distance
        
        # Return the result as a tuple (target_ratio, idx)
        return str(target_ratio)

    def __iter__(self):
        buckets = {}
        images_count = 0
        while True:
            discarded_images = 0
            for img, caption in self.dataset:
                images_count = images_count + 1
                img = self.pipe.image_processor.pil_to_numpy(img)
                img = torch.tensor(img).to(dtype=self.pipe.dtype)
                img = torch.moveaxis(img, -1, 1)
            
                # calculate the closest aspect ratio for the image
                ratio = self.find_closest_ratio(img)
                height_target = int(self.aspect_ratios[ratio][0])
                width_target = int(self.aspect_ratios[ratio][1])
                width = img.shape[-1]
                height = img.shape[-2]

                # check if we discard this image due to a too low resolution
                if height_target > height and width_target > width and self.discard_low_res:
                    # don't push this image in any bucket
                    discarded_images = discarded_images + 1
                    continue

                resize_transform = Resize((height_target, width_target))
                img = resize_transform(img)

                # find if this bucket already exists
                if not ratio in buckets.keys():
                    buckets[ratio] = []
                buckets[ratio].append((img, caption))

                # check if the bucket is full
                if len(buckets[ratio]) == self.total_batch_size:
                    # transform the PIL image to a tensor
                    images_tmp = []
                    captions_tmp = []
                    for elem in buckets[ratio]:
                        img_tmp, caption_tmp = elem
                        images_tmp.append(img_tmp)
                        captions_tmp.append(caption_tmp)

                        if len(images_tmp) == self.batch_size:
                            with torch.no_grad():
                                latents = self.extract_latents(images_tmp)
                                embeddings, prompt_attention_masks = self.extract_embeddings(captions_tmp)
                            yield latents, embeddings, prompt_attention_masks
                            images_tmp.clear()
                            captions_tmp.clear()
                    buckets.pop(ratio)