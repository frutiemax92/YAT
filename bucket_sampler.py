import multiprocessing
from tqdm import tqdm
from torch.utils.data import BatchSampler, IterableDataset
import random
from copy import copy
from torchvision.transforms import Resize, Compose, RandomCrop, RandomHorizontalFlip, CenterCrop, PILToTensor
import torch

def find_closest_ratio(idx, img, aspect_ratios):
    width = img.width
    height = img.height
    ratio = height / width

    # find the closest ratio from the aspect ratios table
    min_distance = 100
    target_ratio = 0.6
    for r in aspect_ratios.keys():
        distance = abs(float(r) - ratio)
        if min_distance > distance:
            target_ratio = r
            min_distance = distance
    
    # Return the result as a tuple (target_ratio, idx)
    return str(target_ratio), idx

def get_bucket_indices(dataset, aspect_ratios, batch_size):
    buckets = {}
    idx = 0
    for img, caption in tqdm(dataset, desc='Calculating buckets...'):
        ratio, idx = find_closest_ratio(idx, img, aspect_ratios)
        if ratio not in buckets.keys():
            buckets[ratio] = []
        buckets[ratio].append(idx)
        idx = idx + 1

    # create batches out of that
    indices = []
    batch = []
    for key, value in tqdm(buckets.items(), desc='Calculating reordered dataset'):
        value_copy = copy(value)
        random.shuffle(value_copy)

        for idx in value_copy:
            batch.append(idx)
            indices.append(idx)

            if len(batch) == batch_size:
                batch = []
        if batch != []:
            while len(batch) != batch_size:
                idx = random.choice(value_copy)
                batch.append(idx)
                indices.append(idx)
            batch = []
    return indices

class BucketSampler(BatchSampler):
    def __init__(self,
                 dataset,
                 aspect_ratios,
                 num_processes,
                 batch_size):
        random.seed(0)

        self.buckets = {}
        self.aspect_ratios = aspect_ratios
        self.batch_size = batch_size

        idx = 0
        for img, caption in tqdm(dataset, desc='Calculating buckets...'):
            ratio, idx = find_closest_ratio(idx, img, self.aspect_ratios)
            if ratio not in self.buckets.keys():
                self.buckets[ratio] = []
            self.buckets[ratio].append(idx)
            idx = idx + 1 
    
    def __iter__(self):
        batch = []
        for key, value in self.buckets.items():
            value_copy = copy(value)
            random.shuffle(value_copy)

            for idx in value_copy:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if batch != []:
                while len(batch) != self.batch_size:
                    idx = random.choice(value_copy)
                    batch.append(idx)
                yield batch
                batch = []
    
    def __len__(self):
        return int(self.num_images // self.batch_size)

class BatchCollater:
    def __init__(self, aspect_ratios):
        self.aspect_ratios = aspect_ratios

    def __call__(self, *args, **kwds):
        samples = args[0]
        images = []
        captions = []

        # choose the closest ratio from the first image
        img, caption = samples[0]
        closest_ratio, idx = find_closest_ratio(0, img, self.aspect_ratios)
        height_target = int(self.aspect_ratios[closest_ratio][0])
        width_target = int(self.aspect_ratios[closest_ratio][1])

        # define a transformation so we get all images with the size of the first image in the batch
        crop_transform = CenterCrop((height_target, width_target))
        pil_to_tensor = PILToTensor()

        for sample in samples:
            img, caption = sample
            captions.append(caption)

            # this is bad if the image is low resolution
                    # resize the image so it fits best the target
            width = img.width
            height = img.height

            if width > height:
                ratio = width_target / width
                new_height = height * ratio
                new_width = width_target
            else:
                ratio = height_target / height
                new_width = width * ratio
                new_height = height_target
            transform = Resize((int(new_height), int(new_width)))

            img = transform(img)
            img = crop_transform(img)
            img = pil_to_tensor(img)
            images.append(img)
        
        images = torch.stack(images)
        return images, captions

class BucketDataset(IterableDataset):
    def __init__(self, 
                 dataset,
                 batch_size,
                 aspect_ratios,
                 discard_low_res=True):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.discard_low_res = discard_low_res
    
    def find_closest_ratio(self, img):
        width = img.width
        height = img.height
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
        # those are the left over images that didn't fit in any batch in the last epoch
        left_overs = []
        pil_to_tensor = PILToTensor()
        # finally the buckets
        buckets = {}
        while True:
            discarded_images = 0
            for img, caption in self.dataset:
                # calculate the closest aspect ratio for the image
                ratio = self.find_closest_ratio(img)
                height_target = int(self.aspect_ratios[ratio][0])
                width_target = int(self.aspect_ratios[ratio][1])
                width = img.width
                height = img.height

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

                    # also push a left over from the last epoch if any
                    if len(left_overs) != 0:
                        left_over = left_overs.pop()
                        img, caption = left_over
                        crop_transform = CenterCrop((height_target, width_target))
                        img = crop_transform(img)
                        buckets[ratio].append((img, caption))

                # handle the case of batch size = 1???
                if self.batch_size != 1:
                    buckets[ratio].append((img, caption))

                # check if any bucket is full
                keys_to_remove = []
                for key, bucket in buckets.items():
                    if len(bucket) == self.batch_size:
                        # transform the PIL image to a tensor
                        images = []
                        captions = []
                        for elem in bucket:
                            img, caption = elem
                            images.append(pil_to_tensor(img))
                            captions.append(caption)
                        yield torch.stack(images), captions
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    buckets.pop(key)
    
            # check for left overs
            for key, bucket in enumerate(buckets):
                if len(bucket) != 0:
                    left_overs.extend(bucket)
            
            # print some statistics
            if self.discard_low_res:
                tqdm.write(f'There were {discarded_images} low resolution images in this epoch')
            tqdm.write(f'There are {len(left_overs)} images from this epoch that did not fit in any bucket')

            # finally reinitialize the buckets
            buckets = {}
            discarded_images = 0