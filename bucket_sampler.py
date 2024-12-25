import multiprocessing
from tqdm import tqdm
from torch.utils.data import BatchSampler, IterableDataset
import random
from copy import copy
from torchvision.transforms import Resize, Compose, RandomCrop, RandomHorizontalFlip, CenterCrop, PILToTensor
import torch
from accelerate import Accelerator

class BucketDataset(IterableDataset):
    def __init__(self,
                 num_epochs, 
                 dataset,
                 batch_size,
                 aspect_ratios,
                 accelerator : Accelerator,
                 discard_low_res=True):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.discard_low_res = discard_low_res
        self.num_epochs = num_epochs
        self.accelerator = accelerator
    
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
        for epoch in tqdm(range(self.num_epochs)):
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
                        left_over_img, left_over_caption = left_over
                        crop_transform = CenterCrop((height_target, width_target))
                        left_over_img = crop_transform(left_over_img)
                        buckets[ratio].append((left_over_img, left_over_caption))

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
            for key, bucket in buckets.items():
                if len(bucket) != 0:
                    left_overs.extend(bucket)
            
            # print some statistics
            if self.discard_low_res:
                tqdm.write(f'There were {discarded_images} low resolution images in this epoch')
            tqdm.write(f'There are {len(left_overs)} images from this epoch that did not fit in any bucket')

            # finally reinitialize the buckets and wait for all the processes to come by
            buckets = {}
            discarded_images = 0
            self.accelerator.wait_for_everyone()