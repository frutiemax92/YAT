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

def flush():
    gc.collect()
    torch.cuda.empty_cache()

class BucketDataset(IterableDataset):
    def __init__(self,
                 num_epochs, 
                 dataset,
                 batch_size,
                 aspect_ratios,
                 accelerator : Accelerator,
                 pipe : SanaPAGPipeline,
                 discard_low_res=True):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size * accelerator.num_processes
        self.aspect_ratios = aspect_ratios
        self.discard_low_res = discard_low_res
        self.num_epochs = num_epochs
        self.accelerator = accelerator
        self.pipe = pipe
    
    def extract_embeddings(self, captions : list[str]):
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = \
            self.pipe.encode_prompt(captions, do_classifier_free_guidance=False)
        return prompt_embeds, prompt_attention_mask

    def extract_latents(self, images : torch.Tensor):
        image_processor = self.pipe.image_processor
        images = image_processor.pil_to_numpy(images)
        images = torch.tensor(images, device=self.pipe.device, dtype=self.pipe.dtype)
        #images = torch.squeeze(images, dim=0)
        images = torch.swapaxes(images, 1, -1)
        images = image_processor.preprocess(images)
        flush()

        output = self.pipe.vae.encode(images.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)).latent
        return output * self.pipe.vae.config.scaling_factor

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

                try:
                    img = resize_transform(img)
                    if img.height != height_target or img.width != width_target:
                        raise Exception('The resized image does not match the targetted size!')
                        
                except Exception as e:
                    tqdm.write(f'Catched exception while resizing image: {e}')
                    continue

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
                buckets[ratio].append((img, caption))

                # check if the bucket is full
                if len(buckets[ratio]) == self.batch_size:
                    # some safety checks
                    img_width = buckets[ratio][0][0].width
                    img_height = buckets[ratio][0][0].height
                    for bucket in buckets[ratio]:
                        if bucket[0].width != img_width or bucket[0].height != img_height:
                            # bad bucket, throw that one in the garbage!
                            tqdm.write('Caught a bad bucket!')
                            buckets.pop(ratio)
                            continue

                    # transform the PIL image to a tensor
                    for elem in buckets[ratio]:
                        img_tmp, caption_tmp = elem
                        with torch.no_grad():
                            latent = self.extract_latents(img_tmp)
                            embeddings, prompt_attention_mask = self.extract_embeddings(caption_tmp)
                        yield latent, embeddings, prompt_attention_mask
                    buckets.pop(ratio)
    
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