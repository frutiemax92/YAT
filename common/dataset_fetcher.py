import webdataset as wds
from common.cloudflare import get_secured_urls
from torchvision import transforms
import torch

class DatasetFetcher:
    def __init__(self, shards : list[str], 
                    r2_access_key,
                    r2_secret_key,
                    r2_endpoint,
                    r2_bucket_name,
                    batch_size,
                    model):
        self.shards = shards
        self.r2_access_key = r2_access_key
        self.r2_secret_key = r2_secret_key
        self.r2_endpoint = r2_endpoint
        self.r2_bucket_name = r2_bucket_name
        self.current_shard_index = 0
        self.num_shards = len(self.shards)
        self.batch_size = batch_size
        self.model = model

        self.queues = {}
    
    def __iter__(self):
        while True:
            # generate a secured url for the shard
            dataset_url = get_secured_urls(self.r2_access_key,
                     self.r2_secret_key,
                     self.r2_endpoint,
                     self.r2_bucket_name,
                     [self.shards[self.current_shard_index]])[0]
            
            # generate a dataset for it
            def assign_bucket(img):
                w, h = img.size
                return self.model.find_closest_ratio(h / w)
            
            def with_bucket(sample):
                image, caption = sample
                bucket = assign_bucket(image)
                return {"jpg": image, "txt": caption, "bucket": bucket}
            
            to_tensor = transforms.ToTensor()
            def bucketed_batcher(data_iter, batch_size):
                to_tensor = transforms.ToTensor() 

                for sample in data_iter:
                    bucket = sample["bucket"]

                    if bucket not in self.queues.keys():
                        self.queues[bucket] = []
                    
                    # here we need to resize the image for the correct size for the model
                    img = sample['jpg']
                    aspect_ratio = self.model.aspect_ratios[sample['bucket']]
                    target_height = int(aspect_ratio[0])
                    target_width = int(aspect_ratio[1])
                    img = img.resize((target_width, target_height))

                    # transform that image to a tensor
                    sample['jpg'] = to_tensor(img)
                    self.queues[bucket].append(sample)

                    if len(self.queues[bucket]) >= batch_size:
                        batch = self.queues[bucket][:batch_size]
                        self.queues[bucket] = []
                        yield batch
            
            dataset = wds.WebDataset(dataset_url, shardshuffle=False,
                                    nodesplitter=None,
                                    workersplitter=None)\
                .shuffle(1000)\
                .decode('pil')\
                .to_tuple('jpg', 'txt')\
                .map(with_bucket)
            
            for sample in dataset:
                bucket = sample["bucket"]

                if bucket not in self.queues.keys():
                    self.queues[bucket] = []
                
                # here we need to resize the image for the correct size for the model
                img = sample['jpg']
                aspect_ratio = self.model.aspect_ratios[sample['bucket']]
                target_height = int(aspect_ratio[0])
                target_width = int(aspect_ratio[1])
                img = img.resize((target_width, target_height))

                # transform that image to a tensor
                sample['jpg'] = to_tensor(img)
                self.queues[bucket].append(sample)

                if len(self.queues[bucket]) >= self.batch_size:
                    batch = self.queues[bucket][:self.batch_size]
                    self.queues[bucket] = []
                    images = torch.stack([x["jpg"] for x in batch])
                    captions = [x["txt"] for x in batch]
                    yield images, captions

            self.current_shard_index = (self.current_shard_index + 1) % self.num_shards