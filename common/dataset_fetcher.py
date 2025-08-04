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
        # Compose transforms: resize and to tensor
        def get_transform(bucket):
            aspect_ratio = self.model.aspect_ratios[bucket]
            target_height = int(aspect_ratio[0])
            target_width = int(aspect_ratio[1])
            return transforms.Compose([
                transforms.Resize((target_height, target_width)),
                transforms.ToTensor()
            ])
            
        while True:
            dataset_url = get_secured_urls(
                self.r2_access_key,
                self.r2_secret_key,
                self.r2_endpoint,
                self.r2_bucket_name,
                [self.shards[self.current_shard_index]]
            )[0]

            def assign_bucket(img):
                w, h = img.size
                return self.model.find_closest_ratio(h / w)

            dataset = (
                wds.WebDataset(dataset_url, shardshuffle=False, nodesplitter=None, workersplitter=None)
                .decode('pil')
                .batched(16)
                .to_tuple('jpg', 'txt')
            )
            for batch in dataset:
                images = batch[0]
                captions = batch[1]
                for idx in range(len(images)):
                    img = images[idx]
                    caption = captions[idx]

                    # Apply transform
                    w, h = img.size
                    bucket = self.model.find_closest_ratio(h / w)

                    if bucket not in self.queues:
                        self.queues[bucket] = []
                    self.queues[bucket].append((img, caption))

                    if len(self.queues[bucket]) >= self.batch_size:
                        batch = self.queues[bucket][:self.batch_size]
                        self.queues[bucket] = []
                        
                        transform = get_transform(bucket)
                        batch_images = torch.stack([transform(x[0]) for x in batch])
                        batch_captions = [x[1] for x in batch]

                        # Feature extraction (example)
                        # features = self.model.extract_features(images)
                        # yield features, captions

                        yield batch_images, batch_captions, bucket

            self.current_shard_index = (self.current_shard_index + 1) % self.num_shards