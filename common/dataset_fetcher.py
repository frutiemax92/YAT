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
                transforms.Resize((target_height, target_width))
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

            def with_bucket(sample):
                image, caption = sample
                bucket = assign_bucket(image)
                return {"jpg": image, "txt": caption, "bucket": bucket}



            dataset = (
                wds.WebDataset(dataset_url, shardshuffle=False, nodesplitter=None, workersplitter=None)
                .shuffle(1000)
                .decode('pil')
                .to_tuple('jpg', 'txt')
                .map(with_bucket)
            )
            loader = wds.WebLoader(dataset, batch_size=None, num_workers=4)

            to_tensor = transforms.Compose([
                transforms.ToTensor()
            ])
            for sample in loader:
                bucket = sample["bucket"]
                if bucket not in self.queues:
                    self.queues[bucket] = []

                # Apply transform
                sample["jpg"] = to_tensor(sample["jpg"])
                self.queues[bucket].append(sample)

                if len(self.queues[bucket]) >= self.batch_size:
                    batch = self.queues[bucket][:self.batch_size]
                    self.queues[bucket] = []
                    
                    transform = get_transform(bucket)
                    images = torch.stack([transform(x["jpg"]) for x in batch])
                    captions = [x["txt"] for x in batch]

                    # Feature extraction (example)
                    # features = self.model.extract_features(images)
                    # yield features, captions

                    yield images, captions, bucket

            self.current_shard_index = (self.current_shard_index + 1) % self.num_shards