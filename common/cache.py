from tqdm import tqdm
import torch
import os
import gzip
import json

class CacheFeaturesCompute:
    def __init__(self, save_to_disk=True, save_img=False):
        self.save_to_disk = save_to_disk
    
    def advance_iterator(self, trainer):
        num_processes = trainer.accelerator.num_processes
        dataset_url, filename, img, caption = next(trainer.data_extractor_iter)

    def run(self, trainer):
        trainer.model = trainer.model.cpu()
        cache_idx = 0

        total = trainer.params.cache_size * trainer.accelerator.num_processes
        pbar = tqdm(total=trainer.params.cache_size * trainer.accelerator.num_processes, desc='Extracting latents and captions')
        while cache_idx < total:
            dataset_url, filename, img, caption = next(trainer.data_extractor_iter)
            if (cache_idx % trainer.accelerator.num_processes) == trainer.accelerator.process_index:
                dataset_url = dataset_url[0]
                filename = filename[0]
                features_path = f'datasets/{dataset_url}/{filename}.npy'
    
                if os.path.exists(features_path) == False:
                    to_save = trainer.cache_latents_embeddings(img, caption[0], cache_idx)
    
                    if self.save_to_disk:
                        with gzip.open(features_path, 'wb') as f:
                            torch.save(to_save, f)
                else:
                    try:
                        with gzip.open(features_path, 'rb') as f:
                            to_save = torch.load(f)
                    except:
                        # in case of a corrupted file, just repeat
                        to_save = trainer.cache_latents_embeddings(img, caption[0], cache_idx)
    
                        if self.save_to_disk:
                            with gzip.open(features_path, 'wb') as f:
                                torch.save(to_save, f)
                torch.save(to_save, f'cache/{cache_idx}.npy')
            cache_idx = cache_idx + 1
            pbar.update(1)

class CacheLoadFeatures:
    def __init__(self):
        pass
    
    def run(self, trainer):
        cache_idx = 0

        # swap the model to the cpu while caching
        trainer.model = trainer.model.cpu()

        if trainer.accelerator.is_main_process:
            it = range(trainer.params.cache_size * trainer.accelerator.num_processes)
        else:
            it = range(trainer.accelerator.process_index,
                                            trainer.params.cache_size * trainer.accelerator.num_processes,
                                            trainer.accelerator.num_processes)
        
        if trainer.accelerator.is_main_process:
            for cache_idx in tqdm.tqdm(it, desc='Extracting latents and captions'):
                item = next(trainer.data_extractor_iter)
                ratio, latent, embedding = item[0]
                embedding = torch.stack(embedding)
                embedding = torch.swapaxes(embedding, 0, 1)
                embeddings = torch.zeros((1, 300, 2304))
                non_zeros = embedding.shape[1]
                embeddings[0, :non_zeros] = embedding
                embeddings = torch.squeeze(embeddings)

                embedding_mask = torch.zeros((1, 300))
                embedding_mask[0, :non_zeros] = 1.0
                embedding_mask = torch.squeeze(embedding_mask)
                
                embeddings = embeddings, embedding_mask
                latent = torch.squeeze(latent)
                to_save = (ratio, latent, embeddings)
                torch.save(to_save, f'cache/{cache_idx}.npy')
