import tqdm
import torch
import os
import gzip

class CacheFeaturesCompute:
    def __init__(self, save_to_disk=True):
        self.save_to_disk = save_to_disk
    
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
        for cache_idx in tqdm.tqdm(it, desc='Extracting latents and captions'):
            if trainer.accelerator.is_main_process:
                dataset_url, filename, img, caption = next(trainer.data_extractor_iter)
                dataset_url = dataset_url[0]
                filename = filename[0]
                features_path = f'datasets/{dataset_url}/{filename}.npy'

                if cache_idx % trainer.accelerator.num_processes != 0:
                    torch.save(img[0], f'cache/{cache_idx}.mpy')
                    with open(f'cache/{cache_idx}.txt', 'w') as f:
                        f.write(caption[0])
                    
                    # also save the filename associated with the cache_idx in case we save to disk
                    if self.save_to_disk:
                        with open(f'cache/{cache_idx}', 'w') as f:
                            f.write(f'datasets/{dataset_url}/{filename}.npy')
                else:
                    if os.path.exists(features_path) == False:
                        to_save = trainer.cache_latents_embeddings(img, caption[0], cache_idx)

                        if self.save_to_disk:
                            with gzip.open(features_path, 'wb') as f:
                                torch.save(to_save, f)
                    else:
                        with gzip.open(features_path, 'rb') as f:
                            to_save = torch.load(f)
                        torch.save(to_save, f'cache/{cache_idx}.npy')
            else:
                # try to read the image and caption when they're available
                while True:
                    try:
                        img = torch.load(f'cache/{cache_idx}.mpy')
                        with open(f'cache/{cache_idx}.txt') as f:
                            caption = f.read()
                        break
                    except:
                        continue
            
                # start with the caching
                # read the filename first
                cache_file_name = f'cache/{cache_idx}'
                found_features = False
                if os.path.exists(cache_file_name):
                    with open(f'cache/{cache_idx}', 'r') as f:
                        features_path = f.read()
                    if os.path.exists(features_path):
                        found_features = True

                if found_features == False:
                    to_save = trainer.cache_latents_embeddings(img, caption, cache_idx)
                    if self.save_to_disk:
                        with gzip.open(features_path, 'wb') as f:
                            torch.save(to_save, f)
                else:
                    with gzip.open(features_path, 'rb') as f:
                        to_save = torch.load(f)
                    torch.save(to_save, f'cache/{cache_idx}.npy')

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
