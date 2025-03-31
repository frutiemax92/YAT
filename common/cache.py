import tqdm
import torch

class CacheFeaturesCompute:
    def __init__(self):
        pass
    
    def run(self, trainer):
        cache_idx = 0

        # swap the model to the cpu while caching
        if trainer.params.low_vram:
            trainer.model = trainer.model.cpu()

        if trainer.accelerator.is_main_process:
            it = range(trainer.params.cache_size * trainer.accelerator.num_processes)
        else:
            it = range(trainer.accelerator.process_index,
                                            trainer.params.cache_size * trainer.accelerator.num_processes,
                                            trainer.accelerator.num_processes)
        for cache_idx in tqdm.tqdm(it, desc='Extracting latents and captions'):
            if trainer.accelerator.is_main_process:
                img, caption = next(trainer.data_extractor_iter)

                if cache_idx % trainer.accelerator.num_processes != 0:
                    torch.save(img[0], f'cache/{cache_idx}.mpy')
                    with open(f'cache/{cache_idx}.txt', 'w') as f:
                        f.write(caption[0])
                else:
                    trainer.cache_latents_embeddings(img, caption[0], cache_idx)
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
                trainer.cache_latents_embeddings(img, caption, cache_idx)

class CacheLoadFeatures:
    def __init__(self):
        pass
    
    def run(self, trainer):
        cache_idx = 0

        # swap the model to the cpu while caching
        if trainer.params.low_vram:
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
