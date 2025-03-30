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
