from common.training_parameters_reader import TrainingParameters
import argparse
from common.cloudflare import get_secured_urls
import boto3
from botocore.config import Config
import webdataset as wds
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from huggingface_hub import list_repo_files, hf_hub_download, hf_hub_url
import time
from PIL import Image
import io

def generate_shards(params : TrainingParameters, 
                    local_temp_dir = 'temp',
                    max_pending_uploads=4):   # limit the queue length
    
    caption_repo = 'drawthingsai/megalith-10m-sharecap'
    img_repo = 'drawthingsai/megalith-10m'

    os.makedirs(local_temp_dir, exist_ok=True)
    current_shard = 0
    current_element = 0
    shard_size = int(params.r2_upload_shard_size)
    upload_key = params.r2_upload_key

    caption_files = list_repo_files(caption_repo, repo_type="dataset")
    caption_files = [f for f in caption_files if f.endswith((".tar", ".zip", ".txt", ".jpg"))]
    caption_urls = [hf_hub_url(caption_repo, filename, repo_type='dataset') for filename in caption_files]

    img_files = list_repo_files(img_repo, repo_type="dataset")
    img_files = [f for f in img_files if f.endswith((".tar", ".zip", ".txt", ".jpg"))]
    img_urls = [hf_hub_url(img_repo, filename, repo_type='dataset') for filename in img_files]

    executor = ThreadPoolExecutor(max_workers=2)
    upload_futures = []

    shard_template = f'{local_temp_dir}/shard-%06d.tar'
    shard_filename = f"shard-{current_shard:06d}.tar"
    local_path = os.path.join(local_temp_dir, shard_filename)
    remote_key = f'{upload_key}/{shard_filename}'
    writer = wds.ShardWriter(shard_template, maxcount=shard_size, maxsize=100e9)

    for idx in tqdm(range(len(img_urls)), desc='iterating through urls'):
        session = boto3.Session(
            aws_access_key_id=params.r2_access_key,
            aws_secret_access_key=params.r2_secret_key
        )
        config = Config(signature_version='s3v4')
        s3_client = session.client('s3', endpoint_url=params.r2_endpoint, config=config)

        def upload_and_cleanup(path, bucket, key):
            s3_client.upload_file(path, bucket, key)
            os.remove(path)

        caption_url = caption_urls[idx]

        # we need to get all the images captions first, with the key associated
        captions = {}
        caption_dataset = wds.WebDataset(caption_url).decode()

        for elem in tqdm(caption_dataset, desc='iterating through captions'):
            json_content = elem['json']
            caption = json_content['sharecap_caption']
            key = json_content['key']
            captions[key] = caption
        
        # from there, we can iterate through the image shard
        img_url = img_urls[idx]
        img_dataset = wds.WebDataset(img_url).decode()
        for elem in tqdm(img_dataset, desc='iterating through images'):
            try:
                new_key = elem['__key__']
                caption = captions[new_key]
                img = elem['jpg']
            except:
                print('skipping bad element!')
                continue
            
            # testing
            #image = Image.open(io.BytesIO(img))
            #image.save('test.jpg')

            new_dict = {'__key__': new_key, 'jpg': img, 'txt': caption}
            writer.write(new_dict)
            current_element += 1

            if current_element >= shard_size:
                writer.next_stream()

                # schedule upload
                future = executor.submit(
                    upload_and_cleanup,
                    local_path,
                    params.r2_bucket_name,
                    remote_key
                )
                upload_futures.append(future)

                # stall if too many uploads are pending
                while len(upload_futures) >= max_pending_uploads:
                    # wait for at least one upload to finish
                    done, not_done = [], []
                    for f in upload_futures:
                        if f.done():
                            done.append(f)
                        else:
                            not_done.append(f)
                    for f in done:
                        f.result()  # raise exceptions if any
                    upload_futures = not_done
                    if len(upload_futures) >= max_pending_uploads:
                        time.sleep(1)  # avoid busy spin

                # reset counters for new shard
                current_shard += 1
                current_element = 0
                shard_template = f'{local_temp_dir}/shard-%06d.tar'
                shard_filename = f"shard-{current_shard:06d}.tar"
                local_path = os.path.join(local_temp_dir, shard_filename)
                remote_key = f'{upload_key}/{shard_filename}'

    # wait for remaining uploads
    for f in upload_futures:
        f.result()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)
    generate_shards(params)
