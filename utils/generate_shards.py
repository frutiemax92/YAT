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

def generate_shards(params : TrainingParameters, 
                    local_temp_dir = 'temp',
                    max_pending_uploads=4):   # limit the queue length
    os.makedirs(local_temp_dir, exist_ok=True)
    current_shard = 0
    current_element = 0
    shard_size = int(params.r2_upload_shard_size)
    upload_key = params.r2_upload_key

    if params.huggingface_dataset_repo is not None:
        files = list_repo_files(params.huggingface_dataset_repo, repo_type="dataset")
        files = [f for f in files if f.endswith((".tar", ".zip", ".txt", ".jpg"))]
        urls = [hf_hub_url(params.huggingface_dataset_repo, filename, repo_type='dataset') for filename in files]
    else:
        urls = params.r2_tar_files

    executor = ThreadPoolExecutor(max_workers=2)
    upload_futures = []

    shard_template = f'{local_temp_dir}/shard-%06d.tar'
    shard_filename = f"shard-{current_shard:06d}.tar"
    local_path = os.path.join(local_temp_dir, shard_filename)
    remote_key = f'{upload_key}/{shard_filename}'
    writer = wds.ShardWriter(shard_template, maxcount=shard_size, maxsize=100e9)

    for url in tqdm(urls, desc='iterating through urls'):
        if params.huggingface_dataset_repo is None:
            next_url = get_secured_urls(params.r2_access_key, params.r2_secret_key, params.r2_endpoint, params.r2_bucket_name, [url])
        else:
            next_url = url

        session = boto3.Session(
            aws_access_key_id=params.r2_access_key,
            aws_secret_access_key=params.r2_secret_key
        )
        config = Config(signature_version='s3v4')
        s3_client = session.client('s3', endpoint_url=params.r2_endpoint, config=config)

        def upload_and_cleanup(path, bucket, key):
            s3_client.upload_file(path, bucket, key)
            os.remove(path)

        dataset = wds.WebDataset(next_url).decode()
        for elem in tqdm(dataset, desc='iterating through samples'):
            new_key = f'{current_element:06d}'
            try:
                new_dict = {'__key__': new_key, 'jpg': elem['jpg'], 'txt': elem['txt']}
            except Exception:
                continue
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
