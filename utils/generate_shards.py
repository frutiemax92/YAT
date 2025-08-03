from common.training_parameters_reader import TrainingParameters
import argparse
from common.cloudflare import get_secured_urls
import boto3
from botocore.config import Config
import webdataset as wds
from tqdm import tqdm
import os

def generate_shards(params : TrainingParameters, 
                    local_temp_dir = 'temp'):
    # this helper function make sure to create equal shards and discard incomplete shards
    # this is necessary for efficient features extraction when we will use multiple gpus
    os.makedirs(local_temp_dir, exist_ok=True)
    current_shard = 0
    current_element = 0
    shard_size = int(params.r2_upload_shard_size)
    upload_key = params.r2_upload_key
    urls = params.r2_tar_files

    shard_template = f'{local_temp_dir}/shard-%06d.tar'
    shard_filename = f"shard-{current_shard:06d}.tar"
    local_path = os.path.join(local_temp_dir, shard_filename)
    remote_key = f'{upload_key}/{shard_filename}'
    writer = wds.ShardWriter(shard_template, maxcount=shard_size)
    
    for url in tqdm(urls, desc='iterating through urls'):
        next_url = get_secured_urls(params.r2_access_key, params.r2_secret_key, params.r2_endpoint, params.r2_bucket_name, [url])
        session = boto3.Session(
            aws_access_key_id=params.r2_access_key,
            aws_secret_access_key=params.r2_secret_key
        )
        config = Config(signature_version='s3v4')
        s3_client = session.client('s3', endpoint_url=params.r2_endpoint, config=config)

        dataset = wds.WebDataset(next_url).decode()
        for elem in tqdm(dataset, desc='iterating through samples'):
            writer.write(elem)
            current_element = current_element + 1

            if current_element >= shard_size:
                writer.next_stream()
                s3_client.upload_file(local_path, params.r2_bucket_name, remote_key)
                current_shard = current_shard + 1
                current_element = 0

                # delete the file and increase the shard index
                os.remove(local_path)
                shard_template = f'{local_temp_dir}/shard-%06d.tar'
                shard_filename = f"shard-{current_shard:06d}.tar"
                local_path = os.path.join(local_temp_dir, shard_filename)
                remote_key = f'{upload_key}/{shard_filename}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    params = TrainingParameters()
    params.read_yaml(args.config)
    generate_shards(params)

