import boto3
from botocore.config import Config
import requests
from tqdm import tqdm
import time

def get_client(r2_access_key, r2_secret_key, r2_endpoint):
    session = boto3.Session(
    aws_access_key_id=r2_access_key,
    aws_secret_access_key=r2_secret_key
)
    config = Config(signature_version='s3v4')
    s3_client = session.client('s3', endpoint_url=r2_endpoint, config=config)
    return s3_client

def get_secured_urls(r2_access_key : str,
                     r2_secret_key : str,
                     r2_endpoint : str,
                     r2_bucket_name : str,
                     r2_tar_files : list[str]):
    # get the urls from the cloudflare bucket with the keys
    session = boto3.Session(
        aws_access_key_id=r2_access_key,
        aws_secret_access_key=r2_secret_key
    )
    config = Config(signature_version='s3v4')
    s3_client = session.client('s3', endpoint_url=r2_endpoint, config=config)

    # urls with 1 week expiration
    return [s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': r2_bucket_name, 'Key':tar_file},
        ExpiresIn=604800
    ) for tar_file in r2_tar_files]

def download_tar(url, local_path, max_total_time=60):
    start_time = time.time()

    # if we don't get any chunk under 60 seconds, raise an exception!
    with requests.get(url, stream=True, timeout=240) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), desc='downloading chunks'):
                f.write(chunk)

                if time.time() - start_time > max_total_time:
                    raise DownloadTimeoutError(
                        f"Total download time exceeded {max_total_time} seconds"
                    )