import boto3
from botocore.config import Config

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