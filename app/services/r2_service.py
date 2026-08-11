"""Cloudflare R2 (S3-compatible) service."""

import uuid
from io import BytesIO

import boto3
from botocore.config import Config

from app.core.config import get_settings

settings = get_settings()


def _get_r2_client():
    """Build a boto3 S3 client pointed at Cloudflare R2."""
    endpoint_url = (
        f"https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_image_to_r2(
    file_bytes: bytes,
    original_filename: str,
    content_type: str,
    folder: str = "images",
) -> tuple[str, str]:
    """
    Upload raw bytes to R2 bucket.

    Returns
    -------
    (r2_key, public_url)
        r2_key   – the object key stored in the bucket
        public_url – the publicly accessible URL
    """
    ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else "bin"
    object_key = f"{folder}/{uuid.uuid4()}.{ext}"

    client = _get_r2_client()
    client.put_object(
        Bucket=settings.R2_BUCKET_NAME,
        Key=object_key,
        Body=file_bytes,
        ContentType=content_type,
    )

    public_url = f"{settings.R2_PUBLIC_URL.rstrip('/')}/{object_key}"
    return object_key, public_url
