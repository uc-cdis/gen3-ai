"""Storage helper utilities for the Gen3 AI model repo service."""

from gen3_ai_model_repo.config import (
    LOCAL_STORAGE_PATH,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    S3_ACCESS_KEY_ID,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
    STORAGE_CREATE_BUCKET_IF_MISSING,
    STORAGE_PROVIDER,
)
from gen3_ai_model_repo.storage.local import LocalStorageProvider
from gen3_ai_model_repo.storage.minio import MinioStorageProvider

_provider_cache = None


def get_storage_provider():
    """Return the configured storage provider implementation."""
    global _provider_cache

    if _provider_cache is not None:
        return _provider_cache

    if STORAGE_PROVIDER == "minio":
        _provider_cache = MinioStorageProvider(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket_name=MINIO_BUCKET,
            secure=MINIO_SECURE,
            create_bucket_if_missing=STORAGE_CREATE_BUCKET_IF_MISSING,
        )
        return _provider_cache

    if STORAGE_PROVIDER == "s3":
        from gen3_ai_model_repo.storage.s3 import S3StorageProvider

        _provider_cache = S3StorageProvider(
            bucket_name=S3_BUCKET,
            region=S3_REGION,
            endpoint_url=S3_ENDPOINT_URL,
            access_key_id=S3_ACCESS_KEY_ID,
            secret_access_key=S3_SECRET_ACCESS_KEY,
            create_bucket_if_missing=STORAGE_CREATE_BUCKET_IF_MISSING,
        )
        return _provider_cache

    if STORAGE_PROVIDER != "local":
        raise ValueError(f"Unsupported STORAGE_PROVIDER: {STORAGE_PROVIDER}")

    _provider_cache = LocalStorageProvider(LOCAL_STORAGE_PATH)
    return _provider_cache


__all__ = ["get_storage_provider"]
