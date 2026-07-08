from gen3_ai_model_repo.config import (
    LOCAL_STORAGE_PATH,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    S3_ACCESS_KEY_ID,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
    STORAGE_PROVIDER,
)
from gen3_ai_model_repo.storage.local import LocalStorageProvider
from gen3_ai_model_repo.storage.minio import MinioStorageProvider


def get_storage_provider():
    """Return the configured storage provider implementation."""
    if STORAGE_PROVIDER == "minio":
        return MinioStorageProvider(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket_name=MINIO_BUCKET,
        )

    if STORAGE_PROVIDER == "s3":
        from gen3_ai_model_repo.storage.s3 import S3StorageProvider

        return S3StorageProvider(
            bucket_name=S3_BUCKET,
            region=S3_REGION,
            endpoint_url=S3_ENDPOINT_URL,
            access_key_id=S3_ACCESS_KEY_ID,
            secret_access_key=S3_SECRET_ACCESS_KEY,
        )

    if STORAGE_PROVIDER != "local":
        raise ValueError(f"Unsupported STORAGE_PROVIDER: {STORAGE_PROVIDER}")

    return LocalStorageProvider(LOCAL_STORAGE_PATH)


__all__ = ["get_storage_provider"]
