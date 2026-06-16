from gen3_ai_model_repo.config import (
    LOCAL_STORAGE_PATH,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    STORAGE_PROVIDER,
)
from gen3_ai_model_repo.storage.local import (
    LocalStorageProvider,
)
from gen3_ai_model_repo.storage.minio import (
    MinioStorageProvider,
)


def get_storage_provider():
    if STORAGE_PROVIDER == "minio":
        return MinioStorageProvider(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            bucket_name=MINIO_BUCKET,
        )

    return LocalStorageProvider(
        LOCAL_STORAGE_PATH,
    )
