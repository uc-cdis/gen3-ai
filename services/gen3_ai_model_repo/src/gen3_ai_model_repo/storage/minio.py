from minio import Minio

from gen3_ai_model_repo.storage.provider import StorageProvider


class MinioStorageProvider(StorageProvider):
    """
    MinIO S3-compatible storage provider.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
    ):
        self.bucket_name = bucket_name

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )

    async def upload_file(
        self,
        local_path: str,
        object_key: str,
    ):
        self.client.fput_object(
            self.bucket_name,
            object_key,
            local_path,
        )

    async def download_file(
        self,
        object_key: str,
        local_path: str,
    ):
        self.client.fget_object(
            self.bucket_name,
            object_key,
            local_path,
        )

    async def list_files(
        self,
        prefix: str,
    ) -> list[str]:
        objects = self.client.list_objects(
            self.bucket_name,
            prefix=prefix,
            recursive=True,
        )

        return [obj.object_name for obj in objects if obj.object_name is not None]

    async def file_exists(
        self,
        object_key: str,
    ) -> bool:
        try:
            self.client.stat_object(
                self.bucket_name,
                object_key,
            )
            return True
        except Exception:
            return False
