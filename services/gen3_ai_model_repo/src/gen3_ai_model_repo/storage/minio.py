"""MinIO storage for the Gen3 AI model repo service."""

from datetime import timedelta

from minio import Minio

from gen3_ai_model_repo.storage.provider import StorageProvider


class MinioStorageProvider(StorageProvider):
    """MinIO S3-compatible storage provider."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = False,
        create_bucket_if_missing: bool = True,
    ):
        """Initialize the provider with MinIO connection settings."""
        self.bucket_name = bucket_name
        self.create_bucket_if_missing = create_bucket_if_missing

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    async def ensure_container(self):
        """Ensure the configured MinIO bucket exists."""
        if self.client.bucket_exists(self.bucket_name):
            return

        if self.create_bucket_if_missing:
            self.client.make_bucket(self.bucket_name)
            return

        raise FileNotFoundError(f"Bucket does not exist: {self.bucket_name}")

    async def upload_file(
        self,
        local_path: str,
        object_key: str,
    ):
        """Upload a file to the MinIO bucket."""
        self.client.fput_object(
            self.bucket_name,
            object_key,
            local_path,
        )

    async def upload_stream(self, stream, object_key: str):
        """Upload a stream to the MinIO bucket."""
        self.client.put_object(
            self.bucket_name,
            object_key,
            stream,
            length=-1,
            part_size=10 * 1024 * 1024,
        )

    async def download_file(
        self,
        object_key: str,
        local_path: str,
    ):
        """Download an object from the MinIO bucket to disk."""
        self.client.fget_object(
            self.bucket_name,
            object_key,
            local_path,
        )

    async def list_files(
        self,
        prefix: str,
    ) -> list[str]:
        """List objects stored under a prefix in MinIO."""
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
        """Return whether an object exists in the MinIO bucket."""
        try:
            self.client.stat_object(
                self.bucket_name,
                object_key,
            )
            return True
        except Exception:
            return False

    async def delete_file(
        self,
        object_key: str,
    ):
        """Delete an object from the MinIO bucket."""
        self.client.remove_object(
            self.bucket_name,
            object_key,
        )

    async def delete_prefix(
        self,
        prefix: str,
    ):
        """Delete all objects stored under a prefix in MinIO."""
        objects = self.client.list_objects(
            self.bucket_name,
            prefix=prefix,
            recursive=True,
        )
        self.client.remove_objects(
            self.bucket_name,
            [obj.object_name for obj in objects if obj.object_name],
        )

    async def generate_signed_url(
        self,
        object_key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate a signed URL for downloading an object."""
        return self.client.presigned_get_object(
            self.bucket_name,
            object_key,
            expires=timedelta(seconds=expiry_seconds),
        )

    async def generate_upload_url(
        self,
        object_key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate a signed URL for uploading an object."""
        return self.client.presigned_put_object(
            self.bucket_name,
            object_key,
            expires=timedelta(seconds=expiry_seconds),
        )

    async def get_file_metadata(
        self,
        object_key: str,
    ) -> dict:
        """Return metadata for an object stored in MinIO."""
        stat = self.client.stat_object(
            self.bucket_name,
            object_key,
        )
        return {
            "size": stat.size,
            "etag": stat.etag,
            "last_modified": stat.last_modified,
        }
