from gen3_ai_model_repo.storage.provider import StorageProvider


class S3StorageProvider(StorageProvider):
    """AWS S3-compatible storage provider using boto3."""

    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ):
        """Initialize the provider with S3 connection settings."""
        import boto3

        self.bucket_name = bucket_name
        session = boto3.session.Session()
        self.client = session.client(
            "s3",
            region_name=region,
            endpoint_url=endpoint_url or None,
            aws_access_key_id=access_key_id or None,
            aws_secret_access_key=secret_access_key or None,
        )

    async def upload_file(
        self,
        local_path: str,
        object_key: str,
    ):
        """Upload a local file path to S3."""
        self.client.upload_file(local_path, self.bucket_name, object_key)

    async def upload_stream(self, stream, object_key: str):
        """Upload a stream directly to S3."""
        self.client.upload_fileobj(stream, self.bucket_name, object_key)

    async def download_file(
        self,
        object_key: str,
        local_path: str,
    ):
        """Download an object from S3 to disk."""
        self.client.download_file(self.bucket_name, object_key, local_path)

    async def list_files(
        self,
        prefix: str,
    ) -> list[str]:
        """List object keys under a prefix in S3."""
        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item.get("Key")
                if key:
                    keys.append(key)
        return keys

    async def file_exists(
        self,
        object_key: str,
    ) -> bool:
        """Return whether an object key exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except Exception:
            return False

    async def delete_file(
        self,
        object_key: str,
    ):
        """Delete an object from S3."""
        self.client.delete_object(Bucket=self.bucket_name, Key=object_key)

    async def delete_prefix(
        self,
        prefix: str,
    ):
        """Delete all objects under a prefix in S3."""
        keys = await self.list_files(prefix)
        if not keys:
            return

        # S3 delete_objects supports batches of up to 1000 keys.
        for idx in range(0, len(keys), 1000):
            chunk = keys[idx : idx + 1000]
            self.client.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects": [{"Key": key} for key in chunk]},
            )

    async def generate_signed_url(
        self,
        object_key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate a pre-signed download URL for an S3 object."""
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket_name, "Key": object_key},
            ExpiresIn=expiry_seconds,
        )

    async def generate_upload_url(
        self,
        object_key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate a pre-signed upload URL for an S3 object."""
        return self.client.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": self.bucket_name, "Key": object_key},
            ExpiresIn=expiry_seconds,
        )

    async def get_file_metadata(
        self,
        object_key: str,
    ) -> dict:
        """Return metadata for an S3 object key."""
        response = self.client.head_object(Bucket=self.bucket_name, Key=object_key)
        return {
            "size": int(response.get("ContentLength", 0)),
            "etag": str(response.get("ETag", "")).strip('"') or None,
            "last_modified": response.get("LastModified"),
        }
