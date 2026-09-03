"""Generic storage provider interfaces for the Gen3 AI model repo service."""

from abc import ABC, abstractmethod


class StorageProvider(ABC):
    """Base storage provider interface."""

    @abstractmethod
    async def ensure_container(self):
        """Ensure backing container (bucket/root directory) exists."""
        pass

    @abstractmethod
    async def upload_file(
        self,
        local_path: str,
        object_key: str,
    ):
        """Upload a file from disk to the backing storage."""
        pass

    @abstractmethod
    async def download_file(
        self,
        object_key: str,
        local_path: str,
    ):
        """Download an object from the backing storage to disk."""
        pass

    @abstractmethod
    async def list_files(
        self,
        prefix: str,
    ) -> list[str]:
        """List objects under a prefix in the backing storage."""
        pass

    @abstractmethod
    async def upload_stream(
        self,
        stream,
        object_key: str,
    ):
        """Upload a stream to the backing storage."""
        pass

    @abstractmethod
    async def file_exists(
        self,
        object_key: str,
    ) -> bool:
        """Return whether an object exists in the backing storage."""
        pass

    @abstractmethod
    async def delete_file(
        self,
        object_key: str,
    ):
        """Delete an object from the backing storage."""
        pass

    @abstractmethod
    async def delete_prefix(
        self,
        prefix: str,
    ):
        """Delete all objects with a given prefix."""
        pass

    @abstractmethod
    async def generate_signed_url(
        self,
        object_key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate a signed URL for downloading an object."""
        pass

    @abstractmethod
    async def generate_upload_url(
        self,
        object_key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate a signed URL for uploading an object."""
        pass

    @abstractmethod
    async def get_file_metadata(
        self,
        object_key: str,
    ) -> dict:
        """Return metadata for an object in the backing storage."""
        pass
