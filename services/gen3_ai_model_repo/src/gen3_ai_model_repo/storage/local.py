"""Local filesystem storage for the Gen3 AI model repo service."""

import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from gen3_ai_model_repo.storage.provider import StorageProvider


class LocalStorageProvider(StorageProvider):
    """Local filesystem storage provider."""

    def __init__(self, root_directory: str):
        """Initialize the provider with a root directory for file storage."""
        self.root_directory = Path(root_directory)

    async def ensure_container(self):
        """Ensure the local root directory exists."""
        self.root_directory.mkdir(parents=True, exist_ok=True)

    async def upload_file(
        self,
        local_path: str,
        object_key: str,
    ):
        """Upload a file to the local storage root."""
        destination = self.root_directory / object_key

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with Path(local_path).open("rb") as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

    async def upload_stream(self, stream, object_key: str):
        """Upload a data stream to the local storage root."""
        destination = self.root_directory / object_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as dst:
            shutil.copyfileobj(stream, dst, length=1024 * 1024)

    async def download_file(
        self,
        object_key: str,
        local_path: str,
    ):
        """Download a stored object to a local path."""
        source = self.root_directory / object_key
        with source.open("rb") as src, Path(local_path).open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

    async def list_files(
        self,
        prefix: str,
    ) -> list[str]:
        """List files stored under a prefix."""
        base = self.root_directory / prefix

        if not base.exists():
            return []

        return [str(path.relative_to(self.root_directory)) for path in base.rglob("*") if path.is_file()]

    async def file_exists(
        self,
        object_key: str,
    ) -> bool:
        """Return whether an object exists in local storage."""
        return (self.root_directory / object_key).exists()

    async def delete_file(
        self,
        object_key: str,
    ):
        """Delete a stored object from local storage."""
        target = self.root_directory / object_key
        if target.exists():
            target.unlink()

    async def delete_prefix(
        self,
        prefix: str,
    ):
        """Delete all stored objects beneath a prefix."""
        base = self.root_directory / prefix
        if not base.exists():
            return
        for path in sorted(base.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
        for path in sorted(base.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()

    async def generate_signed_url(
        self,
        object_key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate a signed URL for an object in local storage."""
        del expiry_seconds
        return f"/signed-url/{quote(object_key)}"

    async def generate_upload_url(
        self,
        object_key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate an upload URL for an object in local storage."""
        del expiry_seconds
        return f"/upload-url/{quote(object_key)}?token={uuid4()}"

    async def get_file_metadata(
        self,
        object_key: str,
    ) -> dict:
        """Return file metadata for an object in local storage."""
        path = self.root_directory / object_key
        stat = path.stat()
        return {
            "size": stat.st_size,
            "etag": None,
            "last_modified": datetime.fromtimestamp(stat.st_mtime),
        }
