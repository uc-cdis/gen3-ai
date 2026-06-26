import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from gen3_ai_model_repo.storage.provider import StorageProvider


class LocalStorageProvider(StorageProvider):
    """
    Local filesystem storage provider.
    """

    def __init__(self, root_directory: str):
        self.root_directory = Path(root_directory)

    async def upload_file(
        self,
        local_path: str,
        object_key: str,
    ):
        destination = self.root_directory / object_key

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with Path(local_path).open("rb") as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

    async def upload_stream(self, stream, object_key: str):
        destination = self.root_directory / object_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as dst:
            shutil.copyfileobj(stream, dst, length=1024 * 1024)

    async def download_file(
        self,
        object_key: str,
        local_path: str,
    ):
        source = self.root_directory / object_key
        with source.open("rb") as src, Path(local_path).open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

    async def list_files(
        self,
        prefix: str,
    ) -> list[str]:
        base = self.root_directory / prefix

        if not base.exists():
            return []

        return [str(path.relative_to(self.root_directory)) for path in base.rglob("*") if path.is_file()]

    async def file_exists(
        self,
        object_key: str,
    ) -> bool:
        return (self.root_directory / object_key).exists()

    async def delete_file(
        self,
        object_key: str,
    ):
        target = self.root_directory / object_key
        if target.exists():
            target.unlink()

    async def delete_prefix(
        self,
        prefix: str,
    ):
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
        del expiry_seconds
        return f"/signed-url/{quote(object_key)}"

    async def generate_upload_url(
        self,
        object_key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        del expiry_seconds
        return f"/upload-url/{quote(object_key)}?token={uuid4()}"

    async def get_file_metadata(
        self,
        object_key: str,
    ) -> dict:
        path = self.root_directory / object_key
        stat = path.stat()
        return {
            "size": stat.st_size,
            "etag": None,
            "last_modified": datetime.fromtimestamp(stat.st_mtime),
        }
