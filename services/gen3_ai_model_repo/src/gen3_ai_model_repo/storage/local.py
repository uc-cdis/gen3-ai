from pathlib import Path

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

        destination.write_bytes(Path(local_path).read_bytes())

    async def download_file(
        self,
        object_key: str,
        local_path: str,
    ):
        source = self.root_directory / object_key

        Path(local_path).write_bytes(source.read_bytes())

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
