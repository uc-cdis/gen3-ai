from abc import ABC, abstractmethod


class StorageProvider(ABC):
    """
    Base storage provider interface.
    """

    @abstractmethod
    async def upload_file(
        self,
        local_path: str,
        object_key: str,
    ):
        pass

    @abstractmethod
    async def download_file(
        self,
        object_key: str,
        local_path: str,
    ):
        pass

    @abstractmethod
    async def list_files(
        self,
        prefix: str,
    ) -> list[str]:
        pass

    @abstractmethod
    async def file_exists(
        self,
        object_key: str,
    ) -> bool:
        pass
