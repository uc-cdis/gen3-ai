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
    async def upload_stream(
        self,
        stream,
        object_key: str,
    ):
        pass

    @abstractmethod
    async def file_exists(
        self,
        object_key: str,
    ) -> bool:
        pass

    @abstractmethod
    async def delete_file(
        self,
        object_key: str,
    ):
        pass

    @abstractmethod
    async def delete_prefix(
        self,
        prefix: str,
    ):
        pass

    @abstractmethod
    async def generate_signed_url(
        self,
        object_key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        pass

    @abstractmethod
    async def generate_upload_url(
        self,
        object_key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        pass

    @abstractmethod
    async def get_file_metadata(
        self,
        object_key: str,
    ) -> dict:
        pass
