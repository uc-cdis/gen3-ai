import hashlib
import logging
from http.client import HTTPException
from pathlib import Path


class StorageService:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def get_local_file(self, path_parts: list[str]) -> Path:
        local_path = self.base_dir.joinpath(*path_parts)
        logging.debug(f"looking for file: {local_path}")
        if not local_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        logging.debug("found file!")
        return local_path

    def read_file(self, local_path: Path) -> bytes:
        return local_path.read_bytes()

    def compute_hashes(self, content: bytes) -> tuple[str, str]:
        commit_hash = hashlib.sha256(content).hexdigest()
        etag = hashlib.md5(content).hexdigest()
        return commit_hash, etag
