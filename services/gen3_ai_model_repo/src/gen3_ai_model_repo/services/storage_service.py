import hashlib
import logging
from datetime import datetime
from http.client import HTTPException
from pathlib import Path
from typing import Any


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

    def list_repository_files(self, namespace: str, repo: str):
        repo_path = self.base_dir / Path(namespace) / Path(repo)
        if not repo_path.exists():
            return []
        files = list(repo_path.rglob("*"))

        # gather all files under target. If target is a file, return it
        # as the single entry
        if repo_path.is_file():
            files = [repo_path]
        else:
            files = [path for path in repo_path.rglob("*") if path.is_file()]

        def make_entry(path: Path) -> dict[str, Any]:
            rel = str(path.relative_to(self.base_dir))
            content = path.read_bytes()
            oid, _ = self.compute_hashes(content)
            size = path.stat().st_size
            return {
                "type": "file",
                "oid": oid,
                "size": size,
                "path": rel,
                "lfs": {"oid": oid, "size": size, "pointerSize": size},
                "xetHash": None,
                "lastCommit": {
                    "id": oid,
                    "title": "Mock title",
                    "date": datetime.now().isoformat(timespec="seconds") + "Z",
                },
                "securityFileStatus": {
                    "status": "unscanned",
                    "jFrogScan": {"status": "unscanned"},
                    "protectAiScan": {"status": "unscanned"},
                    "avScan": {"status": "unscanned"},
                    "pickleImportScan": {"status": "unscanned"},
                    "virusTotalScan": {"status": "unscanned"},
                },
            }

        entries = []
        for path in files:
            try:
                entry = make_entry(path)
                entries.append(entry)
            except Exception as e:
                logging.error(f"Error processing file {path}: {e}")
        return entries
