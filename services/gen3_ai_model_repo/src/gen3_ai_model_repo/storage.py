import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException


def get_local_file(base_dir: Path, path_parts: list[str]) -> Path:
    """Resolve and validate the requested local repository file path."""
    local_path = base_dir.joinpath(*path_parts)
    logging.debug(f"looking for file: {local_path}")
    if not local_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    logging.debug("found file!")
    return local_path


def read_file(local_path: Path) -> bytes:
    """Read file contents from a local path."""
    return local_path.read_bytes()


def compute_hashes(content: bytes) -> tuple[str, str]:
    """Compute deterministic SHA-256 and MD5 hashes for file content."""
    commit_hash = hashlib.sha256(content).hexdigest()
    etag = hashlib.md5(content).hexdigest()
    return commit_hash, etag


def list_repository_files(base_dir: Path, namespace: str, repo: str) -> list[dict[str, Any]]:
    """List files in a repository and return a mocked metadata entry for each file."""
    repo_path = base_dir / Path(namespace) / Path(repo)
    if not repo_path.exists():
        return []

    if repo_path.is_file():
        files = [repo_path]
    else:
        files = [path for path in repo_path.rglob("*") if path.is_file()]

    def make_entry(path: Path) -> dict[str, Any]:
        rel = str(path.relative_to(base_dir))
        content = path.read_bytes()
        oid, _ = compute_hashes(content)
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

    entries: list[dict[str, Any]] = []
    for path in files:
        try:
            entries.append(make_entry(path))
        except Exception as e:
            logging.error(f"Error processing file {path}: {e}")
    return entries
