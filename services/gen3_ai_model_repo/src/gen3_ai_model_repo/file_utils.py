import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from gen3_ai_model_repo.constants import DEFAULT_SECURITY_FILE_STATUS


def get_local_file(base_dir: Path, path_parts: list[str]) -> Path:
    """
    Resolve and validate the requested local repository file path.

    Constructs a full path from the base directory and path parts, then validates
    that the resolved path exists and is a regular file. Prevents directory traversal
    attacks by ensuring the path is within the base directory.

    Args:
        base_dir: The base directory where repositories are stored.
        path_parts: List of path components to join (e.g., ['namespace', 'repo', 'file.txt']).

    Returns:
        Path: The validated local file path.

    Raises:
        HTTPException: 404 if the file does not exist or is not a regular file.
    """
    local_path = base_dir.joinpath(*path_parts)
    logging.debug(f"looking for file: {local_path}")
    if not local_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    logging.debug("found file!")
    return local_path


def read_file(local_path: Path) -> bytes:
    """
    Read file contents from a local path.

    Reads the entire file content into memory as bytes. Use with caution for
    large files; consider streaming for files larger than available memory.

    Args:
        local_path: The file path to read.

    Returns:
        bytes: The complete file content.

    Raises:
        IOError: If the file cannot be read.
    """
    return local_path.read_bytes()


def compute_hashes(content: bytes) -> tuple[str, str]:
    """
    Compute deterministic SHA-256 and MD5 hashes for file content.

    Generates both SHA-256 (for content integrity checking) and MD5 (for ETag/compatibility)
    hashes for the given content. These hashes are used for deduplication and for building
    commit SHAs from multiple files.

    Args:
        content: The file content as bytes.

    Returns:
        tuple: A 2-tuple containing (commit_hash, etag) where:
            - commit_hash: The SHA-256 hash as a hexadecimal string.
            - etag: The MD5 hash as a hexadecimal string for ETag usage.
    """
    commit_hash = hashlib.sha256(content).hexdigest()
    etag = hashlib.md5(content).hexdigest()
    return commit_hash, etag


def list_repository_files(base_dir: Path, namespace: str, repo: str) -> list[dict[str, Any]]:
    """
    List files in a repository and return metadata entries for each file.

    Recursively walks the repository directory and generates metadata entries for
    all files. Each entry includes file type, object ID (OID), size, path, and
    mocked metadata fields for compatibility with Hugging Face Hub API.

    Args:
        base_dir: The base directory where repositories are stored.
        namespace: The namespace/organization for the repository.
        repo: The repository name.

    Returns:
        list: A list of dictionaries, each containing metadata for a file:
            - type: Always 'file' for this implementation.
            - oid: The SHA-256 hash of the file content (object ID).
            - size: File size in bytes.
            - path: Relative path from the repository root.
            - lfs: Large File Storage metadata (oid and size).
            - xetHash: XetHub hash (mocked as None).
            - lastCommit: Mock commit information (ID, title, timestamp).
            - securityFileStatus: Mock security scan status (unscanned).

    Raises:
        No exceptions raised; errors processing individual files are logged and skipped.
    """
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
            "securityFileStatus": DEFAULT_SECURITY_FILE_STATUS,
        }

    entries: list[dict[str, Any]] = []
    for path in files:
        try:
            entries.append(make_entry(path))
        except Exception as e:
            logging.error(f"Error processing file {path}: {e}")
    return entries


def compute_file_sha256(path: Path) -> str:
    """
    Compute SHA256 without loading file into memory.
    """

    sha = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha.update(chunk)

    return sha.hexdigest()


def compute_file_md5(path: Path) -> str:
    """
    Compute MD5 without loading file into memory.
    """

    md5 = hashlib.md5(usedforsecurity=False)

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            md5.update(chunk)

    return md5.hexdigest()
