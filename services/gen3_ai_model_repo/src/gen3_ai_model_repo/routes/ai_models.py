import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import RedirectResponse, StreamingResponse

from gen3_ai_model_repo.config import logging

ai_models_router = APIRouter()

# note that the folder structure in BASE_FILES_DIR must be:
#   BASE_FILES_DIR / {namespace} / {repo}
#   ex: /testfiles/uc-ctds/bge-large-en-v1.5-bio-mapping
BASE_FILES_DIR = Path(__file__).parent / "testfiles"
FAKE_COMMIT = "mock-commit-hash-123456"
FAKE_ETAG = "mock-etag-123456"

DOMAIN = "http://127.0.0.1:4141"


@ai_models_router.get("/api/models/{namespace}/{repo}/tree/{rev}")
@ai_models_router.get("/api/models/{namespace}/{repo}/tree/{rev}/{path:path}")
async def list_repo_tree(
    namespace: str,
    repo: str,
    rev: str,
    path: str = "",
    expand: bool = Query(False, description="return commit data & minimal security info"),
):
    """
    Return a flat list of entries for the directory *path* (or the file
    itself).  The output matches the structure documented by Hugging Face
    but contains only the essential fields.
    """
    target = BASE_FILES_DIR / Path(path)

    # validate the path exists
    if not target.exists():
        raise HTTPException(status_code=404, detail="Folder not found")

    # gather all files under target. If target is a file, return it
    # as the single entry
    if target.is_file():
        files = [target]
    else:
        files = [path for path in target.rglob("*") if path.is_file()]

    def make_entry(path: Path) -> dict[str, Any]:
        rel = str(path.relative_to(BASE_FILES_DIR))
        oid = FAKE_COMMIT
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

    return [make_entry(path) for path in files]


@ai_models_router.get("/api/models/{namespace}/{repo}/revision/{rev}")
async def get_revision(namespace: str, repo: str, rev: str):
    return {
        "id": f"{namespace}/{repo}",
        "revision": rev,
        "sha": FAKE_COMMIT,
        "commit": FAKE_COMMIT,
        "tags": ["latest", "main"],
    }


@ai_models_router.head("/{namespace}/{repo}/resolve/{rev}/{path:path}")
async def head_file(namespace: str, repo: str, rev: str, path: str):
    path_parts = [namespace, repo]
    path_parts.extend(path.split("/"))
    local_path = _get_local_file(path_parts)
    content = _read_file(local_path)

    size = len(content)
    commit_hash, etag = _compute_hashes(content)

    # also mock the redirected signed URL locally via this same
    # web service. this will stream the file contents as if it
    # was a signed URL
    signed_url = urljoin(
        f"{DOMAIN}/signed-url/",
        f"{namespace}/{repo}/{path}",
    )

    headers = {
        "X-Repo-Commit": commit_hash,
        "X-Linked-Etag": etag,
        "X-Linked-Size": str(size),
        "Location": signed_url,
    }
    return Response(status_code=status.HTTP_200_OK, headers=headers)


@ai_models_router.get("/{namespace}/{repo}/resolve/{rev}/{path:path}")
async def get_file(namespace: str, repo: str, rev: str, path: str):
    signed_url = urljoin(
        f"{DOMAIN}/signed-url/",
        f"{namespace}/{repo}/{path}",
    )
    # this redirect is how our service would work. we'd do auth checks, find
    # the file in s3, create a signed URL and return
    return RedirectResponse(url=signed_url, status_code=status.HTTP_302_FOUND)


@ai_models_router.get("/health")
async def health():
    return Response()


@ai_models_router.get("/signed-url/{path:path}")
async def signed_url(path: str):
    """
    Return the file content as a streaming response.
    This is necessary for large files and guarantees the
    client sees a proper `Content-Length` header.
    """
    local_path = _get_local_file(path.split("/"))
    file_size = local_path.stat().st_size

    media_type = "application/json" if path.endswith(".json") else "application/octet-stream"

    # yields the file in chunks
    def file_iterator(path: Path, chunk_size: int = 65536):
        with path.open("rb") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    headers = {
        "Content-Length": str(file_size),
        "Content-Type": media_type,
    }

    return StreamingResponse(
        file_iterator(local_path),
        media_type=media_type,
        headers=headers,
    )


def _get_local_file(path_parts: list[str]) -> Path:
    local_path = BASE_FILES_DIR.joinpath(*path_parts)
    logging.debug(f"looking for file: {local_path}")
    if not local_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    logging.debug("found file!")
    return local_path


def _read_file(local_path: Path) -> bytes:
    return local_path.read_bytes()


def _compute_hashes(content: bytes) -> tuple[str, str]:
    commit_hash = hashlib.sha256(content).hexdigest()
    etag = hashlib.md5(content).hexdigest()
    return commit_hash, etag
