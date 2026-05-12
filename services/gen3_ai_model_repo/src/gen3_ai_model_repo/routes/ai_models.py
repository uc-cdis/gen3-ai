import json
from datetime import datetime
from pathlib import Path
from shutil import rmtree
from typing import Any
from urllib.parse import urljoin

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

from gen3_ai_model_repo.services.metadata_service import MetadataService
from gen3_ai_model_repo.services.response_service import ResponseService
from gen3_ai_model_repo.services.storage_service import StorageService
from gen3_ai_model_repo.services.url_service import URLService

ai_models_router = APIRouter()


# note that the folder structure in BASE_FILES_DIR must be:
#   BASE_FILES_DIR / {namespace} / {repo}
#   ex: /testfiles/uc-ctds/bge-large-en-v1.5-bio-mapping
BASE_FILES_DIR = Path(__file__).parent / "testfiles"
FAKE_COMMIT = "mock-commit-hash-123456"
FAKE_ETAG = "mock-etag-123456"


DOMAIN = "http://127.0.0.1:4141"

storage_service = StorageService(BASE_FILES_DIR)

metadata_service = MetadataService()
url_service = URLService()
response_service = ResponseService()


class UploadModelRequest(BaseModel):
    description: str
    tags: list[str] = []


@ai_models_router.post("/api/models/{namespace}/{repo}/upload")
async def upload_model(namespace: str, repo: str, request: UploadModelRequest):
    # In a real implementation, you'd handle the file upload, store it in S3,
    # compute hashes, update metadata, etc. Here we just return a success response.

    repo_path = BASE_FILES_DIR / Path(namespace) / Path(repo)
    repo_path.mkdir(parents=True, exist_ok=True)
    metadata_file = repo_path / "metadata.json"

    metadata_content = {
        "namespace": namespace,
        "repo": repo,
        "description": "This is a mock model repository.",
        "tags": request.tags,
        "created_at": datetime.now().isoformat(timespec="seconds") + "Z",
    }
    metadata_file.write_text(json.dumps(metadata_content))
    return {
        "status": "uploaded",
        "repo": f"{namespace}/{repo}",
        "metadata_file": str(metadata_file),
    }


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
        content = path.read_bytes()
        oid, _ = storage_service.compute_hashes(content)
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
    return metadata_service.get_revision(namespace, repo, rev)


@ai_models_router.head("/{namespace}/{repo}/resolve/{rev}/{path:path}")
async def head_file(namespace: str, repo: str, rev: str, path: str):
    path_parts = [namespace, repo]
    path_parts.extend(path.split("/"))
    local_path = storage_service.get_local_file(path_parts)
    content = storage_service.read_file(local_path)

    size = len(content)
    commit_hash, etag = storage_service.compute_hashes(content)

    # also mock the redirected signed URL locally via this same
    # web service. this will stream the file contents as if it
    # was a signed URL
    signed_url = url_service.build_signed_url(namespace, repo, rev, path)

    return response_service.build_head_response(commit_hash, etag, size, signed_url)


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
    local_path = storage_service.get_local_file(path.split("/"))
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


@ai_models_router.get("/api/models/{namespace}/{repo}/info")
async def get_model_info(namespace: str, repo: str):
    """
    Return a mock model info response. This is used by the Hugging Face Hub
    to display metadata about the model in the UI. The actual content of the
    response is not important for our testing purposes, so we return a static
    response with some placeholder data.
    """
    repo_path = BASE_FILES_DIR / Path(namespace) / Path(repo)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")
    files = []
    for path in repo_path.rglob("*"):
        if path.is_file():
            relative_path = str(path.relative_to(BASE_FILES_DIR))
            size = path.stat().st_size
            files.append(
                {
                    "filename": relative_path,
                    "size": size,
                    "etag": FAKE_ETAG,
                }
            )
    metadata_file = repo_path / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
    return {
        "id": f"{namespace}/{repo}",
        "sha": FAKE_COMMIT,
        "etag": FAKE_ETAG,
        "size": 123456,
        "files": files,
        "metadata": metadata,
        "securityStatus": {
            "status": "unscanned",
            "jFrogScan": {"status": "unscanned"},
            "protectAiScan": {"status": "unscanned"},
            "avScan": {"status": "unscanned"},
            "pickleImportScan": {"status": "unscanned"},
            "virusTotalScan": {"status": "unscanned"},
        },
    }


@ai_models_router.get("/api/models")
async def list_models():
    repos = []
    for namespace_dir in BASE_FILES_DIR.iterdir():
        if namespace_dir.is_dir():
            for repo_dir in namespace_dir.iterdir():
                if repo_dir.is_dir():
                    repos.append(
                        {
                            "id": f"{namespace_dir.name}/{repo_dir.name}",
                            "description": "This is a mock model repository.",
                            "tags": ["example", "test"],
                            "created_at": datetime.now().isoformat(timespec="seconds") + "Z",
                        }
                    )

        return repos


@ai_models_router.delete("/api/models/{namespace}/{repo}")
async def delete_model(namespace: str, repo: str):
    repo_path = BASE_FILES_DIR / Path(namespace) / Path(repo)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")
    rmtree(repo_path)
    return {"status": "deleted", "repo": f"{namespace}/{repo}"}
