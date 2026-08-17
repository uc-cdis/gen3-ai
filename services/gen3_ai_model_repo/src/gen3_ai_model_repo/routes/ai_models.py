from pathlib import Path
from urllib.parse import urljoin

from fastapi import APIRouter, Header, HTTPException, Query, Response, status
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

from gen3_ai_model_repo.auth import validate_token
from gen3_ai_model_repo.metadata import (
    create_metadata,
    delete_repository,
    list_repositories,
    load_metadata,
    repository_exists,
)
from gen3_ai_model_repo.metadata import (
    get_revision as metadata_get_revision,
)
from gen3_ai_model_repo.models.repo_model import (
    RepositoryInfoModel,
    RepositoryModel,
    RevisionListResponseModel,
    RevisionModel,
    TreeEntryModel,
    UploadModelResponse,
)
from gen3_ai_model_repo.response import build_head_response
from gen3_ai_model_repo.storage import (
    compute_hashes,
    get_local_file,
    list_repository_files,
    read_file,
)
from gen3_ai_model_repo.url import build_signed_url

ai_models_router = APIRouter()


# note that the folder structure in BASE_FILES_DIR must be:
#   BASE_FILES_DIR / {namespace} / {repo}
#   ex: /testfiles/uc-ctds/bge-large-en-v1.5-bio-mapping
BASE_FILES_DIR = Path(__file__).parent / "testfiles"
FAKE_COMMIT = "mock-commit-hash-123456"
FAKE_ETAG = "mock-etag-123456"


DOMAIN = "http://127.0.0.1:4141"


class UploadModelRequest(BaseModel):
    description: str
    tags: list[str] = []


@ai_models_router.post("/api/models/{namespace}/{repo}/upload", response_model=UploadModelResponse)
async def upload_model(
    namespace: str,
    repo: str,
    request: UploadModelRequest,
    authorization: str | None = Header(default=None),
):
    validate_token(authorization)

    metadata_file = create_metadata(BASE_FILES_DIR, namespace, repo, request.description, request.tags)
    return {
        "status": "uploaded",
        "repo": f"{namespace}/{repo}",
        "metadata_file": str(metadata_file),
    }


@ai_models_router.get("/api/models/{namespace}/{repo}/tree/{rev}", response_model=list[TreeEntryModel])
@ai_models_router.get(
    "/api/models/{namespace}/{repo}/tree/{rev}/{path:path}",
    response_model=list[TreeEntryModel],
)
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
    files = list_repository_files(BASE_FILES_DIR, namespace, repo)
    if not files:
        raise HTTPException(status_code=404, detail="Repository or path not found")
    return files


@ai_models_router.get("/api/models/{namespace}/{repo}/revision/{rev}", response_model=RevisionModel)
async def get_revision(namespace: str, repo: str, rev: str):
    return metadata_get_revision(namespace, repo, rev)


@ai_models_router.head("/api/models/{namespace}/{repo}/resolve/{rev}/{path:path}")
async def head_file(namespace: str, repo: str, rev: str, path: str):
    path_parts = [namespace, repo]
    path_parts.extend(path.split("/"))
    local_path = get_local_file(BASE_FILES_DIR, path_parts)
    content = read_file(local_path)

    size = len(content)
    commit_hash, etag = compute_hashes(content)

    # also mock the redirected signed URL locally via this same
    # web service. this will stream the file contents as if it
    # was a signed URL
    signed_url = build_signed_url(namespace, repo, rev, path)

    return build_head_response(commit_hash, etag, size, signed_url)


@ai_models_router.get("/api/models/{namespace}/{repo}/resolve/{rev}/{path:path}")
async def get_file(namespace: str, repo: str, rev: str, path: str):
    print(f"Received request for file: {namespace}/{repo}/{rev}/{path}")
    signed_url = urljoin(
        f"{DOMAIN}/signed-url/",
        f"{namespace}/{repo}/{path}",
    )
    # this redirect is how our service would work. we'd do auth checks, find
    # the file in s3, create a signed URL and return
    print(f"Redirecting to signed URL: {signed_url}")
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
    local_path = get_local_file(BASE_FILES_DIR, path.split("/"))
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


@ai_models_router.get("/api/models/{namespace}/{repo}/info", response_model=RepositoryInfoModel)
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
                    "type": "file",
                    "oid": relative_path,
                    "size": size,
                }
            )

    try:
        metadata = load_metadata(BASE_FILES_DIR, namespace, repo)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metadata file not found")

    return {
        "id": f"{namespace}/{repo}",
        "sha": FAKE_COMMIT,
        "etag": FAKE_ETAG,
        "size": 123456,
        "files": files,
        "metadata": metadata,
        "security_status": {
            "status": "unscanned",
            "jFrogScan": {"status": "unscanned"},
            "protectAiScan": {"status": "unscanned"},
            "avScan": {"status": "unscanned"},
            "pickleImportScan": {"status": "unscanned"},
            "virusTotalScan": {"status": "unscanned"},
        },
    }


@ai_models_router.get("/api/models", response_model=list[RepositoryModel])
async def list_models():
    return list_repositories(BASE_FILES_DIR)


@ai_models_router.delete("/api/models/{namespace}/{repo}")
async def delete_model(namespace: str, repo: str, authorization: str | None = Header(default=None)):
    validate_token(authorization)
    exists = repository_exists(BASE_FILES_DIR, namespace, repo)
    if not exists:
        raise HTTPException(status_code=404, detail="Repository not found")
    if delete_repository(BASE_FILES_DIR, namespace, repo):
        return {"status": "deleted", "repo": f"{namespace}/{repo}"}
    return {
        "status": "error",
        "message": f"Failed to delete repository {namespace}/{repo}",
    }


@ai_models_router.get("/api/models/{namespace}/{repo}/revisions", response_model=RevisionListResponseModel)
async def list_model_revisions(namespace: str, repo: str, authorization: str | None = Header(default=None)):
    validate_token(authorization)
    repo_path = BASE_FILES_DIR / Path(namespace) / Path(repo)
    if not repo_path.exists():
        raise HTTPException(status_code=404, detail="Repository not found")
    return {
        "repo": f"{namespace}/{repo}",
        "revisions": [
            {
                "id": FAKE_COMMIT,
                "revision": "main",
                "sha": FAKE_COMMIT,
            }
        ],
    }
