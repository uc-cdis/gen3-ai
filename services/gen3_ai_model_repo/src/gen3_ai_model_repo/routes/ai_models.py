import hashlib
from pathlib import Path
from urllib.parse import urljoin

from fastapi import APIRouter, Header, HTTPException, Query, Response, status
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

from gen3_ai_model_repo.auth import validate_token
from gen3_ai_model_repo.database.helper import (
    create_repository_metadata,
    delete_repository_metadata,
    get_or_create_revision,
    get_repository_metadata,
    list_all_repositories,
    list_files_in_revision,
    list_revisions,
    track_file,
)
from gen3_ai_model_repo.database.helper import (
    repository_exists as db_repository_exists,
)
from gen3_ai_model_repo.metadata import (
    create_metadata,
    delete_repository,
)
from gen3_ai_model_repo.metadata import (
    get_revision as metadata_get_revision,
)
from gen3_ai_model_repo.models.schemas import (
    DeleteModelResponse,
    RepositoryFileModel,
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
) -> UploadModelResponse:
    validate_token(authorization)

    # Create metadata file and database record
    metadata_file = create_metadata(BASE_FILES_DIR, namespace, repo, request.description, request.tags)
    metadata_model = await create_repository_metadata(
        namespace=namespace,
        repo_name=repo,
        description=request.description,
        tags=request.tags,
    )

    # Compute hashes for all files in the repository
    repo_path = BASE_FILES_DIR / Path(namespace) / Path(repo)
    all_hashes = []
    total_size = 0

    if repo_path.exists():
        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                content = read_file(file_path)
                commit_hash, etag = compute_hashes(content)
                relative_path = str(file_path.relative_to(repo_path))
                all_hashes.append(commit_hash)
                total_size += len(content)

                # Track file in database
                await track_file(
                    namespace=namespace,
                    repo_name=repo,
                    revision_name="main",
                    file_path=relative_path,
                    file_size=len(content),
                    content_sha=commit_hash,
                    content_etag=etag,
                )

    # Create a combined commit hash from all files
    if all_hashes:
        combined = "".join(all_hashes)
        main_commit_sha = hashlib.sha256(combined.encode()).hexdigest()
    else:
        main_commit_sha = hashlib.sha256(b"").hexdigest()

    # Create the main revision
    await get_or_create_revision(
        namespace=namespace,
        repo_name=repo,
        revision_name="main",
        commit_sha=main_commit_sha,
        etag=None,
    )

    return UploadModelResponse(
        status="uploaded", repo=f"{namespace}/{repo}", metadata_file=str(metadata_file), metadata=metadata_model
    )


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
) -> list[TreeEntryModel]:
    """
    Return a flat list of entries for the directory *path* (or the file
    itself).  The output matches the structure documented by Hugging Face
    but contains only the essential fields.
    """
    files = list_repository_files(BASE_FILES_DIR, namespace, repo)

    if not files:
        raise HTTPException(status_code=404, detail="Repository or path not found")
    return [TreeEntryModel(type=f["type"], oid=f["oid"], size=f["size"]) for f in files]


@ai_models_router.get("/api/models/{namespace}/{repo}/revision/{rev}", response_model=RevisionModel)
async def get_revision(namespace: str, repo: str, rev: str) -> RevisionModel:
    data = metadata_get_revision(namespace, repo, rev)
    return RevisionModel(**data)


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
async def get_model_info(namespace: str, repo: str) -> RepositoryInfoModel:
    """
    Return model info response with metadata from database.
    Files are listed from the repository directory.
    """
    # Check if repo exists in database
    repo_exists_in_db = await db_repository_exists(namespace, repo)
    if not repo_exists_in_db:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Get metadata from database
    metadata = await get_repository_metadata(namespace, repo)
    if not metadata:
        raise HTTPException(status_code=404, detail="Metadata not found")

    # Get revision info (use main as default)
    revision_info = await get_or_create_revision(
        namespace=namespace,
        repo_name=repo,
        revision_name="main",
    )

    if not revision_info:
        raise HTTPException(status_code=404, detail="Revision not found")

    # Get files from database
    files_from_db = await list_files_in_revision(
        namespace=namespace,
        repo_name=repo,
        revision_name="main",
    )

    # Calculate total size
    total_size = sum(f["size"] for f in files_from_db)

    return RepositoryInfoModel(
        id=f"{namespace}/{repo}",
        sha=revision_info["sha"],
        etag=revision_info["etag"] or revision_info["sha"],
        size=total_size,
        files=[RepositoryFileModel(type=f["type"], oid=f["oid"], size=f["size"]) for f in files_from_db],
        metadata=metadata,
        security_status={
            "status": "unscanned",
            "jFrogScan": {"status": "unscanned"},
            "protectAiScan": {"status": "unscanned"},
            "avScan": {"status": "unscanned"},
            "pickleImportScan": {"status": "unscanned"},
            "virusTotalScan": {"status": "unscanned"},
        },
    )


@ai_models_router.get("/api/models", response_model=list[RepositoryModel])
async def list_models() -> list[RepositoryModel]:
    """Retrieve all repositories from database."""
    repos = await list_all_repositories()
    return [
        RepositoryModel(
            id=f"{repo.namespace}/{repo.repo}",
            description=repo.description,
            tags=repo.tags,
            created_at=repo.created_at,
        )
        for repo in repos
    ]


@ai_models_router.delete("/api/models/{namespace}/{repo}", response_model=DeleteModelResponse)
async def delete_model(
    namespace: str, repo: str, authorization: str | None = Header(default=None)
) -> DeleteModelResponse:
    """Delete repository metadata from database and files from disk."""
    validate_token(authorization)

    # Check if repository exists
    repo_exists_check = await db_repository_exists(namespace, repo)
    if not repo_exists_check:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Delete from database
    deleted_from_db = await delete_repository_metadata(namespace, repo)
    if not deleted_from_db:
        raise HTTPException(status_code=500, detail=f"Failed to delete repository {namespace}/{repo} from database")

    # Delete files from disk if they exist
    repo_path = BASE_FILES_DIR / Path(namespace) / Path(repo)
    if repo_path.exists():
        try:
            delete_repository(BASE_FILES_DIR, namespace, repo)
        except Exception as e:
            # Log but don't fail if file deletion fails
            print(f"Warning: Failed to delete repository files: {e}")

    return DeleteModelResponse(status="deleted", repo=f"{namespace}/{repo}")


@ai_models_router.get("/api/models/{namespace}/{repo}/revisions", response_model=RevisionListResponseModel)
async def list_model_revisions(
    namespace: str, repo: str, authorization: str | None = Header(default=None)
) -> RevisionListResponseModel:
    """List model revisions from database."""
    validate_token(authorization)

    # Verify repository exists in database
    repo_exists_check = await db_repository_exists(namespace, repo)
    if not repo_exists_check:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Get all revisions from database
    revisions_data = await list_revisions(namespace, repo)

    if not revisions_data:
        raise HTTPException(status_code=404, detail="No revisions found for repository")

    # Convert to RevisionModel objects
    revisions = [
        RevisionModel(
            id=rev["sha"],
            revision=rev["revision"],
            sha=rev["sha"],
        )
        for rev in revisions_data
    ]

    return RevisionListResponseModel(
        repo=f"{namespace}/{repo}",
        revisions=revisions,
    )
