import hashlib
from pathlib import Path
from urllib.parse import urljoin

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

from gen3_ai_model_repo.auth import verify_authorization
from gen3_ai_model_repo.config import FILE_STREAM_CHUNK_SIZE
from gen3_ai_model_repo.constants import DEFAULT_SECURITY_FILE_STATUS
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
from gen3_ai_model_repo.file_utils import (
    compute_hashes,
    get_local_file,
    list_repository_files,
    read_file,
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


@ai_models_router.post(
    "/api/models/{namespace}/{repo}/upload",
    response_model=UploadModelResponse,
    summary="Upload a new model repository",
    description=(
        "Create a new model repository with metadata and track all files. "
        "Computes hashes for all files in the repository and stores them in the database. "
        "Requires authentication via authorization header."
    ),
    responses={
        status.HTTP_200_OK: {"description": "Model successfully uploaded"},
        status.HTTP_401_UNAUTHORIZED: {"description": "User unauthenticated"},
        status.HTTP_403_FORBIDDEN: {"description": "User does not have access"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Failed to upload model"},
    },
    tags=["Models"],
)
async def upload_model(
    namespace: str,
    repo: str,
    request: UploadModelRequest,
    _: None = Depends(verify_authorization),
) -> UploadModelResponse:
    """
    Upload a new AI model repository to the service.

    Creates a new repository with the specified namespace and name, stores metadata
    in the database, and tracks all files in the repository directory. Computes SHA-256
    and MD5 hashes for all files to enable efficient deduplication.

    Args:
        namespace: The namespace/organization for the model (e.g., 'uc-ctds').
        repo: The repository name for the model (e.g., 'bge-large-en-v1.5-bio-mapping').
        request: The upload request containing description and optional tags.
        _: Authentication dependency (automatically validated by FastAPI).

    Returns:
        UploadModelResponse: Contains status, repository identifier, metadata file path,
            and the full metadata model including namespace, description, tags, and timestamps.

    Raises:
        HTTPException: 401 if authorization is invalid, 500 if database operations fail.
    """

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
        status="uploaded",
        repo=f"{namespace}/{repo}",
        metadata_file=str(metadata_file),
        metadata=metadata_model,
    )


@ai_models_router.get(
    "/api/models/{namespace}/{repo}/tree/{rev}",
    response_model=list[TreeEntryModel],
    summary="List repository directory contents",
    description="Return a flat list of entries for the repository. The output matches the structure documented by Hugging Face.",
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved directory listing"},
        status.HTTP_404_NOT_FOUND: {"description": "Repository or path not found"},
    },
    tags=["Models"],
)
@ai_models_router.get(
    "/api/models/{namespace}/{repo}/tree/{rev}/{path:path}",
    response_model=list[TreeEntryModel],
    summary="List repository path contents",
    description="Return a flat list of entries for the specified path in the repository.",
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved path listing"},
        status.HTTP_404_NOT_FOUND: {"description": "Repository or path not found"},
    },
    tags=["Models"],
)
async def list_repo_tree(
    namespace: str,
    repo: str,
    rev: str,
    path: str = "",
    expand: bool = Query(False, description="If true, return commit data and minimal security info"),
) -> list[TreeEntryModel]:
    """
    List repository directory contents at a specific revision.

    Returns a flat list of file and directory entries for the given path in the repository.
    The output structure matches the Hugging Face Hub repository format but includes only
    essential fields (type, OID, size).

    Args:
        namespace: The namespace/organization for the model.
        repo: The repository name for the model.
        rev: The revision/version to list (e.g., 'main', 'v1.0').
        path: Optional path within the repository. Empty string lists the root directory.
        expand: If True, include commit data and minimal security scanning information.

    Returns:
        List of TreeEntryModel objects containing file/directory type, OID, and size.

    Raises:
        HTTPException: 404 if the repository or path does not exist.
    """
    files = list_repository_files(BASE_FILES_DIR, namespace, repo)

    if not files:
        raise HTTPException(status_code=404, detail="Repository or path not found")
    return [TreeEntryModel(type=f["type"], oid=f["oid"], size=f["size"]) for f in files]


@ai_models_router.get(
    "/api/models/{namespace}/{repo}/revision/{rev}",
    response_model=RevisionModel,
    summary="Get revision metadata",
    description="Retrieve detailed metadata for a specific revision of a model repository.",
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved revision metadata"},
        status.HTTP_404_NOT_FOUND: {"description": "Revision not found"},
    },
    tags=["Models"],
)
async def get_revision(namespace: str, repo: str, rev: str) -> RevisionModel:
    """
    Get detailed metadata for a specific model revision.

    Retrieves the commit SHA, revision name, and associated tags for a specific
    revision of the model repository.

    Args:
        namespace: The namespace/organization for the model.
        repo: The repository name for the model.
        rev: The revision identifier or name (e.g., 'main', 'v1.0').

    Returns:
        RevisionModel: Contains revision ID, name, commit SHA, and tags.

    Raises:
        HTTPException: 404 if the revision does not exist.
    """
    data = metadata_get_revision(namespace, repo, rev)
    return RevisionModel(**data)


@ai_models_router.head(
    "/api/models/{namespace}/{repo}/resolve/{rev}/{path:path}",
    summary="Get file metadata without downloading",
    description="Retrieve file metadata (size, hash, signed URL) without downloading the full file content.",
    responses={
        status.HTTP_200_OK: {"description": "File metadata retrieved successfully"},
        status.HTTP_404_NOT_FOUND: {"description": "File not found"},
    },
    tags=["Models"],
)
async def head_file(namespace: str, repo: str, rev: str, path: str):
    """
    Get file metadata without downloading the file content.

    Returns file metadata including size, content hashes (SHA-256 and MD5), and
    a signed URL for downloading the file. Useful for checking file properties
    without transferring large files.

    Args:
        namespace: The namespace/organization for the model.
        repo: The repository name for the model.
        rev: The revision to fetch from (e.g., 'main').
        path: The file path within the repository.

    Returns:
        Response: Headers include Content-Length, X-Commit-Hash, X-File-Etag, and Location.

    Raises:
        HTTPException: 404 if the file does not exist.
    """
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


@ai_models_router.get(
    "/api/models/{namespace}/{repo}/resolve/{rev}/{path:path}",
    summary="Download model file with redirect",
    description="Retrieve a model file from a specific revision. Returns a redirect to a signed URL for file download.",
    responses={
        status.HTTP_302_FOUND: {"description": "Redirect to signed URL for file download"},
        status.HTTP_404_NOT_FOUND: {"description": "File not found"},
    },
    tags=["Models"],
)
async def get_file(namespace: str, repo: str, rev: str, path: str):
    """
    Download a model file from a specific revision.

    Returns a redirect (HTTP 302) to a signed URL for downloading the requested file.
    In production, this would perform authentication checks and generate a signed URL
    to an S3 bucket. Currently, the signed URL redirects to a local file serving endpoint.

    Args:
        namespace: The namespace/organization for the model.
        repo: The repository name for the model.
        rev: The revision to download from (e.g., 'main').
        path: The file path within the repository to download.

    Returns:
        RedirectResponse: HTTP 302 redirect to the signed URL for file download.

    Raises:
        HTTPException: 404 if the file does not exist.
    """
    print(f"Received request for file: {namespace}/{repo}/{rev}/{path}")
    signed_url = urljoin(
        f"{DOMAIN}/signed-url/",
        f"{namespace}/{repo}/{path}",
    )
    # this redirect is how our service would work. we'd do auth checks, find
    # the file in s3, create a signed URL and return
    print(f"Redirecting to signed URL: {signed_url}")
    return RedirectResponse(url=signed_url, status_code=status.HTTP_302_FOUND)


@ai_models_router.get(
    "/signed-url/{path:path}",
    summary="Stream file content",
    description="Stream file content for download. This endpoint serves files in chunks with proper Content-Length header for large file support.",
    responses={
        status.HTTP_200_OK: {"description": "File content streamed successfully"},
        status.HTTP_404_NOT_FOUND: {"description": "File not found"},
    },
    tags=["Files"],
)
async def signed_url(path: str):
    """
    Stream file content for download.

    Returns file content as a streaming response with proper Content-Length header.
    Files are streamed in 64KB chunks to efficiently handle large files without
    loading them entirely into memory.

    Args:
        path: The file path within the repository (e.g., 'namespace/repo/filename.bin').

    Returns:
        StreamingResponse: File content streamed in chunks with appropriate media type.

    Raises:
        HTTPException: 404 if the file does not exist.
    """
    local_path = get_local_file(BASE_FILES_DIR, path.split("/"))
    file_size = local_path.stat().st_size

    media_type = "application/json" if path.endswith(".json") else "application/octet-stream"

    # yields the file in chunks
    def file_iterator(path: Path, chunk_size: int = FILE_STREAM_CHUNK_SIZE):
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


@ai_models_router.get(
    "/api/models/{namespace}/{repo}/info",
    response_model=RepositoryInfoModel,
    summary="Get model repository information",
    description="Retrieve comprehensive information about a model repository including metadata, files, and revision info.",
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved model information"},
        status.HTTP_404_NOT_FOUND: {"description": "Repository or metadata not found"},
    },
    tags=["Models"],
)
async def get_model_info(namespace: str, repo: str) -> RepositoryInfoModel:
    """
    Get comprehensive information about a model repository.

    Returns detailed information including repository metadata, list of files,
    revision information, security scan status, and aggregate file statistics.

    Args:
        namespace: The namespace/organization for the model.
        repo: The repository name for the model.

    Returns:
        RepositoryInfoModel: Contains repository ID, SHA, ETag, total size, file list,
            metadata, and security status information.

    Raises:
        HTTPException: 404 if the repository, metadata, or revision is not found.
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
        security_status=DEFAULT_SECURITY_FILE_STATUS,
    )


@ai_models_router.get(
    "/api/models",
    response_model=list[RepositoryModel],
    summary="List all model repositories",
    description="Retrieve a list of all available model repositories with basic metadata.",
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved repository list"},
    },
    tags=["Models"],
)
async def list_models() -> list[RepositoryModel]:
    """
    Retrieve all available model repositories.

    Returns a list of all repositories stored in the database, including
    basic information such as namespace, name, description, tags, and creation date.

    Returns:
        List of RepositoryModel objects containing repository information.
    """
    repos = await list_all_repositories()
    return [
        RepositoryModel(
            id=f"{repo.namespace}/{repo.repo}",
            description=repo.description or "",
            tags=repo.tags,
            created_at=repo.created_at,
        )
        for repo in repos
    ]


@ai_models_router.delete(
    "/api/models/{namespace}/{repo}",
    response_model=DeleteModelResponse,
    summary="Delete a model repository",
    description="Delete a model repository including its metadata from the database and files from disk.",
    responses={
        status.HTTP_200_OK: {"description": "Model successfully deleted"},
        status.HTTP_401_UNAUTHORIZED: {"description": "User unauthenticated"},
        status.HTTP_403_FORBIDDEN: {"description": "User does not have access"},
        status.HTTP_404_NOT_FOUND: {"description": "Repository not found"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Failed to delete repository"},
    },
    tags=["Models"],
)
async def delete_model(namespace: str, repo: str, _: None = Depends(verify_authorization)) -> DeleteModelResponse:
    """
    Delete a model repository.

    Removes the repository from the database and deletes all associated files from disk.
    Requires authentication. If file deletion fails, the operation continues but logs a warning.

    Args:
        namespace: The namespace/organization for the model.
        repo: The repository name for the model.
        _: Authentication dependency (automatically validated by FastAPI).

    Returns:
        DeleteModelResponse: Contains status and repository identifier.

    Raises:
        HTTPException: 401 if authorization is invalid, 404 if repository not found,
            500 if database deletion fails.
    """
    # Check if repository exists
    repo_exists_check = await db_repository_exists(namespace, repo)
    if not repo_exists_check:
        raise HTTPException(status_code=404, detail="Repository not found")

    # Delete from database
    deleted_from_db = await delete_repository_metadata(namespace, repo)
    if not deleted_from_db:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete repository {namespace}/{repo} from database",
        )

    # Delete files from disk if they exist
    repo_path = BASE_FILES_DIR / Path(namespace) / Path(repo)
    if repo_path.exists():
        try:
            delete_repository(BASE_FILES_DIR, namespace, repo)
        except Exception as e:
            # Log but don't fail if file deletion fails
            print(f"Warning: Failed to delete repository files: {e}")

    return DeleteModelResponse(status="deleted", repo=f"{namespace}/{repo}")


@ai_models_router.get(
    "/api/models/{namespace}/{repo}/revisions",
    response_model=RevisionListResponseModel,
    summary="List model revisions",
    description="Retrieve all revisions of a model repository from the database.",
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved revision list"},
        status.HTTP_401_UNAUTHORIZED: {"description": "User unauthenticated"},
        status.HTTP_403_FORBIDDEN: {"description": "User does not have access"},
        status.HTTP_404_NOT_FOUND: {"description": "Repository not found or no revisions exist"},
    },
    tags=["Models"],
)
async def list_model_revisions(
    namespace: str, repo: str, _: None = Depends(verify_authorization)
) -> RevisionListResponseModel:
    """
    List all revisions of a model repository.

    Retrieves the complete list of revisions (versions) for a repository,
    including revision names, commit SHAs, and other metadata.

    Args:
        namespace: The namespace/organization for the model.
        repo: The repository name for the model.
        _: Authentication dependency (automatically validated by FastAPI).

    Returns:
        RevisionListResponseModel: Contains repository identifier and list of revisions.

    Raises:
        HTTPException: 401 if authorization is invalid, 404 if repository not found
            or if no revisions exist.
    """

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
