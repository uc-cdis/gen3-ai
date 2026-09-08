"""File routes for the Gen3 AI model repo service."""

from fastapi import Depends, HTTPException
from fastapi.responses import RedirectResponse
from starlette import status

from gen3_ai_model_repo.auth import AuthorizedRouter, verify_authorization
from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.database.file_tracking import (
    delete_file,
    delete_files_for_revision,
    get_file_record,
    list_files_in_revision,
)
from gen3_ai_model_repo.database.repo_metadata import model_exists as db_model_exists
from gen3_ai_model_repo.database.revisions import delete_revision
from gen3_ai_model_repo.database.revisions import get_revision as db_get_revision
from gen3_ai_model_repo.models.schemas import (
    FileListResponseModel,
    FileMetadataModel,
    RevisionDeleteResponse,
    RevisionModel,
    TreeEntryModel,
)
from gen3_ai_model_repo.response import build_head_response
from gen3_ai_model_repo.storage.helpers import get_storage_provider

ai_models_files_router = AuthorizedRouter(dependencies=[Depends(verify_authorization)])
REVISION_NOT_FOUND_DETAIL = "Revision not found"
FILE_NOT_FOUND_DETAIL = "File not found"
INVALID_FILE_ID_DETAIL = "file_id must be namespace:repo:revision:path for the requested repository"


def _parse_file_id(namespace: str, repo: str, file_id: str) -> tuple[str, str]:
    """
    Validate and split a composite file ID.

    Returns:
        A tuple containing the revision and file path.

    Raises:
        HTTPException: If the ID is malformed or targets another repository.
    """
    parts = file_id.split(":", 3)
    if len(parts) != 4 or not all(parts):
        raise HTTPException(status_code=422, detail=INVALID_FILE_ID_DETAIL)

    file_namespace, file_repo, revision, path = parts
    if file_namespace != namespace or file_repo != repo:
        raise HTTPException(status_code=422, detail=INVALID_FILE_ID_DETAIL)

    return revision, path


@ai_models_files_router.get(
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
@ai_models_files_router.get(
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
) -> list[TreeEntryModel]:
    """
    List repository directory contents at a specific revision.

    Returns:
        list[TreeEntryModel]: A list of tree entry models for the directory contents.

    Raises:
        HTTPException: If the repository is not found.
    """
    repo_exists = await db_model_exists(namespace, repo)
    if not repo_exists:
        raise HTTPException(status_code=404, detail="Repository not found")

    files = await list_files_in_revision(
        namespace=namespace,
        model_name=repo,
        revision_name=rev,
    )

    if path:
        prefix = path.rstrip("/") + "/"
        files = [f for f in files if f["path"] == path or f["path"].startswith(prefix)]

    return [TreeEntryModel(type=f["type"], oid=f["oid"], size=f["size"], path=f["path"]) for f in files]


@ai_models_files_router.get(
    "/api/models/{namespace}/{repo}/revisions/{revision}",
    response_model=RevisionModel,
    summary="Get revision metadata",
    description="Retrieve detailed metadata for a specific revision of a model repository including commit hash and ETag.",
    tags=["Models"],
)
async def get_model_revision(namespace: str, repo: str, revision: str) -> RevisionModel:
    """
    Retrieve revision metadata by revision name.

    Returns:
        RevisionModel: The revision metadata model.

    Raises:
        HTTPException: If the revision is not found.
    """

    data = await db_get_revision(namespace, repo, revision)
    if not data:
        raise HTTPException(status_code=404, detail=REVISION_NOT_FOUND_DETAIL)
    return RevisionModel(id=str(data["id"]), revision=data["revision"], sha=data["sha"] or "")


@ai_models_files_router.head(
    "/api/models/{namespace}/{repo}/resolve/{rev}/{path:path}",
    summary="Get file metadata without downloading",
    description="Retrieve file metadata (size, hash, signed URL) without downloading the full file content.",
    responses={
        status.HTTP_302_FOUND: {"description": "Redirect to signed URL with metadata headers"},
        status.HTTP_404_NOT_FOUND: {"description": "File not found"},
    },
    tags=["Models"],
)
async def head_file(namespace: str, repo: str, rev: str, path: str):
    """
    Get file metadata without downloading the file content.

    Returns:
        RedirectResponse: A redirect response with file metadata headers.

    Raises:
        HTTPException: If the file is not found.
    """

    file_record = await get_file_record(
        namespace=namespace,
        model_name=repo,
        revision_name=rev,
        file_path=path,
    )

    if not file_record:
        raise HTTPException(
            status_code=404,
            detail=FILE_NOT_FOUND_DETAIL,
        )

    size = file_record["size"]
    commit_hash = file_record["sha"]
    etag = file_record["etag"]

    provider = get_storage_provider()
    signed_url = await provider.generate_signed_url(file_record["object_key"])

    return build_head_response(commit_hash, etag, size, signed_url)


@ai_models_files_router.get(
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

    Returns:
        RedirectResponse: A redirect response with signed URL to download the file.

    Raises:
        HTTPException: If the file is not found.
    """
    logging.info(f"Received request for file: {namespace}/{repo}/{rev}/{path}")
    file_record = await get_file_record(
        namespace=namespace,
        model_name=repo,
        revision_name=rev,
        file_path=path,
    )
    if not file_record:
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_DETAIL)

    provider = get_storage_provider()
    signed_url = await provider.generate_signed_url(file_record["object_key"])
    logging.info(f"Redirecting to signed URL for {file_record['object_key']}")
    return RedirectResponse(url=signed_url, status_code=status.HTTP_302_FOUND)


@ai_models_files_router.get(
    "/api/models/{namespace}/{repo}/files",
    response_model=FileListResponseModel,
    summary="List model files",
    description="List all tracked files in a repository revision with their metadata including size and content hashes.",
    tags=["Models"],
)
async def list_model_files(namespace: str, repo: str, revision: str = "main") -> FileListResponseModel:
    """
    List tracked files for a repository revision.

    Returns:
        FileListResponseModel: A model containing the list of files.
    """

    files = await list_files_in_revision(namespace, repo, revision)
    return FileListResponseModel(
        repo=f"{namespace}/{repo}",
        files=[
            FileMetadataModel(
                file_id=f"{namespace}:{repo}:{revision}:{f['path']}",
                path=f["path"],
                size=f["size"],
                sha=f["oid"],
                etag=f["etag"],
                object_key=f"{namespace}/{repo}/{revision}/{f['path']}",
            )
            for f in files
        ],
    )


@ai_models_files_router.get(
    "/api/models/{namespace}/{repo}/files/{file_id}",
    response_model=FileMetadataModel,
    summary="Get file metadata",
    description="Retrieve metadata for a specific file in a repository including size, content hash, and storage location.",
    tags=["Models"],
)
async def get_model_file(namespace: str, repo: str, file_id: str) -> FileMetadataModel:
    """
    Retrieve file metadata from a repository using a file identifier.

    Returns:
        FileMetadataModel: The file metadata model.

    Raises:
        HTTPException: If the file or repository is not found.
    """

    revision, path = _parse_file_id(namespace, repo, file_id)
    record = await get_file_record(namespace, repo, revision, path)
    if not record:
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_DETAIL)
    return FileMetadataModel(
        file_id=file_id,
        path=record["path"],
        size=record["size"],
        sha=record["sha"],
        etag=record["etag"],
        object_key=record["object_key"],
    )


@ai_models_files_router.delete(
    "/api/models/{namespace}/{repo}/files/{file_id}",
    response_model=RevisionDeleteResponse,
    summary="Delete a file",
    description="Delete a tracked file from a repository revision. This removes the file tracking record but may not immediately delete storage.",
    tags=["Models"],
)
async def delete_model_file(namespace: str, repo: str, file_id: str) -> RevisionDeleteResponse:
    """
    Delete a tracked file from a repository revision.

    Returns:
        RevisionDeleteResponse: Response indicating successful deletion.

    Raises:
        HTTPException: If the file or repository is not found.
    """

    revision, path = _parse_file_id(namespace, repo, file_id)
    deleted = await delete_file(namespace, repo, revision, path)
    if not deleted:
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_DETAIL)
    return RevisionDeleteResponse(status="deleted", repo=f"{namespace}/{repo}", revision=revision)


@ai_models_files_router.delete(
    "/api/models/{namespace}/{repo}/revisions/{revision}",
    response_model=RevisionDeleteResponse,
    summary="Delete a revision",
    description="Delete a specific revision and all files tracked under it from the repository.",
    tags=["Models"],
)
async def delete_model_revision(namespace: str, repo: str, revision: str) -> RevisionDeleteResponse:
    """
    Delete a revision and all files tracked under it.

    Returns:
        RevisionDeleteResponse: Response indicating successful deletion.

    Raises:
        HTTPException: If the revision or repository is not found.
    """

    deleted_files = await delete_files_for_revision(namespace, repo, revision)
    deleted_revision = await delete_revision(namespace, repo, revision)
    if not deleted_revision and not deleted_files:
        raise HTTPException(status_code=404, detail=REVISION_NOT_FOUND_DETAIL)
    return RevisionDeleteResponse(status="deleted", repo=f"{namespace}/{repo}", revision=revision)
