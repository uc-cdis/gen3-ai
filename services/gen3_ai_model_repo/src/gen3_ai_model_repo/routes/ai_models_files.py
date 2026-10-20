"""File routes for the Gen3 AI model repo service."""

from fastapi import Depends, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse
from starlette import status

from gen3_ai_model_repo.auth import AuthorizedRouter, verify_authorization
from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.database.file_tracking import (
    delete_file,
    delete_files_for_revision,
    get_file_record,
    get_storage_keys_for_revision,
    list_files_in_revision,
    storage_key_is_referenced,
)
from gen3_ai_model_repo.database.repo_metadata import get_model_metadata
from gen3_ai_model_repo.database.repo_metadata import model_exists as db_model_exists
from gen3_ai_model_repo.database.revisions import delete_revision
from gen3_ai_model_repo.database.revisions import get_revision as db_get_revision
from gen3_ai_model_repo.models.schemas import (
    FileListResponseModel,
    FileMetadataModel,
    RevisionDeleteResponse,
    RevisionInfoModel,
    TreeEntryModel,
)
from gen3_ai_model_repo.response import build_file_metadata_headers, build_head_response
from gen3_ai_model_repo.routes.ai_models_shared import serialize_huggingface_siblings, validate_repository_path
from gen3_ai_model_repo.storage.helpers import get_storage_provider

ai_models_files_router = AuthorizedRouter(
    dependencies=[Depends(verify_authorization), Depends(validate_repository_path)]
)
REVISION_NOT_FOUND_DETAIL = "Revision not found"
FILE_NOT_FOUND_DETAIL = "File not found"
INVALID_FILE_ID_DETAIL = "file_id must be namespace:repo:revision:path for the requested repository"


def _local_file_response(local_path, file_record):
    """
    Build the local response used by both GET and HEAD resolve routes.

    Returns:
        FileResponse: A response that lets Starlette derive file metadata and
        suppress the body automatically for HEAD requests.
    """
    return FileResponse(
        local_path,
        media_type="application/octet-stream",
        headers=build_file_metadata_headers(file_record["sha"], file_record["etag"] or "", file_record["size"]),
    )


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
    recursive: bool = Query(False),
    expand: bool = Query(False),
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

    revision = await db_get_revision(namespace, repo, rev)
    if not revision:
        raise HTTPException(status_code=404, detail=REVISION_NOT_FOUND_DETAIL)

    files = await list_files_in_revision(
        namespace=namespace,
        model_name=repo,
        revision_name=revision["revision"],
    )

    if path:
        prefix = path.rstrip("/") + "/"
        files = [f for f in files if f["path"] == path or f["path"].startswith(prefix)]

    if not recursive:
        prefix = path.rstrip("/") + "/" if path else ""
        immediate: dict[str, dict] = {}
        for file_record in files:
            relative = (
                file_record["path"][len(prefix) :]
                if prefix and file_record["path"].startswith(prefix)
                else file_record["path"]
            )
            if prefix and not file_record["path"].startswith(prefix):
                continue
            first = relative.split("/", 1)[0]
            if "/" in relative:
                immediate.setdefault(first, {"type": "directory", "oid": None, "size": 0, "path": prefix + first})
            else:
                immediate[first] = file_record
        files = list(immediate.values())

    return [TreeEntryModel(type=f["type"], oid=f.get("oid"), size=f["size"], path=f["path"]) for f in files]


@ai_models_files_router.get(
    "/api/models/{namespace}/{repo}/revisions/{revision}",
    response_model=RevisionInfoModel,
    summary="Get revision metadata",
    description="Retrieve detailed metadata for a specific revision of a model repository including commit hash and ETag.",
    tags=["Models"],
)
@ai_models_files_router.get(
    "/api/models/{namespace}/{repo}/revision/{revision}",
    description="Retrieve metadata for a repository revision.",
)
async def get_model_revision(namespace: str, repo: str, revision: str) -> RevisionInfoModel:
    """
    Retrieve revision metadata by revision name.

    Returns:
        RevisionInfoModel: Hugging Face-compatible revision metadata.

    Raises:
        HTTPException: If the revision is not found.
    """

    data = await db_get_revision(namespace, repo, revision)
    if not data:
        raise HTTPException(status_code=404, detail=REVISION_NOT_FOUND_DETAIL)
    files = await list_files_in_revision(namespace, repo, revision)
    siblings = serialize_huggingface_siblings(files)
    metadata = await get_model_metadata(namespace, repo)
    if not metadata:
        raise HTTPException(status_code=404, detail="Repository not found")
    return RevisionInfoModel(
        id=f"{namespace}/{repo}",
        sha=data["sha"] or "",
        etag=data.get("etag") or data["sha"] or "",
        size=sum(item["size"] for item in files),
        files=[
            TreeEntryModel(type=item["type"], oid=item["oid"], size=item["size"], path=item["path"]) for item in files
        ],
        siblings=siblings,
        metadata=metadata,
        security_status={},
    )


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
@ai_models_files_router.head(
    "/{namespace}/{repo}/resolve/{rev}/{path:path}",
    description="Retrieve file metadata without downloading file content.",
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
    local_path = await provider.get_file_path(file_record["object_key"])
    if local_path is not None:
        if not local_path.is_file():
            raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_DETAIL)
        return _local_file_response(local_path, file_record)
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
@ai_models_files_router.get(
    "/{namespace}/{repo}/resolve/{rev}/{path:path}",
    description="Download a model file from a repository revision.",
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
    local_path = await provider.get_file_path(file_record["object_key"])
    if local_path is not None:
        if not local_path.is_file():
            raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_DETAIL)
        return _local_file_response(local_path, file_record)
    signed_url = await provider.generate_signed_url(file_record["object_key"])
    logging.info(f"Redirecting to signed URL for {file_record['object_key']}")
    return RedirectResponse(
        url=signed_url,
        status_code=status.HTTP_302_FOUND,
        headers=build_file_metadata_headers(file_record["sha"], file_record["etag"] or "", file_record["size"]),
    )


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
    record = await get_file_record(namespace, repo, revision, path)
    if not record:
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_DETAIL)
    deleted = await delete_file(namespace, repo, revision, path)
    if not deleted:
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_DETAIL)
    object_key = record["object_key"]
    if object_key and not await storage_key_is_referenced(object_key):
        try:
            await get_storage_provider().delete_file(object_key)
        except Exception:
            logging.exception("File metadata deleted but storage cleanup failed", extra={"object_key": object_key})
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

    storage_keys = await get_storage_keys_for_revision(namespace, repo, revision)
    deleted_files = await delete_files_for_revision(namespace, repo, revision)
    deleted_revision = await delete_revision(namespace, repo, revision)
    if not deleted_revision and not deleted_files:
        raise HTTPException(status_code=404, detail=REVISION_NOT_FOUND_DETAIL)
    provider = get_storage_provider()
    for object_key in storage_keys:
        if await storage_key_is_referenced(object_key):
            continue
        try:
            await provider.delete_file(object_key)
        except Exception:
            logging.exception(
                "Revision metadata deleted but storage cleanup failed",
                extra={"namespace": namespace, "repo": repo, "revision": revision, "object_key": object_key},
            )
    return RevisionDeleteResponse(status="deleted", repo=f"{namespace}/{repo}", revision=revision)
