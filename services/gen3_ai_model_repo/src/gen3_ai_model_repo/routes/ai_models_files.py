from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from starlette import status

from gen3_ai_model_repo.auth import verify_authorization
from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.database.file_tracking import (
    delete_file,
    delete_files_for_revision,
    get_file_record,
    list_files_in_revision,
)
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

ai_models_files_router = APIRouter()
REVISION_NOT_FOUND_DETAIL = "Revision not found"
FILE_NOT_FOUND_DETAIL = "File not found"


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
    expand: bool = Query(False, description="If true, return commit data and minimal security info"),
) -> list[TreeEntryModel]:
    """
    List repository directory contents at a specific revision.
    """
    if rev != "main":
        raise HTTPException(
            status_code=400,
            detail="Only 'main' revision is currently supported",
        )

    files = await list_files_in_revision(
        namespace=namespace,
        model_name=repo,
        revision_name=rev,
    )

    if not files:
        raise HTTPException(status_code=404, detail="Repository or path not found")
    if path:
        files = [f for f in files if f["path"].startswith(path)]

    return [TreeEntryModel(type=f["type"], oid=f["oid"], size=f["size"]) for f in files]


@ai_models_files_router.get(
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
    """
    data = await db_get_revision(namespace, repo, rev)
    if not data:
        raise HTTPException(status_code=404, detail=REVISION_NOT_FOUND_DETAIL)
    return RevisionModel(id=str(data["id"]), revision=data["revision"], sha=data["sha"] or "")


@ai_models_files_router.get(
    "/api/models/{namespace}/{repo}/revisions/{revision}",
    response_model=RevisionModel,
    summary="Get revision metadata",
    tags=["Models"],
)
async def get_model_revision(namespace: str, repo: str, revision: str) -> RevisionModel:
    """
    Retrieve revision metadata by revision name.
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
        status.HTTP_200_OK: {"description": "File metadata retrieved successfully"},
        status.HTTP_404_NOT_FOUND: {"description": "File not found"},
    },
    tags=["Models"],
)
async def head_file(namespace: str, repo: str, rev: str, path: str):
    """
    Get file metadata without downloading the file content.
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
    signed_url = await provider.generate_signed_url(file_record["s3_key"])

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
    signed_url = await provider.generate_signed_url(file_record["s3_key"])
    logging.info(f"Redirecting to signed URL: {signed_url}")
    return RedirectResponse(url=signed_url, status_code=status.HTTP_302_FOUND)


@ai_models_files_router.get(
    "/api/models/{namespace}/{repo}/files", response_model=FileListResponseModel, tags=["Models"]
)
async def list_model_files(namespace: str, repo: str, revision: str = "main") -> FileListResponseModel:
    """
    List tracked files for a repository revision.
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
                s3_key=f"{namespace}/{repo}/{revision}/{f['path']}",
            )
            for f in files
        ],
    )


@ai_models_files_router.get(
    "/api/models/{namespace}/{repo}/files/{file_id}", response_model=FileMetadataModel, tags=["Models"]
)
async def get_model_file(namespace: str, repo: str, file_id: str) -> FileMetadataModel:
    """
    Retrieve file metadata from a repository using a file identifier.
    """

    parts = file_id.split(":", 3)
    if len(parts) == 4:
        _, _, revision, path = parts
    else:
        revision, path = "main", file_id
    record = await get_file_record(namespace, repo, revision, path)
    if not record:
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_DETAIL)
    return FileMetadataModel(
        file_id=file_id,
        path=record["path"],
        size=record["size"],
        sha=record["sha"],
        etag=record["etag"],
        s3_key=record["s3_key"],
    )


@ai_models_files_router.delete(
    "/api/models/{namespace}/{repo}/files/{file_id}", response_model=RevisionDeleteResponse, tags=["Models"]
)
async def delete_model_file(
    namespace: str, repo: str, file_id: str, _: None = Depends(verify_authorization)
) -> RevisionDeleteResponse:
    """
    Delete a tracked file from a repository revision.
    """

    parts = file_id.split(":", 3)
    revision, path = ("main", file_id) if len(parts) != 4 else (parts[2], parts[3])
    deleted = await delete_file(namespace, repo, revision, path)
    if not deleted:
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND_DETAIL)
    return RevisionDeleteResponse(status="deleted", repo=f"{namespace}/{repo}", revision=revision)


@ai_models_files_router.delete(
    "/api/models/{namespace}/{repo}/revisions/{revision}",
    response_model=RevisionDeleteResponse,
    tags=["Models"],
)
async def delete_model_revision(
    namespace: str, repo: str, revision: str, _: None = Depends(verify_authorization)
) -> RevisionDeleteResponse:
    """
    Delete a revision and all files tracked under it.
    """

    deleted_files = await delete_files_for_revision(namespace, repo, revision)
    deleted_revision = await delete_revision(namespace, repo, revision)
    if not deleted_revision and not deleted_files:
        raise HTTPException(status_code=404, detail=REVISION_NOT_FOUND_DETAIL)
    return RevisionDeleteResponse(status="deleted", repo=f"{namespace}/{repo}", revision=revision)


@ai_models_files_router.get(
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
    Deprecated local signed-url endpoint.
    """
    raise HTTPException(status_code=410, detail="Local streaming endpoint is deprecated; use storage signed URLs")
