import hashlib
import os
import tempfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from gen3_ai_model_repo.auth import verify_authorization
from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.constants import DEFAULT_SECURITY_FILE_STATUS
from gen3_ai_model_repo.database.helper import (
    create_repository_metadata,
    create_revision,
    delete_file,
    delete_files_for_revision,
    delete_repository_metadata,
    delete_revision,
    get_file_record,
    get_or_create_revision,
    get_repository_metadata,
    list_files_in_revision,
    list_repositories,
    list_revisions,
    track_file,
)
from gen3_ai_model_repo.database.helper import (
    get_revision as db_get_revision,
)
from gen3_ai_model_repo.database.helper import (
    repository_exists as db_repository_exists,
)
from gen3_ai_model_repo.database.revisions import get_revision_identifier_column
from gen3_ai_model_repo.models.schemas import (
    DeleteModelResponse,
    FileListResponseModel,
    FileMetadataModel,
    RepositoryFileModel,
    RepositoryInfoModel,
    RepositoryMetadataModel,
    RepositoryModel,
    RevisionCreateRequest,
    RevisionDeleteResponse,
    RevisionListResponseModel,
    RevisionModel,
    TreeEntryModel,
    UploadUrlRequest,
    UploadUrlResponse,
)
from gen3_ai_model_repo.response import build_head_response
from gen3_ai_model_repo.storage import get_storage_provider

ai_models_router = APIRouter()


class RepositoryCreateRequest(BaseModel):
    """
    Request payload for creating a repository.

    Attributes:
        description: Optional human-readable repository description.
        tags: Optional list of tags used for filtering and discovery.
    """

    description: str | None = None
    tags: list[str] = []


class MultipartUploadResponse(BaseModel):
    """
    Response payload for multipart model uploads.

    Attributes:
        status: Upload status string.
        repo: Repository identifier in namespace/repo form.
        revision: Revision name associated with the upload.
        files: Number of uploaded files.
        total_size: Total uploaded size in bytes.
    """

    status: str
    repo: str
    revision: str
    files: int
    total_size: int


async def repository_has_current_revision(conn) -> bool:
    """
    Check whether the model_repositories table has a current_revision column.

    Args:
        conn: Active database connection.

    Returns:
        True if the current_revision column exists, otherwise False.
    """

    return bool(
        await conn.fetchval(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'model_repositories'
              AND column_name = 'current_revision';
            """
        )
    )


async def model_files_optional_columns(conn) -> tuple[bool, bool]:
    """
    Detect optional columns in the model_files table.

    Args:
        conn: Active database connection.

    Returns:
        Tuple[bool, bool]: Flags for s3_key and file_type column presence.
    """

    rows = await conn.fetch(
        """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'model_files'
                    AND column_name IN ('s3_key', 'file_type');
                """
    )
    column_names = {row["column_name"] for row in rows}
    return "s3_key" in column_names, "file_type" in column_names


@ai_models_router.post("/api/models/{namespace}/{repo}/upload", response_model=MultipartUploadResponse, tags=["Models"])
async def upload_model(
    namespace: str,
    repo: str,
    revision_name: str = Form("main"),
    files: list[UploadFile] = File(...),
    _: None = Depends(verify_authorization),
) -> MultipartUploadResponse:
    """
    Upload one or more files and create a repository revision.

    This endpoint creates the repository and revision records when needed,
    uploads files to the configured storage provider, and tracks metadata in the
    database inside a transaction.

    Args:
        namespace: Namespace/organization for the repository.
        repo: Repository name.
        revision_name: Revision name to create or update.
        files: Uploaded files included in the multipart request.
        _: Authorization dependency validated by FastAPI.

    Returns:
        MultipartUploadResponse with upload summary details.

    Raises:
        HTTPException: 409 if repository already exists, 422 for invalid uploads.
    """

    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required")

    provider = get_storage_provider()
    pool = await __import__("gen3_ai_model_repo.database.db", fromlist=["get_db_pool"]).get_db_pool()
    uploaded_objects: list[str] = []
    total_size = 0

    async with pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            identifier_column = await get_revision_identifier_column(conn)
            has_current_revision = await repository_has_current_revision(conn)
            has_s3_key, has_file_type = await model_files_optional_columns(conn)
            exists = await conn.fetchval(
                "SELECT 1 FROM model_repositories WHERE namespace=$1 AND repo_name=$2",
                namespace,
                repo,
            )
            if exists:
                raise HTTPException(status_code=409, detail=f"Repository {namespace}/{repo} already exists")

            if has_current_revision:
                await conn.execute(
                    "INSERT INTO model_repositories (namespace, repo_name, description, tags, current_revision) VALUES ($1,$2,$3,$4,$5)",
                    namespace,
                    repo,
                    None,
                    [],
                    revision_name,
                )
            else:
                await conn.execute(
                    "INSERT INTO model_repositories (namespace, repo_name, description, tags) VALUES ($1,$2,$3,$4)",
                    namespace,
                    repo,
                    None,
                    [],
                )
            repo_id = await conn.fetchval(
                "SELECT id FROM model_repositories WHERE namespace=$1 AND repo_name=$2",
                namespace,
                repo,
            )
            # Legacy databases may enforce NOT NULL on commit_sha/revision identifier.
            # Seed with a deterministic placeholder and overwrite with final hash after upload.
            initial_revision_hash = hashlib.sha256(f"{namespace}/{repo}:{revision_name}".encode()).hexdigest()
            await conn.execute(
                f"INSERT INTO model_revisions (repository_id, revision_name, {identifier_column}, etag) VALUES ($1,$2,$3,$4)",
                repo_id,
                revision_name,
                initial_revision_hash,
                initial_revision_hash[:32],
            )
            revision_id = await conn.fetchval(
                "SELECT id FROM model_revisions WHERE repository_id=$1 AND revision_name=$2",
                repo_id,
                revision_name,
            )

            file_columns = ["revision_id", "file_path", "file_size", "content_sha", "content_etag"]
            if has_s3_key:
                file_columns.append("s3_key")
            if has_file_type:
                file_columns.append("file_type")
            file_insert_sql = (
                f"INSERT INTO model_files ({', '.join(file_columns)}) "
                f"VALUES ({', '.join(f'${idx}' for idx in range(1, len(file_columns) + 1))})"
            )

            digest = hashlib.sha256()
            for upload in files:
                if not upload.filename:
                    raise HTTPException(status_code=422, detail="Each uploaded file must have a filename")
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp_path = tmp.name
                    while chunk := await upload.read(1024 * 1024):
                        tmp.write(chunk)
                        digest.update(chunk)
                        total_size += len(chunk)
                object_key = f"{namespace}/{repo}/{revision_name}/{upload.filename}"
                await provider.upload_file(tmp_path, object_key)
                os.unlink(tmp_path)
                meta = await provider.get_file_metadata(object_key)
                sha = digest.hexdigest()
                etag = meta.get("etag") or sha[:32]
                file_values: list[object] = [
                    revision_id,
                    upload.filename,
                    int(meta["size"]),
                    sha,
                    etag,
                ]
                if has_s3_key:
                    file_values.append(object_key)
                if has_file_type:
                    file_values.append(upload.content_type)
                await conn.execute(file_insert_sql, *file_values)
                uploaded_objects.append(object_key)

            revision_hash = hashlib.sha256("".join(sorted(upload.filename for upload in files)).encode()).hexdigest()
            await conn.execute(
                f"UPDATE model_revisions SET {identifier_column}=$1, etag=$2 WHERE id=$3",
                revision_hash,
                revision_hash[:32],
                revision_id,
            )
            if has_current_revision:
                await conn.execute(
                    "UPDATE model_repositories SET current_revision=$1 WHERE id=$2",
                    revision_name,
                    repo_id,
                )
            await tx.commit()
        except Exception:
            await tx.rollback()
            for object_key in uploaded_objects:
                try:
                    await provider.delete_file(object_key)
                except Exception:
                    pass
            raise

    return MultipartUploadResponse(
        status="uploaded",
        repo=f"{namespace}/{repo}",
        revision=revision_name,
        files=len(files),
        total_size=total_size,
    )


@ai_models_router.post(
    "/api/models/{namespace}/{repo}/revisions", response_model=RevisionModel, summary="Create revision", tags=["Models"]
)
async def create_model_revision(
    namespace: str,
    repo: str,
    request: RevisionCreateRequest,
    _: None = Depends(verify_authorization),
) -> RevisionModel:
    """
    Create a revision for an existing repository.

    Args:
        namespace: Namespace/organization for the repository.
        repo: Repository name.
        request: Revision creation payload.
        _: Authorization dependency validated by FastAPI.

    Returns:
        RevisionModel containing created revision metadata.

    Raises:
        HTTPException: 404 if the repository does not exist.
    """

    revision = await create_revision(namespace, repo, request.revision_name, request.revision_identifier, request.etag)
    if not revision:
        raise HTTPException(status_code=404, detail="Repository not found")
    return RevisionModel(**{"id": str(revision["id"]), "revision": revision["revision"], "sha": revision["sha"] or ""})


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
    if rev != "main":
        raise HTTPException(
            status_code=400,
            detail="Only 'main' revision is currently supported",
        )

    files = await list_files_in_revision(
        namespace=namespace,
        repo_name=repo,
        revision_name=rev,
    )

    if not files:
        raise HTTPException(status_code=404, detail="Repository or path not found")
    if path:
        files = [f for f in files if f["path"].startswith(path)]

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
    data = await db_get_revision(namespace, repo, rev)
    if not data:
        raise HTTPException(status_code=404, detail="Revision not found")
    return RevisionModel(id=str(data["id"]), revision=data["revision"], sha=data["sha"] or "")


@ai_models_router.get(
    "/api/models/{namespace}/{repo}/revisions/{revision}",
    response_model=RevisionModel,
    summary="Get revision metadata",
    tags=["Models"],
)
async def get_model_revision(namespace: str, repo: str, revision: str) -> RevisionModel:
    """
    Retrieve revision metadata by revision name.

    Args:
        namespace: Namespace/organization for the repository.
        repo: Repository name.
        revision: Revision name to retrieve.

    Returns:
        RevisionModel containing revision metadata.

    Raises:
        HTTPException: 404 if the revision does not exist.
    """

    data = await db_get_revision(namespace, repo, revision)
    if not data:
        raise HTTPException(status_code=404, detail="Revision not found")
    return RevisionModel(id=str(data["id"]), revision=data["revision"], sha=data["sha"] or "")


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
    file_record = await get_file_record(
        namespace=namespace,
        repo_name=repo,
        revision_name=rev,
        file_path=path,
    )

    if not file_record:
        raise HTTPException(
            status_code=404,
            detail="File not found",
        )

    size = file_record["size"]
    commit_hash = file_record["sha"]
    etag = file_record["etag"]

    provider = get_storage_provider()
    signed_url = await provider.generate_signed_url(file_record["s3_key"])

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
    logging.info(f"Received request for file: {namespace}/{repo}/{rev}/{path}")
    file_record = await get_file_record(
        namespace=namespace,
        repo_name=repo,
        revision_name=rev,
        file_path=path,
    )
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")

    provider = get_storage_provider()
    signed_url = await provider.generate_signed_url(file_record["s3_key"])
    logging.info(f"Redirecting to signed URL: {signed_url}")
    return RedirectResponse(url=signed_url, status_code=status.HTTP_302_FOUND)


@ai_models_router.get("/api/models/{namespace}/{repo}/files", response_model=FileListResponseModel, tags=["Models"])
async def list_model_files(namespace: str, repo: str, revision: str = "main") -> FileListResponseModel:
    """
    List tracked files for a repository revision.

    Args:
        namespace: Namespace/organization for the repository.
        repo: Repository name.
        revision: Revision name to list files from.

    Returns:
        FileListResponseModel containing file metadata entries.
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


@ai_models_router.get(
    "/api/models/{namespace}/{repo}/files/{file_id}", response_model=FileMetadataModel, tags=["Models"]
)
async def get_model_file(namespace: str, repo: str, file_id: str) -> FileMetadataModel:
    """
    Retrieve file metadata from a repository using a file identifier.

    Args:
        namespace: Namespace/organization for the repository.
        repo: Repository name.
        file_id: Encoded file identifier or plain file path.

    Returns:
        FileMetadataModel with tracked file details.

    Raises:
        HTTPException: 404 if the file cannot be found.
    """

    parts = file_id.split(":", 3)
    if len(parts) == 4:
        _, _, revision, path = parts
    else:
        revision, path = "main", file_id
    record = await get_file_record(namespace, repo, revision, path)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    return FileMetadataModel(
        file_id=file_id,
        path=record["path"],
        size=record["size"],
        sha=record["sha"],
        etag=record["etag"],
        s3_key=record["s3_key"],
    )


@ai_models_router.delete(
    "/api/models/{namespace}/{repo}/files/{file_id}", response_model=RevisionDeleteResponse, tags=["Models"]
)
async def delete_model_file(
    namespace: str, repo: str, file_id: str, _: None = Depends(verify_authorization)
) -> RevisionDeleteResponse:
    """
    Delete a tracked file from a repository revision.

    Args:
        namespace: Namespace/organization for the repository.
        repo: Repository name.
        file_id: Encoded file identifier or plain file path.
        _: Authorization dependency validated by FastAPI.

    Returns:
        RevisionDeleteResponse describing deletion outcome.

    Raises:
        HTTPException: 404 if the file cannot be found.
    """

    parts = file_id.split(":", 3)
    revision, path = ("main", file_id) if len(parts) != 4 else (parts[2], parts[3])
    deleted = await delete_file(namespace, repo, revision, path)
    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")
    return RevisionDeleteResponse(status="deleted", repo=f"{namespace}/{repo}", revision=revision)


@ai_models_router.delete(
    "/api/models/{namespace}/{repo}/revisions/{revision}", response_model=RevisionDeleteResponse, tags=["Models"]
)
async def delete_model_revision(
    namespace: str, repo: str, revision: str, _: None = Depends(verify_authorization)
) -> RevisionDeleteResponse:
    """
    Delete a revision and all files tracked under it.

    Args:
        namespace: Namespace/organization for the repository.
        repo: Repository name.
        revision: Revision name to delete.
        _: Authorization dependency validated by FastAPI.

    Returns:
        RevisionDeleteResponse describing deletion outcome.

    Raises:
        HTTPException: 404 if the revision and its files are not found.
    """

    deleted_files = await delete_files_for_revision(namespace, repo, revision)
    deleted_revision = await delete_revision(namespace, repo, revision)
    if not deleted_revision and not deleted_files:
        raise HTTPException(status_code=404, detail="Revision not found")
    return RevisionDeleteResponse(status="deleted", repo=f"{namespace}/{repo}", revision=revision)


@ai_models_router.post("/api/models/{namespace}/{repo}/upload-url", response_model=UploadUrlResponse, tags=["Models"])
async def generate_upload_url(
    namespace: str, repo: str, request: UploadUrlRequest, _: None = Depends(verify_authorization)
) -> UploadUrlResponse:
    """
    Generate a storage upload URL for a file in a revision.

    Args:
        namespace: Namespace/organization for the repository.
        repo: Repository name.
        request: Upload URL request payload.
        _: Authorization dependency validated by FastAPI.

    Returns:
        UploadUrlResponse containing a pre-signed upload URL and object key.
    """

    provider = get_storage_provider()
    object_key = f"{namespace}/{repo}/{request.revision_name}/{request.file_name}"
    upload_url = await provider.generate_upload_url(object_key)
    return UploadUrlResponse(upload_url=upload_url, object_key=object_key)


@ai_models_router.post("/api/models/{namespace}/{repo}/complete-upload", response_model=RevisionModel, tags=["Models"])
async def complete_upload(
    namespace: str, repo: str, request: RevisionCreateRequest, _: None = Depends(verify_authorization)
) -> RevisionModel:
    """
    Finalize an upload by creating/updating revision and file tracking records.

    Args:
        namespace: Namespace/organization for the repository.
        repo: Repository name.
        request: Revision completion payload.
        _: Authorization dependency validated by FastAPI.

    Returns:
        RevisionModel containing finalized revision metadata.

    Raises:
        HTTPException: 404 if the repository does not exist.
    """

    provider = get_storage_provider()
    storage_prefix = f"{namespace}/{repo}/{request.revision_name}"
    object_keys = await provider.list_files(storage_prefix)

    derived_revision_identifier = request.revision_identifier
    if not derived_revision_identifier:
        digest = hashlib.sha256()
        for object_key in sorted(object_keys):
            digest.update(object_key.encode())
        # Keep revision identifiers non-null for legacy schemas that enforce commit_sha NOT NULL.
        derived_revision_identifier = digest.hexdigest()

    revision = await create_revision(
        namespace,
        repo,
        request.revision_name,
        derived_revision_identifier,
        request.etag or derived_revision_identifier[:32],
    )
    if not revision:
        raise HTTPException(status_code=404, detail="Repository not found")

    prefix_with_slash = f"{storage_prefix}/"
    for object_key in object_keys:
        if not object_key.startswith(prefix_with_slash):
            continue
        file_path = object_key[len(prefix_with_slash) :]
        if not file_path:
            continue

        metadata = await provider.get_file_metadata(object_key)
        content_etag = metadata.get("etag")
        # Keep content SHA stable and non-null for schemas that require it.
        content_sha = hashlib.sha256(f"{object_key}:{content_etag or ''}".encode()).hexdigest()

        await track_file(
            namespace=namespace,
            repo_name=repo,
            revision_name=request.revision_name,
            file_path=file_path,
            file_size=int(metadata["size"]),
            content_sha=content_sha,
            content_etag=content_etag,
            s3_key=object_key,
        )

    return RevisionModel(id=str(revision["id"]), revision=revision["revision"], sha=revision["sha"] or "")


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
    Deprecated local signed-url endpoint.

    This route is intentionally disabled and returns HTTP 410.

    Args:
        path: The file path within the repository (e.g., 'namespace/repo/filename.bin').

    Returns:
        Never returns file content; always raises an exception.

    Raises:
        HTTPException: 410 because local streaming is deprecated.
    """
    raise HTTPException(status_code=410, detail="Local streaming endpoint is deprecated; use storage signed URLs")


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
async def list_models(
    namespace: str | None = Query(None),
    tags: list[str] | None = Query(None),
    search: str | None = Query(None),
) -> list[RepositoryModel]:
    """
    Retrieve all available model repositories.

    Returns a list of all repositories stored in the database, including
    basic information such as namespace, name, description, tags, and creation date.

    Returns:
        List of RepositoryModel objects containing repository information.
    """
    repos = await list_repositories(namespace=namespace, tags=tags, search=search)
    return [
        RepositoryModel(
            id=f"{repo.namespace}/{repo.repo}",
            description=repo.description or "",
            tags=repo.tags,
            created_at=repo.created_at,
        )
        for repo in repos
    ]


@ai_models_router.get("/api/models/{namespace}/{repo}", response_model=RepositoryInfoModel, tags=["Models"])
async def get_repository(namespace: str, repo: str) -> RepositoryInfoModel:
    """
    Retrieve repository information, main revision metadata, and tracked files.

    Args:
        namespace: Namespace/organization for the repository.
        repo: Repository name.

    Returns:
        RepositoryInfoModel with repository metadata, revision summary, and files.

    Raises:
        HTTPException: 404 if repository metadata is not found.
    """

    metadata = await get_repository_metadata(namespace, repo)
    if not metadata:
        raise HTTPException(status_code=404, detail="Repository not found")
    revision_info = await db_get_revision(namespace, repo, metadata.repo and "main")
    files_from_db = (
        await list_files_in_revision(namespace=namespace, repo_name=repo, revision_name="main") if revision_info else []
    )
    return RepositoryInfoModel(
        id=f"{namespace}/{repo}",
        sha=(revision_info["sha"] or "") if revision_info else "",
        etag=(revision_info["etag"] or revision_info["sha"] or "") if revision_info else "",
        size=sum(f["size"] for f in files_from_db),
        files=[RepositoryFileModel(type=f["type"], oid=f["oid"], size=f["size"]) for f in files_from_db],
        metadata=metadata,
        security_status=DEFAULT_SECURITY_FILE_STATUS,
    )


@ai_models_router.post("/api/models/{namespace}/{repo}", response_model=RepositoryMetadataModel, tags=["Models"])
async def create_repository(
    namespace: str,
    repo: str,
    request: RepositoryCreateRequest,
    _: None = Depends(verify_authorization),
) -> RepositoryMetadataModel:
    """
    Create repository metadata for a new repository.

    Args:
        namespace: Namespace/organization for the repository.
        repo: Repository name.
        request: Repository metadata creation payload.
        _: Authorization dependency validated by FastAPI.

    Returns:
        RepositoryMetadataModel for the newly created repository.

    Raises:
        HTTPException: 409 if the repository already exists.
    """

    if await db_repository_exists(namespace, repo):
        raise HTTPException(status_code=409, detail=f"Repository {namespace}/{repo} already exists")

    return await create_repository_metadata(
        namespace=namespace,
        repo_name=repo,
        description=request.description,
        tags=request.tags,
    )


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

    # Delete S3 objects
    provider = get_storage_provider()
    prefix = f"{namespace}/{repo}/"
    await provider.delete_prefix(prefix)

    # Delete from database
    deleted_from_db = await delete_repository_metadata(namespace, repo)
    if not deleted_from_db:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete repository {namespace}/{repo} from database",
        )

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
