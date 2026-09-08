"""Upload routes for the Gen3 AI model repo service."""

import hashlib

from fastapi import Depends, File, Form, HTTPException, Request, UploadFile

from gen3_ai_model_repo import config
from gen3_ai_model_repo.auth import AuthorizedRouter, verify_authorization
from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.database.db import get_db_pool
from gen3_ai_model_repo.database.file_tracking import track_file
from gen3_ai_model_repo.database.revisions import create_revision
from gen3_ai_model_repo.models.schemas import RevisionCreateRequest, RevisionModel, UploadUrlRequest, UploadUrlResponse
from gen3_ai_model_repo.routes.ai_models_shared import MultipartUploadResponse
from gen3_ai_model_repo.storage.helpers import get_storage_provider
from gen3_ai_model_repo.storage.keys import build_object_key, validate_key_component

ai_models_uploads_router = AuthorizedRouter(dependencies=[Depends(verify_authorization)])


def _build_object_key(namespace: str, repo: str, revision_name: str, filename: str) -> str:
    """
    Build a storage key and translate invalid client input to HTTP 422.

    Returns:
        The validated storage object key.

    Raises:
        HTTPException: If any key component or the filename is invalid.
    """
    try:
        return build_object_key(namespace, repo, revision_name, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _validate_request_components(namespace: str, repo: str, revision_name: str) -> None:
    """
    Validate storage-key components shared by upload operations.

    Raises:
        HTTPException: If any component is invalid.
    """
    try:
        validate_key_component(namespace, "namespace")
        validate_key_component(repo, "repo")
        validate_key_component(revision_name, "revision")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _hash_upload_file(upload: UploadFile) -> tuple[str, str, int]:
    """
    Compute SHA256 and MD5 digests for an uploaded file without loading it into memory.

    Returns:
        tuple[str, str, int]: SHA256 digest, MD5 digest, and file size in bytes.
    """
    await upload.seek(0)
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    total_size = 0

    while chunk := await upload.read(1024 * 1024):
        sha256.update(chunk)
        md5.update(chunk)
        total_size += len(chunk)

    await upload.seek(0)
    return sha256.hexdigest(), md5.hexdigest(), total_size


async def _create_model_and_initial_revision(
    conn,
    namespace: str,
    repo: str,
    revision_name: str,
) -> tuple[int, int]:
    """
    Create repository and placeholder revision records.

    Returns:
        tuple[int, int]: Model ID and revision ID.

    Raises:
        HTTPException: If the repository already exists.
    """
    exists_stmt = await conn.prepare("SELECT 1 FROM models WHERE namespace=$1 AND model_name=$2")
    exists = await exists_stmt.fetchval(namespace, repo)
    if exists:
        raise HTTPException(status_code=409, detail=f"Repository {namespace}/{repo} already exists")

    insert_repo_stmt = await conn.prepare(
        "INSERT INTO models (namespace, model_name, description, tags, current_revision) VALUES ($1,$2,$3,$4,$5)"
    )
    await insert_repo_stmt.fetch(namespace, repo, None, [], revision_name)

    model_id_stmt = await conn.prepare("SELECT id FROM models WHERE namespace=$1 AND model_name=$2")
    model_id = await model_id_stmt.fetchval(namespace, repo)

    initial_revision_hash = hashlib.sha256(f"{namespace}/{repo}:{revision_name}".encode()).hexdigest()
    insert_revision_stmt = await conn.prepare(
        "INSERT INTO model_revisions (model_id, revision_name, revision_identifier, etag) VALUES ($1,$2,$3,$4)"
    )
    await insert_revision_stmt.fetch(model_id, revision_name, initial_revision_hash, initial_revision_hash[:32])

    revision_id_stmt = await conn.prepare("SELECT id FROM model_revisions WHERE model_id=$1 AND revision_name=$2")
    revision_id = await revision_id_stmt.fetchval(model_id, revision_name)

    return model_id, revision_id


async def _prepare_file_insert_stmt(conn):
    """
    Build and prepare the INSERT statement for model files.

    Returns:
        asyncpg.connection.PreparedStatement: The prepared SQL statement for file insertion.
    """
    return await conn.prepare(
        """
        INSERT INTO model_files (revision_id, file_path, file_size, content_sha, content_etag, s3_key, file_type)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
    )


async def _process_uploaded_files(
    files: list[UploadFile],
    provider,
    namespace: str,
    repo: str,
    revision_name: str,
) -> tuple[list[str], list[tuple[str, str, str, str, int, str | None]], int]:
    """
    Upload files to object storage and collect file metadata for persistence.

    Returns:
        tuple[list[str], list[tuple[str, str, str, str, int, str | None]], int]:
            Object keys, file metadata tuples, and total file size.

    Raises:
        HTTPException: If a file is missing a filename.
    """
    uploaded_objects: list[str] = []
    uploaded_file_records: list[tuple[str, str, str, str, int, str | None]] = []
    total_size = 0

    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=422, detail="Each uploaded file must have a filename")

        file_sha, file_md5, file_size = await _hash_upload_file(upload)
        if file_size > config.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded file exceeds the maximum allowed size")
        total_size += file_size
        if total_size > config.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Total upload size exceeds the maximum allowed size")

        object_key = _build_object_key(namespace, repo, revision_name, upload.filename)
        await provider.upload_stream(upload.file, object_key)

        meta = await provider.get_file_metadata(object_key)
        etag = meta.get("etag") or file_md5
        stored_size = int(meta.get("size") or file_size)
        uploaded_objects.append(object_key)
        uploaded_file_records.append((upload.filename, object_key, file_sha, etag, stored_size, upload.content_type))

    return uploaded_objects, uploaded_file_records, total_size


async def _cleanup_uploaded_objects(provider, uploaded_objects: list[str]) -> None:
    """Delete objects uploaded by a request that did not complete successfully."""
    for object_key in uploaded_objects:
        try:
            await provider.delete_file(object_key)
        except Exception:
            logging.exception("Failed to clean up uploaded object", extra={"object_key": object_key})


@ai_models_uploads_router.post(
    "/api/models/{namespace}/{repo}/upload",
    response_model=MultipartUploadResponse,
    summary="Upload model files",
    description="Upload one or more files to create or update a model repository revision. Files are stored and tracked with their content hashes.",
    tags=["Models"],
)
async def upload_model(
    request: Request,
    namespace: str,
    repo: str,
    revision_name: str = Form("main"),
    files: list[UploadFile] = File(...),
) -> MultipartUploadResponse:
    """
    Upload one or more files and create a repository revision.

    Returns:
        MultipartUploadResponse: Response containing upload status and file details.

    Raises:
        HTTPException: If no files are provided or upload fails.
    """

    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required")
    if len(files) > config.MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail="Too many files in upload")
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the maximum allowed size")

    _validate_request_components(namespace, repo, revision_name)
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=422, detail="Each uploaded file must have a filename")
        _build_object_key(namespace, repo, revision_name, upload.filename)

    provider = get_storage_provider()
    uploaded_objects: list[str] = []
    try:
        uploaded_objects, uploaded_file_records, total_size = await _process_uploaded_files(
            files,
            provider,
            namespace,
            repo,
            revision_name,
        )

        pool = await get_db_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                model_id, revision_id = await _create_model_and_initial_revision(conn, namespace, repo, revision_name)
                file_insert_stmt = await _prepare_file_insert_stmt(conn)
                for filename, object_key, file_sha, etag, stored_size, content_type in uploaded_file_records:
                    await file_insert_stmt.fetch(
                        revision_id,
                        filename,
                        stored_size,
                        file_sha,
                        etag,
                        object_key,
                        content_type,
                    )

                revision_hasher = hashlib.sha256()
                for filename, _, sha, etag, _, _ in sorted(uploaded_file_records):
                    revision_hasher.update(f"{filename}:{sha}:{etag}".encode())
                revision_hash = revision_hasher.hexdigest()
                update_revision_stmt = await conn.prepare(
                    "UPDATE model_revisions SET revision_identifier=$1, etag=$2 WHERE id=$3"
                )
                await update_revision_stmt.fetch(revision_hash, revision_hash[:32], revision_id)
                update_repo_stmt = await conn.prepare("UPDATE models SET current_revision=$1 WHERE id=$2")
                await update_repo_stmt.fetch(revision_name, model_id)
    except HTTPException:
        await _cleanup_uploaded_objects(provider, uploaded_objects)
        raise
    except Exception:
        await _cleanup_uploaded_objects(provider, uploaded_objects)
        logging.exception(
            "Multipart upload failed",
            extra={"namespace": namespace, "repo": repo, "revision_name": revision_name},
        )
        raise HTTPException(status_code=500, detail="Failed to upload model files")

    return MultipartUploadResponse(
        status="uploaded",
        repo=f"{namespace}/{repo}",
        revision=revision_name,
        files=len(files),
        total_size=total_size,
    )


@ai_models_uploads_router.post(
    "/api/models/{namespace}/{repo}/revisions",
    response_model=RevisionModel,
    summary="Create revision",
    description="Create a new revision for an existing repository with an optional revision identifier and ETag.",
    tags=["Models"],
)
async def create_model_revision(
    namespace: str,
    repo: str,
    request: RevisionCreateRequest,
) -> RevisionModel:
    """
    Create a revision for an existing repository.

    Returns:
        RevisionModel: The created revision model.

    Raises:
        HTTPException: If the repository is not found.
    """

    _validate_request_components(namespace, repo, request.revision_name)
    revision = await create_revision(namespace, repo, request.revision_name, request.revision_identifier, request.etag)
    if not revision:
        raise HTTPException(status_code=404, detail="Repository not found")
    return RevisionModel(**{"id": str(revision["id"]), "revision": revision["revision"], "sha": revision["sha"] or ""})


@ai_models_uploads_router.post(
    "/api/models/{namespace}/{repo}/upload-url",
    response_model=UploadUrlResponse,
    summary="Generate upload URL",
    description="Generate a presigned URL for uploading a file directly to object storage for a specific repository revision.",
    tags=["Models"],
)
async def generate_upload_url(namespace: str, repo: str, request: UploadUrlRequest) -> UploadUrlResponse:
    """
    Generate a storage upload URL for a file in a revision.

    Returns:
        UploadUrlResponse: Response containing the upload URL and object key.
    """

    object_key = _build_object_key(namespace, repo, request.revision_name, request.file_name)
    provider = get_storage_provider()
    upload_url = await provider.generate_upload_url(object_key)
    return UploadUrlResponse(upload_url=upload_url, object_key=object_key, method="PUT")


@ai_models_uploads_router.post(
    "/api/models/{namespace}/{repo}/complete-upload",
    response_model=RevisionModel,
    summary="Complete file upload",
    description="Finalize an upload by creating or updating a revision and tracking records for all uploaded files in object storage.",
    tags=["Models"],
)
async def complete_upload(
    namespace: str,
    repo: str,
    request: RevisionCreateRequest,
) -> RevisionModel:
    """
    Finalize an upload by creating/updating revision and file tracking records.

    Returns:
        RevisionModel: The created or updated revision model.

    Raises:
        HTTPException: If no uploaded files are found or creation fails.
    """

    _validate_request_components(namespace, repo, request.revision_name)
    provider = get_storage_provider()
    storage_prefix = f"{namespace}/{repo}/{request.revision_name}"
    object_keys = await provider.list_files(storage_prefix)
    if not object_keys:
        raise HTTPException(status_code=404, detail="No uploaded files found for the requested revision")

    prefix_with_slash = f"{storage_prefix}/"
    file_records = []
    for object_key in object_keys:
        if not object_key.startswith(prefix_with_slash):
            continue
        file_path = object_key[len(prefix_with_slash) :]
        if not file_path:
            continue
        file_records.append((file_path, object_key, await provider.get_file_metadata(object_key)))

    if not file_records:
        raise HTTPException(status_code=404, detail="No uploaded files found for the requested revision")

    derived_revision_identifier = request.revision_identifier
    if not derived_revision_identifier:
        checksums = [(file_path, metadata.get("checksum_sha256")) for file_path, _, metadata in file_records]
        if all(checksum for _, checksum in checksums):
            digest = hashlib.sha256()
            for file_path, checksum in sorted(checksums):
                digest.update(f"{file_path}:{checksum}".encode())
            derived_revision_identifier = digest.hexdigest()

    revision = await create_revision(
        namespace,
        repo,
        request.revision_name,
        derived_revision_identifier,
        request.etag or (derived_revision_identifier[:32] if derived_revision_identifier else None),
    )
    if not revision:
        raise HTTPException(status_code=404, detail="Repository not found")

    for file_path, object_key, metadata in file_records:
        content_etag = metadata.get("etag")
        content_sha = metadata.get("checksum_sha256")

        await track_file(
            namespace=namespace,
            model_name=repo,
            revision_name=request.revision_name,
            file_path=file_path,
            file_size=int(metadata["size"]),
            content_sha=content_sha,
            content_etag=content_etag,
            s3_key=object_key,
        )

    return RevisionModel(id=str(revision["id"]), revision=revision["revision"], sha=revision["sha"] or "")
