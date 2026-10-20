"""Upload routes for the Gen3 AI model repo service."""

import base64
import hashlib
import io
import json
import tempfile
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from gen3_ai_model_repo import config
from gen3_ai_model_repo.auth import AuthorizedRouter, verify_authorization
from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.database.db import get_db_pool
from gen3_ai_model_repo.database.file_tracking import storage_key_is_referenced
from gen3_ai_model_repo.database.repo_metadata import model_exists as db_model_exists
from gen3_ai_model_repo.database.revisions import create_revision
from gen3_ai_model_repo.database.upload_intents import (
    get_active_intent_by_checksum,
    get_upload_intents,
    reserve_upload,
)
from gen3_ai_model_repo.models.schemas import (
    DirectUploadCompleteRequest,
    RevisionCreateRequest,
    RevisionModel,
    UploadUrlRequest,
    UploadUrlResponse,
)
from gen3_ai_model_repo.routes.ai_models_shared import MultipartUploadResponse, validate_repository_path
from gen3_ai_model_repo.storage.helpers import get_storage_provider
from gen3_ai_model_repo.storage.keys import build_object_key, validate_key_component

ai_models_uploads_router = AuthorizedRouter(
    dependencies=[Depends(verify_authorization), Depends(validate_repository_path)]
)
# The capability in the URL is the authorization for the PUT itself.  The HF
# basic uploader does not forward action headers or bearer headers on this PUT.
lfs_proxy_router = APIRouter()


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
    attempt_id: str,
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

        object_key = _build_object_key(namespace, repo, revision_name, f"{attempt_id}/{upload.filename}")
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
    "/{namespace}/{repo}.git/info/lfs/objects/batch",
    summary="Negotiate a Hugging Face LFS upload",
    description=(
        "Accept a SHA-256 LFS upload batch request and return basic-transfer "
        "upload and verification actions. Each object is bounded by the service "
        "size and quota limits, backed by a durable upload intent, and must pass "
        "checksum and size validation before it can be committed. Only model "
        "repositories and the legacy basic LFS transfer are supported; Xet and "
        "other LFS transfers are not provided."
    ),
    response_description="LFS transfer selection and per-object upload/verify actions.",
    tags=["Models"],
)
async def lfs_batch(namespace: str, repo: str, request: Request) -> dict:
    """
    Reserve LFS objects and return basic-transfer upload instructions.

    Returns:
        dict: The negotiated transfer and object actions.

    Raises:
        HTTPException: If the request or object reservation is invalid.
    """
    _validate_request_components(namespace, repo, "main")
    payload = await request.json()
    if payload.get("operation") != "upload" or payload.get("hash_algo") != "sha256":
        raise HTTPException(status_code=422, detail="Only SHA-256 LFS uploads are supported")
    revision = (payload.get("ref") or {}).get("name") or "main"
    objects = payload.get("objects")
    if not isinstance(objects, list) or len(objects) > config.MAX_UPLOAD_FILES:
        raise HTTPException(status_code=422, detail="objects must be a bounded list")
    response_objects = []
    for item in objects:
        oid = item.get("oid") if isinstance(item, dict) else None
        size = item.get("size") if isinstance(item, dict) else None
        if not isinstance(oid, str) or len(oid) != 64 or not isinstance(size, int) or size < 0:
            raise HTTPException(status_code=422, detail="Invalid LFS object")
        if size > config.DIRECT_UPLOAD_MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="LFS object exceeds maximum size")
        existing = await get_active_intent_by_checksum(namespace, repo, revision, oid)
        if existing:
            object_key = existing["object_key"]
            intent_id = existing["id"]
        else:
            intent_id = uuid4()
            object_key = _build_object_key(namespace, repo, revision, f"lfs/{intent_id}/{oid}")
            await reserve_upload(namespace, repo, revision, oid, object_key, oid, size, intent_id=intent_id)
        upload_url = str(request.base_url).rstrip("/") + f"/api/models/{namespace}/{repo}/lfs/upload/{intent_id}"
        response_objects.append(
            {
                "oid": oid,
                "size": size,
                "actions": {
                    "upload": {
                        "href": upload_url,
                        "header": {
                            "x-amz-sdk-checksum-algorithm": "SHA256",
                            "x-amz-checksum-sha256": base64.b64encode(bytes.fromhex(oid)).decode(),
                        },
                    },
                    "verify": {
                        "href": str(request.base_url).rstrip("/")
                        + f"/api/models/{namespace}/{repo}/lfs/verify/{oid}/{revision}"
                    },
                },
            }
        )
    return {"transfer": "basic", "objects": response_objects}


async def _receive_lfs_object(request: Request, expected_size: int, expected_oid: str):
    """
    Buffer one bounded LFS object before committing it to object storage.

    Returns:
        tempfile.SpooledTemporaryFile: A rewound verified object stream.

    Raises:
        HTTPException: If the object exceeds its reservation or fails verification.
    """
    total = 0
    digest = hashlib.sha256()
    temporary = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
    async for chunk in request.stream():
        total += len(chunk)
        if total > expected_size or total > config.DIRECT_UPLOAD_MAX_FILE_BYTES:
            temporary.close()
            raise HTTPException(status_code=413, detail="LFS object exceeds reserved size")
        digest.update(chunk)
        temporary.write(chunk)
    if total != expected_size or digest.hexdigest() != expected_oid:
        temporary.close()
        raise HTTPException(status_code=422, detail="LFS object checksum or size mismatch")
    temporary.seek(0)
    return temporary


@lfs_proxy_router.put(
    "/api/models/{namespace}/{repo}/lfs/upload/{intent_id}",
    include_in_schema=False,
)
async def lfs_upload_object(namespace: str, repo: str, intent_id: UUID, request: Request) -> dict[str, str]:
    """
    Receive a locally routed LFS basic transfer.

    Returns:
        dict[str, str]: Upload status.

    Raises:
        HTTPException: If the intent is missing or the object is invalid.
    """
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        intent = await conn.fetchrow(
            """SELECT object_key, reserved_bytes, checksum_sha256 FROM model_upload_intents
               WHERE id=$1 AND namespace=$2 AND repo=$3 AND completed_at IS NULL AND expires_at > NOW()""",
            intent_id,
            namespace,
            repo,
        )
    if not intent:
        raise HTTPException(status_code=404, detail="LFS upload intent not found")
    temporary = await _receive_lfs_object(request, int(intent["reserved_bytes"]), intent["checksum_sha256"])
    try:
        provider = get_storage_provider()
        try:
            await provider.upload_stream(temporary, intent["object_key"])
        except Exception:
            try:
                await provider.delete_file(intent["object_key"])
            except Exception:
                logging.exception("Failed to clean up failed LFS proxy object")
            raise
    finally:
        temporary.close()
    return {"status": "uploaded"}


@ai_models_uploads_router.post(
    "/api/models/{namespace}/{repo}/lfs/verify/{oid}/{revision}",
    include_in_schema=False,
)
async def lfs_verify_object(namespace: str, repo: str, oid: str, revision: str) -> dict[str, str]:
    """
    Verify that a negotiated LFS object exists with the reserved size.

    Returns:
        dict[str, str]: An empty successful verification response.

    Raises:
        HTTPException: If the intent or stored object is invalid.
    """
    intent = await get_active_intent_by_checksum(namespace, repo, revision, oid)
    if not intent:
        raise HTTPException(status_code=404, detail="LFS upload intent not found")
    metadata = await get_storage_provider().get_file_metadata(intent["object_key"])
    if int(metadata.get("size", -1)) != int(intent["reserved_bytes"]):
        raise HTTPException(status_code=422, detail="Stored LFS object size mismatch")
    return {}


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
    # The multipart endpoint creates a repository. Reject existing repositories
    # before touching storage; callers adding a revision must use the revision
    # and direct-upload completion flow.
    if await db_model_exists(namespace, repo):
        raise HTTPException(status_code=409, detail=f"Repository {namespace}/{repo} already exists")
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=422, detail="Each uploaded file must have a filename")
        _build_object_key(namespace, repo, revision_name, upload.filename)

    provider = get_storage_provider()
    attempt_id = uuid4().hex
    uploaded_objects: list[str] = []
    try:
        uploaded_objects, uploaded_file_records, total_size = await _process_uploaded_files(
            files,
            provider,
            namespace,
            repo,
            revision_name,
            attempt_id,
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

    Raises:
        HTTPException: If local storage does not support direct upload URLs.
    """

    try:
        checksum_bytes = base64.b64decode(request.checksum_sha256, validate=True)
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="checksum_sha256 must be valid base64") from None
    if len(checksum_bytes) != 32:
        raise HTTPException(status_code=422, detail="checksum_sha256 must decode to 32 bytes")
    checksum_hex = checksum_bytes.hex()
    intent_id = uuid4()
    object_key = _build_object_key(namespace, repo, request.revision_name, f"{intent_id}/{request.file_name}")
    provider = get_storage_provider()
    try:
        upload_url = await provider.generate_upload_url(object_key, checksum_sha256=request.checksum_sha256)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=409,
            detail="Direct upload URLs are not supported for local storage; use the multipart upload endpoint",
        ) from exc
    try:
        upload_id = await reserve_upload(
            namespace,
            repo,
            request.revision_name,
            request.file_name,
            object_key,
            checksum_hex,
            request.file_size,
            intent_id=intent_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return UploadUrlResponse(
        upload_url=upload_url,
        object_key=object_key,
        method="PUT",
        upload_id=str(upload_id),
        headers={
            "x-amz-sdk-checksum-algorithm": "SHA256",
            "x-amz-checksum-sha256": request.checksum_sha256,
        },
    )


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
    request: DirectUploadCompleteRequest,
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
    if not request.upload_ids or len(request.upload_ids) > config.DIRECT_UPLOAD_MAX_INTENTS:
        raise HTTPException(status_code=422, detail="upload_ids must contain between 1 and the configured maximum")
    if len(set(request.upload_ids)) != len(request.upload_ids):
        raise HTTPException(status_code=409, detail="upload_ids must be unique")
    try:
        intent_ids = [UUID(value) for value in request.upload_ids]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid upload intent") from exc
    intents = await get_upload_intents(intent_ids, namespace, repo, request.revision_name)
    if len(intents) != len(intent_ids):
        raise HTTPException(status_code=404, detail="One or more upload intents are missing, expired, or completed")
    if len({row["file_name"] for row in intents}) != len(intents):
        raise HTTPException(status_code=409, detail="Upload intents contain duplicate file paths")

    file_records = []
    for row in intents:
        metadata = await provider.get_file_metadata(row["object_key"])
        size = int(metadata.get("size") or 0)
        if size > config.DIRECT_UPLOAD_MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"Uploaded object is too large: {row['file_name']}")
        actual_checksum = (metadata.get("checksum_sha256") or "").lower()
        if actual_checksum != row["checksum_sha256"]:
            raise HTTPException(status_code=400, detail=f"Checksum mismatch for {row['file_name']}")
        file_records.append((row["file_name"], row["object_key"], metadata))

    derived_revision_identifier = request.revision_identifier
    if not derived_revision_identifier:
        checksums = [(file_path, metadata.get("checksum_sha256")) for file_path, _, metadata in file_records]
        if all(checksum for _, checksum in checksums):
            digest = hashlib.sha256()
            for file_path, checksum in sorted(checksums):
                digest.update(f"{file_path}:{checksum}".encode())
            derived_revision_identifier = digest.hexdigest()

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            locked = await conn.fetch(
                """SELECT id, file_name, object_key, checksum_sha256, reserved_bytes FROM model_upload_intents
                   WHERE id=ANY($1::uuid[]) AND completed_at IS NULL AND expires_at > NOW()
                   FOR UPDATE""",
                intent_ids,
            )
            metadata_by_key = {item[1]: item[2] for item in file_records}
            if len(locked) != len(intent_ids) or any(
                row["checksum_sha256"]
                != (metadata_by_key.get(row["object_key"], {}).get("checksum_sha256") or "").lower()
                or int(metadata_by_key.get(row["object_key"], {}).get("size") or 0) > int(row["reserved_bytes"])
                for row in locked
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Upload intents changed or stored objects exceed their reservations",
                )
            model_row = await conn.fetchrow(
                "SELECT id FROM models WHERE namespace=$1 AND model_name=$2 FOR UPDATE", namespace, repo
            )
            if not model_row:
                raise HTTPException(status_code=404, detail="Repository not found")
            revision = await conn.fetchrow(
                """INSERT INTO model_revisions (model_id, revision_name, revision_identifier, etag)
                   VALUES ($1,$2,$3,$4)
                   ON CONFLICT (model_id, revision_name) DO UPDATE SET revision_identifier=EXCLUDED.revision_identifier, etag=EXCLUDED.etag
                   RETURNING id, revision_name, revision_identifier, etag""",
                model_row["id"],
                request.revision_name,
                derived_revision_identifier,
                request.etag or (derived_revision_identifier[:32] if derived_revision_identifier else None),
            )
            for file_path, object_key, metadata in file_records:
                await conn.execute(
                    """INSERT INTO model_files (revision_id,file_path,file_size,content_sha,content_etag,s3_key)
                       VALUES ($1,$2,$3,$4,$5,$6)
                       ON CONFLICT (revision_id,file_path) DO UPDATE SET file_size=EXCLUDED.file_size,
                       content_sha=EXCLUDED.content_sha, content_etag=EXCLUDED.content_etag, s3_key=EXCLUDED.s3_key""",
                    revision["id"],
                    file_path,
                    int(metadata["size"]),
                    metadata["checksum_sha256"],
                    metadata.get("etag"),
                    object_key,
                )
            await conn.execute(
                "UPDATE model_upload_intents SET completed_at=NOW() WHERE id=ANY($1::uuid[])", intent_ids
            )
    return RevisionModel(
        id=str(revision["id"]),
        revision=revision["revision_name"],
        sha=revision["revision_identifier"] or "",
    )


@ai_models_uploads_router.post(
    "/api/models/{namespace}/{repo}/preupload/{revision}",
    summary="Prepare a Hugging Face commit upload",
    description=(
        "Inspect a bounded list of files from a Hugging Face commit request and "
        "select regular or legacy LFS upload mode by file size. The response "
        "contains one upload decision per path; storage limits and total request "
        "limits are enforced. This endpoint does not make files visible; use the "
        "corresponding upload and commit operations. Xet is not supported."
    ),
    response_description="Per-file upload modes and ignore decisions.",
    tags=["Models"],
)
async def preupload_huggingface_files(namespace: str, repo: str, revision: str, request: Request) -> dict:
    """
    Select regular upload mode for Hugging Face commit operations.

    The regular commit path keeps the service's atomic database completion
    guarantees. LFS is not advertised here until a storage-backed LFS batch
    implementation is available.

    Returns:
        dict: Upload modes for each requested file.

    Raises:
        HTTPException: If the revision or request payload is invalid.
    """
    _validate_request_components(namespace, repo, revision)
    payload = await request.json()
    files = payload.get("files")
    if not isinstance(files, list):
        raise HTTPException(status_code=422, detail="files must be a list")
    if len(files) > config.MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail="Too many files in upload")
    total_size = 0
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise HTTPException(status_code=422, detail="Each file must include a path")
        size = item.get("size", 0)
        if not isinstance(size, int) or size < 0:
            raise HTTPException(status_code=422, detail="Each file must include a non-negative size")
        if size > config.DIRECT_UPLOAD_MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"Uploaded object is too large: {item['path']}")
        total_size += size
    if total_size > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Total upload size exceeds the maximum allowed size")
    return {
        "files": [
            {
                "path": item["path"],
                "uploadMode": "lfs" if item.get("size", 0) >= config.HF_LFS_THRESHOLD_BYTES else "regular",
                "shouldIgnore": False,
                "oid": None,
            }
            for item in files
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        ]
    }


@ai_models_uploads_router.post(
    "/api/models/{namespace}/{repo}/commit/{revision}",
    summary="Apply a Hugging Face commit",
    description=(
        "Apply newline-delimited JSON file, LFS-file, and deletion operations "
        "atomically to a model revision. Regular uploads use bounded service "
        "storage and LFS files must reference validated durable upload intents; "
        "existing repository contents remain unchanged if validation or commit "
        "fails. Repository authorization, quotas, checksums, and size limits "
        "apply. Xet commit operations are not supported."
    ),
    response_description="The committed revision identifier and commit URL.",
    tags=["Models"],
)
async def commit_huggingface_files(namespace: str, repo: str, revision: str, request: Request) -> dict[str, str | None]:
    """
    Apply regular-file Hugging Face commit operations atomically.

    Returns:
        dict[str, str | None]: Commit metadata expected by the Hub client.

    Raises:
        HTTPException: If the commit is invalid or uses unsupported LFS operations.
    """
    _validate_request_components(namespace, repo, revision)
    raw_lines = (await request.body()).splitlines()
    try:
        records = [json.loads(line) for line in raw_lines if line.strip()]
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Commit body must be valid NDJSON") from exc
    operations = [
        record for record in records if record.get("key") in {"file", "lfsFile", "deletedFile", "deletedFolder"}
    ]
    if not await db_model_exists(namespace, repo):
        raise HTTPException(status_code=404, detail="Repository not found")

    additions: list[tuple[str, bytes]] = []
    lfs_additions: list[tuple[str, str, int]] = []
    deletes: list[tuple[str, bool]] = []
    for record in operations:
        value = record.get("value") or {}
        path = value.get("path")
        if not isinstance(path, str):
            raise HTTPException(status_code=422, detail="Commit operation path is required")
        if record["key"] == "file":
            if value.get("encoding") != "base64":
                raise HTTPException(status_code=422, detail="Regular commit files must use base64 encoding")
            try:
                content = base64.b64decode(value["content"], validate=True)
            except (KeyError, ValueError, TypeError) as exc:
                raise HTTPException(status_code=422, detail="Invalid base64 commit content") from exc
            if len(content) > config.DIRECT_UPLOAD_MAX_FILE_BYTES:
                raise HTTPException(status_code=413, detail=f"Uploaded object is too large: {path}")
            additions.append((path, content))
        elif record["key"] == "lfsFile":
            oid = value.get("oid")
            size = value.get("size")
            if not isinstance(oid, str) or len(oid) != 64 or not isinstance(size, int) or size < 0:
                raise HTTPException(status_code=422, detail="Invalid LFS commit object")
            lfs_additions.append((path, oid, size))
        else:
            deletes.append((path, record["key"] == "deletedFolder"))

    provider = get_storage_provider()
    staged: list[tuple[str, str, bytes, str]] = []
    staged_sizes: dict[str, int] = {}
    for path, content in additions:
        object_key = _build_object_key(namespace, repo, revision, f"{uuid4()}/{path}")
        await provider.upload_stream(io.BytesIO(content), object_key)
        staged.append((path, object_key, content, hashlib.sha256(content).hexdigest()))
        staged_sizes[path] = len(content)
    lfs_intent_ids: list[UUID] = []
    for path, oid, size in lfs_additions:
        intent = await get_active_intent_by_checksum(namespace, repo, revision, oid)
        if not intent:
            raise HTTPException(status_code=422, detail=f"No active LFS upload for {path}")
        metadata = await provider.get_file_metadata(intent["object_key"])
        if int(metadata.get("size", -1)) != size or size != int(intent["reserved_bytes"]):
            raise HTTPException(status_code=422, detail=f"LFS object size mismatch: {path}")
        if metadata.get("checksum_sha256") and metadata["checksum_sha256"] != oid:
            raise HTTPException(status_code=422, detail=f"LFS object checksum mismatch: {path}")
        staged.append((path, intent["object_key"], b"", oid))
        staged_sizes[path] = size
        lfs_intent_ids.append(intent["id"])

    old_keys: list[str] = []
    pool = await get_db_pool()
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                model_row = await conn.fetchrow(
                    "SELECT id FROM models WHERE namespace=$1 AND model_name=$2 FOR UPDATE", namespace, repo
                )
                if not model_row:
                    raise HTTPException(status_code=404, detail="Repository not found")
                revision_row = await conn.fetchrow(
                    "SELECT id FROM model_revisions WHERE model_id=$1 AND revision_name=$2 FOR UPDATE",
                    model_row["id"],
                    revision,
                )
                if not revision_row:
                    revision_row = await conn.fetchrow(
                        """INSERT INTO model_revisions (model_id, revision_name, revision_identifier, etag)
                           VALUES ($1,$2,$3,$4) RETURNING id""",
                        model_row["id"],
                        revision,
                        hashlib.sha256(b"").hexdigest(),
                        None,
                    )
                for path, _, _, _ in staged:
                    old = await conn.fetchval(
                        "SELECT s3_key FROM model_files WHERE revision_id=$1 AND file_path=$2",
                        revision_row["id"],
                        path,
                    )
                    if old:
                        old_keys.append(old)
                    staged_item = next(item for item in staged if item[0] == path)
                    await conn.execute(
                        """INSERT INTO model_files (revision_id,file_path,file_size,content_sha,s3_key)
                           VALUES ($1,$2,$3,$4,$5)
                           ON CONFLICT (revision_id,file_path) DO UPDATE SET file_size=EXCLUDED.file_size,
                           content_sha=EXCLUDED.content_sha,s3_key=EXCLUDED.s3_key,updated_at=NOW()""",
                        revision_row["id"],
                        path,
                        staged_sizes[path],
                        staged_item[3],
                        staged_item[1],
                    )
                for path, is_folder in deletes:
                    pattern = path.rstrip("/") + "/%" if is_folder else None
                    rows = await conn.fetch(
                        "SELECT s3_key FROM model_files WHERE revision_id=$1 AND (file_path=$2 OR ($3::text IS NOT NULL AND file_path LIKE $3))",
                        revision_row["id"],
                        path,
                        pattern,
                    )
                    old_keys.extend(row["s3_key"] for row in rows)
                    await conn.execute(
                        "DELETE FROM model_files WHERE revision_id=$1 AND (file_path=$2 OR ($3::text IS NOT NULL AND file_path LIKE $3))",
                        revision_row["id"],
                        path,
                        pattern,
                    )
                commit_oid = hashlib.sha256(
                    b"".join(f"{path}:{sha}".encode() for path, _, _, sha in sorted(staged))
                ).hexdigest()
                await conn.execute(
                    "UPDATE model_revisions SET revision_identifier=$1, updated_at=NOW() WHERE id=$2",
                    commit_oid,
                    revision_row["id"],
                )
                if lfs_intent_ids:
                    await conn.execute(
                        "UPDATE model_upload_intents SET completed_at=NOW() WHERE id=ANY($1::uuid[])",
                        lfs_intent_ids,
                    )
    except Exception:
        for _, object_key, _, _ in staged:
            await provider.delete_file(object_key)
        raise

    for object_key in old_keys:
        if not await storage_key_is_referenced(object_key):
            await provider.delete_file(object_key)
    commit_url = str(request.base_url).rstrip("/") + f"/{namespace}/{repo}/commit/{commit_oid}"
    return {"commitUrl": commit_url, "commitOid": commit_oid, "pullRequestUrl": None}
