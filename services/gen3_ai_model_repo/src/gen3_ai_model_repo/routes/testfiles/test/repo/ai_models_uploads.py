import hashlib

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from gen3_ai_model_repo.auth import verify_authorization
from gen3_ai_model_repo.database.db import get_db_pool
from gen3_ai_model_repo.database.file_tracking import track_file
from gen3_ai_model_repo.database.revisions import create_revision, get_revision_identifier_column
from gen3_ai_model_repo.models.schemas import RevisionCreateRequest, RevisionModel, UploadUrlRequest, UploadUrlResponse
from gen3_ai_model_repo.routes.ai_models_shared import (
    MultipartUploadResponse,
    model_files_optional_columns,
    repository_has_current_revision,
)
from gen3_ai_model_repo.storage.helpers import get_storage_provider

ai_models_uploads_router = APIRouter()


async def _hash_upload_file(upload: UploadFile) -> tuple[str, str, int]:
    """Compute SHA256 and MD5 digests for an uploaded file without loading it into memory."""
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


async def _create_repository_and_initial_revision(
    conn,
    namespace: str,
    repo: str,
    revision_name: str,
) -> tuple[int, int, str, bool, bool, bool]:
    """Create repository and placeholder revision records and return schema capability flags."""
    identifier_column = await get_revision_identifier_column(conn)
    has_current_revision = await repository_has_current_revision(conn)
    has_s3_key, has_file_type = await model_files_optional_columns(conn)

    exists_stmt = await conn.prepare("SELECT 1 FROM model_repositories WHERE namespace=$1 AND repo_name=$2")
    exists = await exists_stmt.fetchval(namespace, repo)
    if exists:
        raise HTTPException(status_code=409, detail=f"Repository {namespace}/{repo} already exists")

    if has_current_revision:
        insert_repo_stmt = await conn.prepare(
            "INSERT INTO model_repositories (namespace, repo_name, description, tags, current_revision) VALUES ($1,$2,$3,$4,$5)"
        )
        await insert_repo_stmt.fetch(namespace, repo, None, [], revision_name)
    else:
        insert_repo_stmt = await conn.prepare(
            "INSERT INTO model_repositories (namespace, repo_name, description, tags) VALUES ($1,$2,$3,$4)"
        )
        await insert_repo_stmt.fetch(namespace, repo, None, [])

    repo_id_stmt = await conn.prepare("SELECT id FROM model_repositories WHERE namespace=$1 AND repo_name=$2")
    repo_id = await repo_id_stmt.fetchval(namespace, repo)

    initial_revision_hash = hashlib.sha256(f"{namespace}/{repo}:{revision_name}".encode()).hexdigest()
    insert_revision_stmt = await conn.prepare(
        f"INSERT INTO model_revisions (repository_id, revision_name, {identifier_column}, etag) VALUES ($1,$2,$3,$4)"
    )
    await insert_revision_stmt.fetch(repo_id, revision_name, initial_revision_hash, initial_revision_hash[:32])

    revision_id_stmt = await conn.prepare("SELECT id FROM model_revisions WHERE repository_id=$1 AND revision_name=$2")
    revision_id = await revision_id_stmt.fetchval(repo_id, revision_name)

    return repo_id, revision_id, identifier_column, has_current_revision, has_s3_key, has_file_type


async def _prepare_file_insert_stmt(conn, has_s3_key: bool, has_file_type: bool):
    """Build and prepare the INSERT statement for model files based on optional columns."""
    file_columns = ["revision_id", "file_path", "file_size", "content_sha", "content_etag"]
    if has_s3_key:
        file_columns.append("s3_key")
    if has_file_type:
        file_columns.append("file_type")

    file_insert_sql = (
        f"INSERT INTO model_files ({', '.join(file_columns)}) "
        f"VALUES ({', '.join(f'${idx}' for idx in range(1, len(file_columns) + 1))})"
    )
    return await conn.prepare(file_insert_sql)


async def _process_uploaded_files(
    files: list[UploadFile],
    provider,
    namespace: str,
    repo: str,
    revision_name: str,
    revision_id: int,
    file_insert_stmt,
    has_s3_key: bool,
    has_file_type: bool,
) -> tuple[list[str], list[tuple[str, str, str]], int]:
    """Upload files to object storage and persist file metadata rows."""
    uploaded_objects: list[str] = []
    uploaded_file_digests: list[tuple[str, str, str]] = []
    total_size = 0

    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=422, detail="Each uploaded file must have a filename")

        file_sha, file_md5, file_size = await _hash_upload_file(upload)
        total_size += file_size

        object_key = f"{namespace}/{repo}/{revision_name}/{upload.filename}"
        await provider.upload_stream(upload.file, object_key)

        meta = await provider.get_file_metadata(object_key)
        etag = meta.get("etag") or file_md5
        file_values: list[object] = [
            revision_id,
            upload.filename,
            int(meta.get("size") or file_size),
            file_sha,
            etag,
        ]
        if has_s3_key:
            file_values.append(object_key)
        if has_file_type:
            file_values.append(upload.content_type)
        await file_insert_stmt.fetch(*file_values)
        uploaded_objects.append(object_key)
        uploaded_file_digests.append((upload.filename, file_sha, etag))

    return uploaded_objects, uploaded_file_digests, total_size


@ai_models_uploads_router.post(
    "/api/repositories/{namespace}/{repo}/upload", response_model=MultipartUploadResponse, tags=["Models"]
)
async def upload_model(
    namespace: str,
    repo: str,
    revision_name: str = Form("main"),
    files: list[UploadFile] = File(...),
    _: None = Depends(verify_authorization),
) -> MultipartUploadResponse:
    """
    Upload one or more files and create a repository revision.
    """

    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required")

    provider = get_storage_provider()
    pool = await get_db_pool()
    uploaded_objects: list[str] = []
    total_size = 0

    async with pool.acquire() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            (
                repo_id,
                revision_id,
                identifier_column,
                has_current_revision,
                has_s3_key,
                has_file_type,
            ) = await _create_repository_and_initial_revision(conn, namespace, repo, revision_name)
            file_insert_stmt = await _prepare_file_insert_stmt(conn, has_s3_key, has_file_type)

            uploaded_objects, uploaded_file_digests, total_size = await _process_uploaded_files(
                files,
                provider,
                namespace,
                repo,
                revision_name,
                revision_id,
                file_insert_stmt,
                has_s3_key,
                has_file_type,
            )

            revision_hasher = hashlib.sha256()
            for filename, sha, etag in sorted(uploaded_file_digests):
                revision_hasher.update(f"{filename}:{sha}:{etag}".encode())
            revision_hash = revision_hasher.hexdigest()
            update_revision_stmt = await conn.prepare(
                f"UPDATE model_revisions SET {identifier_column}=$1, etag=$2 WHERE id=$3"
            )
            await update_revision_stmt.fetch(revision_hash, revision_hash[:32], revision_id)
            if has_current_revision:
                update_repo_stmt = await conn.prepare("UPDATE model_repositories SET current_revision=$1 WHERE id=$2")
                await update_repo_stmt.fetch(revision_name, repo_id)
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


@ai_models_uploads_router.post(
    "/api/repositories/{namespace}/{repo}/revisions",
    response_model=RevisionModel,
    summary="Create revision",
    tags=["Models"],
)
async def create_model_revision(
    namespace: str,
    repo: str,
    request: RevisionCreateRequest,
    _: None = Depends(verify_authorization),
) -> RevisionModel:
    """
    Create a revision for an existing repository.
    """

    revision = await create_revision(namespace, repo, request.revision_name, request.revision_identifier, request.etag)
    if not revision:
        raise HTTPException(status_code=404, detail="Repository not found")
    return RevisionModel(**{"id": str(revision["id"]), "revision": revision["revision"], "sha": revision["sha"] or ""})


@ai_models_uploads_router.post(
    "/api/repositories/{namespace}/{repo}/upload-url", response_model=UploadUrlResponse, tags=["Models"]
)
async def generate_upload_url(
    namespace: str, repo: str, request: UploadUrlRequest, _: None = Depends(verify_authorization)
) -> UploadUrlResponse:
    """
    Generate a storage upload URL for a file in a revision.
    """

    provider = get_storage_provider()
    object_key = f"{namespace}/{repo}/{request.revision_name}/{request.file_name}"
    upload_url = await provider.generate_upload_url(object_key)
    return UploadUrlResponse(upload_url=upload_url, object_key=object_key, method="PUT")


@ai_models_uploads_router.post(
    "/api/repositories/{namespace}/{repo}/complete-upload", response_model=RevisionModel, tags=["Models"]
)
async def complete_upload(
    namespace: str, repo: str, request: RevisionCreateRequest, _: None = Depends(verify_authorization)
) -> RevisionModel:
    """
    Finalize an upload by creating/updating revision and file tracking records.
    """

    provider = get_storage_provider()
    storage_prefix = f"{namespace}/{repo}/{request.revision_name}"
    object_keys = await provider.list_files(storage_prefix)

    derived_revision_identifier = request.revision_identifier
    if not derived_revision_identifier:
        digest = hashlib.sha256()
        for object_key in sorted(object_keys):
            digest.update(object_key.encode())
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
