"""Storage routes for the Gen3 AI model repo service."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from gen3_ai_model_repo.auth import verify_authorization
from gen3_ai_model_repo.storage.helpers import get_storage_provider

storage_router = APIRouter()


class StorageVerifyRequest(BaseModel):
    """
    Request payload for verifying storage contents at a prefix.

    Attributes:
        storage_prefix: Path prefix to inspect inside the storage backend.
    """

    storage_prefix: str


class StorageVerifyResponse(BaseModel):
    """
    Response payload describing the state of a storage prefix.

    Attributes:
        exists: True if at least one file exists at the prefix.
        file_count: Number of files found under the prefix.
        total_size: Aggregate size in bytes of all files under the prefix.
    """

    exists: bool
    file_count: int
    total_size: int


class StorageDownloadUrlRequest(BaseModel):
    """
    Request payload for generating a pre-signed download URL.

    Attributes:
        object_key: Full object key identifying the file in the storage backend.
    """

    object_key: str


class StorageDownloadUrlResponse(BaseModel):
    """
    Response payload containing a pre-signed download URL.

    Attributes:
        presigned_url: Time-limited URL granting direct access to the object.
    """

    presigned_url: str


@storage_router.post(
    "/api/storage/verify",
    response_model=StorageVerifyResponse,
    summary="Verify a storage prefix",
    description="Check whether a storage prefix exists, count the files it contains, and total their sizes.",
    tags=["Storage"],
)
@storage_router.post(
    "/storage/verify",
    response_model=StorageVerifyResponse,
    summary="Verify a storage prefix",
    description="Check whether a storage prefix exists, count the files it contains, and total their sizes.",
    tags=["Storage"],
    include_in_schema=False,
)
async def verify_storage(
    request: StorageVerifyRequest, _: None = Depends(verify_authorization)
) -> StorageVerifyResponse:
    """Check whether a storage prefix exists and how much data it contains."""

    provider = get_storage_provider()
    files = await provider.list_files(request.storage_prefix)
    total_size = 0
    for object_key in files:
        meta = await provider.get_file_metadata(object_key)
        total_size += int(meta["size"])
    return StorageVerifyResponse(exists=bool(files), file_count=len(files), total_size=total_size)


@storage_router.post(
    "/api/storage/download-url",
    response_model=StorageDownloadUrlResponse,
    summary="Generate a signed download URL",
    description="Create a short-lived URL that allows an authenticated user to download a storage object.",
    tags=["Storage"],
)
@storage_router.post(
    "/storage/download-url",
    response_model=StorageDownloadUrlResponse,
    summary="Generate a signed download URL",
    description="Create a short-lived URL that allows an authenticated user to download a storage object.",
    tags=["Storage"],
    include_in_schema=False,
)
async def storage_download_url(
    request: StorageDownloadUrlRequest, _: None = Depends(verify_authorization)
) -> StorageDownloadUrlResponse:
    """Create a pre-signed URL for downloading a stored object."""

    provider = get_storage_provider()
    return StorageDownloadUrlResponse(presigned_url=await provider.generate_signed_url(request.object_key))
