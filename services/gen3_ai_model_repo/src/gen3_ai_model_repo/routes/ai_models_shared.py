"""Shared route helpers for the Gen3 AI model repo service."""

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from gen3_ai_model_repo.storage.keys import validate_key_component


def serialize_huggingface_siblings(files: list[dict]) -> list[dict[str, object]]:
    """
    Serialize tracked files without mislabeling regular objects as LFS.

    Returns:
        list[dict[str, object]]: HF-compatible sibling objects.

    Raises:
        ValueError: If an explicitly marked LFS file lacks pointer metadata.
    """
    siblings: list[dict[str, object]] = []
    for file_record in files:
        sibling: dict[str, object] = {
            "rfilename": file_record["path"],
            "size": file_record["size"],
            "blob_id": file_record["oid"],
        }
        if file_record.get("storage_protocol") == "lfs":
            pointer_size = file_record.get("pointer_size")
            if not isinstance(pointer_size, int) or pointer_size < 1:
                raise ValueError("LFS file metadata is missing pointer_size")
            sibling["lfs"] = {
                "sha256": file_record["oid"],
                "size": file_record["size"],
                "pointerSize": pointer_size,
            }
        siblings.append(sibling)
    return siblings


async def validate_repository_path(request: Request) -> None:
    """
    Reject repository path components that cannot be represented safely.

    Raises:
        HTTPException: If a repository path component is invalid.
    """
    for name in ("namespace", "repo"):
        value = request.path_params.get(name)
        if value is None:
            continue
        try:
            validate_key_component(value, name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


class RepositoryCreateRequest(BaseModel):
    """
    Request payload for creating a repository.

    Attributes:
        description: Optional human-readable repository description.
        tags: Optional list of tags used for filtering and discovery.
    """

    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class RepositoryUpdateRequest(BaseModel):
    """Request payload for updating repository metadata fields."""

    description: str | None = None
    tags: list[str] | None = None


class HuggingFaceCreateRepositoryRequest(BaseModel):
    """Subset of the Hugging Face repository-creation request we support."""

    name: str
    organization: str | None = None
    private: bool = False
    repo_type: str | None = None


class HuggingFaceDeleteRepositoryRequest(BaseModel):
    """Request payload used by ``HfApi.delete_repo``."""

    name: str
    organization: str | None = None
    repo_type: str | None = Field(default=None, alias="type")


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
