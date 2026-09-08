"""Shared route helpers for the Gen3 AI model repo service."""

from pydantic import BaseModel, Field


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
