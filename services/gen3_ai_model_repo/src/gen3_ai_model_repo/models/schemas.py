from datetime import datetime

from pydantic import BaseModel, Field


class RepositoryModel(BaseModel):
    """Repository information returned by the API."""

    id: str
    description: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class RepositoryFileModel(BaseModel):
    """A file entry included in repository metadata."""

    type: str
    oid: str
    size: int


class RepositoryMetadataModel(BaseModel):
    """Metadata describing a repository and its current state."""

    namespace: str
    repo: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class UploadModelResponse(BaseModel):
    """Response payload for upload operations."""

    status: str
    repo: str
    metadata_file: str
    metadata: RepositoryMetadataModel


class RepositoryInfoModel(BaseModel):
    """Detailed repository information returned by the API."""

    id: str
    sha: str
    etag: str
    size: int
    files: list[RepositoryFileModel]
    metadata: RepositoryMetadataModel
    security_status: dict


class RevisionModel(BaseModel):
    """Revision metadata for a repository revision."""

    id: str
    revision: str
    sha: str


class RevisionListResponseModel(BaseModel):
    """Response payload containing a list of revisions."""

    repo: str
    revisions: list[RevisionModel]


class TreeEntryModel(BaseModel):
    """Tree entry describing a file in a repository tree."""

    type: str
    oid: str
    size: int


class DeleteModelResponse(BaseModel):
    """Response payload for delete operations."""

    status: str
    repo: str


class RevisionCreateRequest(BaseModel):
    """Request payload for creating a revision."""

    revision_name: str = "main"
    revision_identifier: str | None = None
    etag: str | None = None


class RevisionDeleteResponse(BaseModel):
    """Response payload for revision deletion."""

    status: str
    repo: str
    revision: str


class UploadUrlRequest(BaseModel):
    """Request payload for generating an upload URL."""

    revision_name: str = "main"
    file_name: str


class UploadUrlResponse(BaseModel):
    """Response payload containing an upload URL and object key."""

    upload_url: str
    object_key: str
    method: str = "PUT"


class FileMetadataModel(BaseModel):
    """Metadata describing a stored file."""

    file_id: str
    path: str
    size: int
    sha: str | None = None
    etag: str | None = None
    object_key: str


class FileListResponseModel(BaseModel):
    """Response payload containing a list of file metadata entries."""

    repo: str
    files: list[FileMetadataModel]
