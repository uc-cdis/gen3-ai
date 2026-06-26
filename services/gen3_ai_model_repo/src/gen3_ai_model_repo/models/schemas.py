from datetime import datetime

from pydantic import BaseModel


class RepositoryModel(BaseModel):
    id: str
    description: str
    tags: list[str] = []
    created_at: datetime | None = None


class RepositoryFileModel(BaseModel):
    type: str
    oid: str
    size: int


class RepositoryMetadataModel(BaseModel):
    namespace: str
    repo: str
    description: str | None = None
    tags: list[str] = []
    created_at: datetime | None = None


class UploadModelResponse(BaseModel):
    status: str
    repo: str
    metadata_file: str
    metadata: RepositoryMetadataModel


class RepositoryInfoModel(BaseModel):
    id: str
    sha: str
    etag: str
    size: int
    files: list[RepositoryFileModel]
    metadata: RepositoryMetadataModel
    security_status: dict


class RevisionModel(BaseModel):
    id: str
    revision: str
    sha: str


class RevisionListResponseModel(BaseModel):
    repo: str
    revisions: list[RevisionModel]


class TreeEntryModel(BaseModel):
    type: str
    oid: str
    size: int


class DeleteModelResponse(BaseModel):
    status: str
    repo: str


class RevisionCreateRequest(BaseModel):
    revision_name: str = "main"
    revision_identifier: str | None = None
    etag: str | None = None


class RevisionDeleteResponse(BaseModel):
    status: str
    repo: str
    revision: str


class UploadUrlRequest(BaseModel):
    revision_name: str = "main"
    file_name: str


class UploadUrlResponse(BaseModel):
    upload_url: str
    object_key: str


class FileMetadataModel(BaseModel):
    file_id: str
    path: str
    size: int
    sha: str | None = None
    etag: str | None = None
    s3_key: str


class FileListResponseModel(BaseModel):
    repo: str
    files: list[FileMetadataModel]
