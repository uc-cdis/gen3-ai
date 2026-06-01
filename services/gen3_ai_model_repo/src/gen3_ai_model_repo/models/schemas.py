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
