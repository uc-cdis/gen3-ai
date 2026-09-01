"""Repository routes for the Gen3 AI model repo service."""

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette import status

from gen3_ai_model_repo.auth import verify_authorization
from gen3_ai_model_repo.constants import DEFAULT_SECURITY_FILE_STATUS
from gen3_ai_model_repo.database.file_tracking import list_files_in_revision
from gen3_ai_model_repo.database.repo_metadata import (
    create_model_metadata,
    delete_model_metadata,
    get_model_metadata,
    list_models,
    update_model_metadata,
)
from gen3_ai_model_repo.database.repo_metadata import model_exists as db_model_exists
from gen3_ai_model_repo.database.revisions import get_revision as db_get_revision
from gen3_ai_model_repo.database.revisions import list_revisions
from gen3_ai_model_repo.models.schemas import (
    DeleteModelResponse,
    RepositoryFileModel,
    RepositoryInfoModel,
    RepositoryMetadataModel,
    RepositoryModel,
    RevisionListResponseModel,
    RevisionModel,
)
from gen3_ai_model_repo.routes.ai_models_shared import RepositoryCreateRequest, RepositoryUpdateRequest
from gen3_ai_model_repo.storage.helpers import get_storage_provider

ai_models_repositories_router = APIRouter()
REPOSITORY_NOT_FOUND_DETAIL = "Repository not found"


@ai_models_repositories_router.get(
    "/api/models",
    response_model=list[RepositoryModel],
    summary="List all model repositories",
    description="Retrieve a list of all available model repositories with basic metadata.",
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved repository list"},
    },
    tags=["Models"],
)
async def list_models_route(
    namespace: str | None = Query(None),
    tags: list[str] | None = Query(None),
    search: str | None = Query(None),
) -> list[RepositoryModel]:
    """
    Retrieve all available model repositories.

    Returns:
        list[RepositoryModel]: A list of all available model repositories.
    """
    repos = await list_models(namespace=namespace, tags=tags, search=search)
    return [
        RepositoryModel(
            id=f"{repo.namespace}/{repo.repo}",
            description=repo.description or "",
            tags=repo.tags,
            created_at=repo.created_at,
        )
        for repo in repos
    ]


@ai_models_repositories_router.get(
    "/api/models/{namespace}/{repo}",
    response_model=RepositoryInfoModel,
    summary="Get repository information",
    description="Retrieve detailed information about a model repository including its metadata and main revision files.",
    tags=["Models"],
)
async def get_repository(namespace: str, repo: str) -> RepositoryInfoModel:
    """
    Retrieve repository information, main revision metadata, and tracked files.

    Returns:
        RepositoryInfoModel: Comprehensive information about the repository.

    Raises:
        HTTPException: If the repository is not found.
    """

    metadata = await get_model_metadata(namespace, repo)
    if not metadata:
        raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND_DETAIL)
    revision_info = await db_get_revision(namespace, repo, "main")
    files_from_db = (
        await list_files_in_revision(namespace=namespace, model_name=repo, revision_name="main")
        if revision_info
        else []
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


@ai_models_repositories_router.post(
    "/api/models/{namespace}/{repo}",
    response_model=RepositoryMetadataModel,
    summary="Create a new model repository",
    description="Create metadata for a new model repository with optional description and tags.",
    tags=["Models"],
)
async def create_repository(
    namespace: str,
    repo: str,
    request: RepositoryCreateRequest,
    _: None = Depends(verify_authorization),
) -> RepositoryMetadataModel:
    """
    Create repository metadata for a new repository.

    Returns:
        RepositoryMetadataModel: The created repository metadata.

    Raises:
        HTTPException: If the repository already exists or creation fails.
    """

    if await db_model_exists(namespace, repo):
        raise HTTPException(status_code=409, detail=f"Repository {namespace}/{repo} already exists")

    return await create_model_metadata(
        namespace=namespace,
        model_name=repo,
        description=request.description or "",
        tags=request.tags,
    )


@ai_models_repositories_router.patch(
    "/api/models/{namespace}/{repo}",
    response_model=RepositoryMetadataModel,
    summary="Update repository metadata",
    description="Update mutable metadata fields such as description and tags for a model repository.",
    tags=["Models"],
)
async def update_repository(
    namespace: str,
    repo: str,
    request: RepositoryUpdateRequest,
    # _: None = Depends(verify_authorization),
) -> RepositoryMetadataModel:
    """
    Update mutable metadata fields for a repository.

    Returns:
        RepositoryMetadataModel: The updated repository metadata.

    Raises:
        HTTPException: If the repository is not found.
    """

    if not await db_model_exists(namespace, repo):
        raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND_DETAIL)

    updated = await update_model_metadata(
        namespace=namespace,
        model_name=repo,
        description=request.description,
        tags=request.tags,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND_DETAIL)
    return updated


@ai_models_repositories_router.delete(
    "/api/models/{namespace}/{repo}",
    response_model=DeleteModelResponse,
    summary="Delete a model repository",
    description="Delete a model repository including its metadata from the database and files from disk.",
    responses={
        status.HTTP_200_OK: {"description": "Model successfully deleted"},
        status.HTTP_401_UNAUTHORIZED: {"description": "User unauthenticated"},
        status.HTTP_403_FORBIDDEN: {"description": "User does not have access"},
        status.HTTP_404_NOT_FOUND: {"description": "Repository not found"},
    },
    tags=["Models"],
)
async def delete_model(namespace: str, repo: str, _: None = Depends(verify_authorization)) -> DeleteModelResponse:
    """
    Delete a model repository.

    Returns:
        DeleteModelResponse: Response indicating successful deletion.

    Raises:
        HTTPException: If the repository is not found or deletion fails.
    """
    repo_exists_check = await db_model_exists(namespace, repo)
    if not repo_exists_check:
        raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND_DETAIL)

    provider = get_storage_provider()
    prefix = f"{namespace}/{repo}/"
    await provider.delete_prefix(prefix)

    deleted_from_db = await delete_model_metadata(namespace, repo)
    if not deleted_from_db:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete repository {namespace}/{repo} from database",
        )

    return DeleteModelResponse(status="deleted", repo=f"{namespace}/{repo}")


@ai_models_repositories_router.get(
    "/api/models/{namespace}/{repo}/revisions",
    response_model=RevisionListResponseModel,
    summary="List model revisions",
    description="Retrieve all revisions of a model repository from the database.",
    responses={
        status.HTTP_200_OK: {"description": "Successfully retrieved revision list"},
        status.HTTP_401_UNAUTHORIZED: {"description": "User unauthenticated"},
        status.HTTP_403_FORBIDDEN: {"description": "User does not have access"},
        status.HTTP_404_NOT_FOUND: {"description": "Repository not found"},
    },
    tags=["Models"],
)
async def list_model_revisions(
    namespace: str, repo: str, _: None = Depends(verify_authorization)
) -> RevisionListResponseModel:
    """
    List all revisions of a model repository.

    Returns:
        RevisionListResponseModel: A list of all revisions for the repository.

    Raises:
        HTTPException: If the repository is not found.
    """

    repo_exists_check = await db_model_exists(namespace, repo)
    if not repo_exists_check:
        raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND_DETAIL)

    revisions_data = await list_revisions(namespace, repo)

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


@ai_models_repositories_router.get(
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

    Returns:
        RepositoryInfoModel: Detailed information about the repository.

    Raises:
        HTTPException: If the repository or metadata is not found.
    """
    repo_exists_in_db = await db_model_exists(namespace, repo)
    if not repo_exists_in_db:
        raise HTTPException(status_code=404, detail=REPOSITORY_NOT_FOUND_DETAIL)

    metadata = await get_model_metadata(namespace, repo)
    if not metadata:
        raise HTTPException(status_code=404, detail="Metadata not found")

    revision_info = await db_get_revision(namespace, repo, "main")

    files_from_db = await list_files_in_revision(
        namespace=namespace,
        model_name=repo,
        revision_name="main",
    )

    total_size = sum(f["size"] for f in files_from_db)

    return RepositoryInfoModel(
        id=f"{namespace}/{repo}",
        sha=(revision_info["sha"] if revision_info else ""),
        etag=((revision_info["etag"] or revision_info["sha"]) if revision_info else ""),
        size=total_size,
        files=[RepositoryFileModel(type=f["type"], oid=f["oid"], size=f["size"]) for f in files_from_db],
        metadata=metadata,
        security_status=DEFAULT_SECURITY_FILE_STATUS,
    )
