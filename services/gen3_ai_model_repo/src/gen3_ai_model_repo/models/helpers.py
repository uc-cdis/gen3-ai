"""Model helper utilities for the Gen3 AI model repo service."""

from gen3_ai_model_repo.database.db import get_db_pool
from gen3_ai_model_repo.database.revisions import get_revision_identifier_column
from gen3_ai_model_repo.models.schemas import (
    RepositoryMetadataModel,
    RevisionModel,
    TreeEntryModel,
)


def repository_metadata_to_model(data: dict) -> RepositoryMetadataModel:
    """
    Convert repository metadata dict into schema model.

    Returns:
        RepositoryMetadataModel: The converted repository metadata model.

    Raises:
        KeyError: If repository metadata is missing a repo name.
    """
    repo_name = data.get("repo")
    if repo_name is None:
        repo_name = data.get("repo_name")
    if repo_name is None:
        raise KeyError("Repository metadata is missing a repo name")

    return RepositoryMetadataModel(
        namespace=data["namespace"],
        repo=str(repo_name),
        description=data["description"],
        tags=data.get("tags", []),
        created_at=data.get("created_at", ""),
    )


def revision_to_model(revision_data: dict) -> RevisionModel:
    """
    Convert revision data into API schema.

    Returns:
        RevisionModel: The converted revision model.
    """
    return RevisionModel(
        id=revision_data["id"],
        revision=revision_data["revision"],
        sha=revision_data["sha"],
    )


def repository_file_to_model(file_data: dict) -> TreeEntryModel:
    """
    Convert repository file metadata into API schema.

    Returns:
        TreeEntryModel: The converted tree entry model.
    """
    return TreeEntryModel(
        type=file_data["type"],
        oid=file_data["oid"],
        size=file_data["size"],
    )


async def update_revision_commit(
    namespace: str,
    repo_name: str,
    revision_name: str,
    commit_sha: str,
    etag: str | None = None,
):
    """Update the commit and ETag for a repository revision."""
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        identifier_column = await get_revision_identifier_column(conn)
        stmt = await conn.prepare(
            f"""
            UPDATE model_revisions
            SET {identifier_column} = $1,
                etag = $2
            WHERE id = (
                SELECT mr.id
                FROM model_revisions mr
                JOIN model_repositories repo ON repo.id = mr.repository_id
                WHERE repo.namespace = $3
                  AND repo.repo_name = $4
                  AND mr.revision_name = $5
            );
            """
        )
        await stmt.fetch(commit_sha, etag, namespace, repo_name, revision_name)
