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
    """
    return RepositoryMetadataModel(
        namespace=data["namespace"],
        repo=data.get("repo") or data.get("repo_name"),
        description=data["description"],
        tags=data.get("tags", []),
        created_at=data.get("created_at", ""),
    )


def revision_to_model(revision_data: dict) -> RevisionModel:
    """
    Convert revision data into API schema.
    """
    return RevisionModel(
        id=revision_data["id"],
        revision=revision_data["revision"],
        sha=revision_data["sha"],
    )


def repository_file_to_model(file_data: dict) -> TreeEntryModel:
    """
    Convert repository file metadata into API schema.
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
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        identifier_column = await get_revision_identifier_column(conn)
        await conn.execute(
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
            """,
            commit_sha,
            etag,
            namespace,
            repo_name,
            revision_name,
        )
