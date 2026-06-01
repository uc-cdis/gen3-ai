from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.database.db import get_db_pool
from gen3_ai_model_repo.models.schemas import RepositoryMetadataModel


async def check_db_connection():
    """
    Helper function to check database connection.
    """
    try:
        db_pool = await get_db_pool()

        async with db_pool.acquire() as connection:
            await connection.execute("SELECT 1")

        logging.debug("Database connection check PASSED.")

        return True

    except Exception:
        logging.exception("Database connection check FAILED.")
        return False


async def create_repository_metadata(
    namespace: str,
    repo_name: str,
    description: str,
    tags: list[str] | None = None,
) -> RepositoryMetadataModel:
    """
    Insert repository metadata into database.
    """

    if tags is None:
        tags = []

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO model_repositories (
                namespace,
                repo_name,
                description,
                tags
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (namespace, repo_name)
            DO UPDATE SET
                description = EXCLUDED.description,
                tags = EXCLUDED.tags,
                updated_at = NOW();
            """,
            namespace,
            repo_name,
            description,
            tags,
        )

    return await get_repository_metadata(
        namespace,
        repo_name,
    )


async def get_repository_metadata(
    namespace: str,
    repo_name: str,
) -> RepositoryMetadataModel | None:
    """
    Retrieve repository metadata from database.
    """

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                namespace,
                repo_name,
                description,
                tags,
                created_at
            FROM model_repositories
            WHERE namespace = $1
            AND repo_name = $2;
            """,
            namespace,
            repo_name,
        )

    if row is None:
        return None

    return RepositoryMetadataModel(
        namespace=row["namespace"],
        repo=row["repo_name"],
        description=row["description"],
        tags=row["tags"],
        created_at=row["created_at"],
    )


async def delete_repository_metadata(
    namespace: str,
    repo_name: str,
) -> bool:
    """
    Delete repository metadata from database.
    """

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM model_repositories
            WHERE namespace = $1
            AND repo_name = $2;
            """,
            namespace,
            repo_name,
        )

    return result == "DELETE 1"


async def repository_exists(
    namespace: str,
    repo_name: str,
) -> bool:
    """
    Check whether a repository exists.
    """

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 1
            FROM model_repositories
            WHERE namespace = $1
            AND repo_name = $2;
            """,
            namespace,
            repo_name,
        )

    return row is not None


async def list_all_repositories() -> list[RepositoryMetadataModel]:
    """
    Return all repositories.
    """

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                namespace,
                repo_name,
                description,
                tags,
                created_at
            FROM model_repositories;
            """
        )

    return [
        RepositoryMetadataModel(
            namespace=row["namespace"],
            repo=row["repo_name"],
            description=row["description"],
            tags=row["tags"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


async def update_repository_metadata(
    namespace: str,
    repo_name: str,
    description: str | None = None,
    tags: list[str] | None = None,
) -> RepositoryMetadataModel | None:
    """
    Update repository metadata.
    """

    pool = await get_db_pool()

    set_clauses = []
    values = []

    if description is not None:
        values.append(description)
        set_clauses.append(f"description = ${len(values)}")

    if tags is not None:
        values.append(tags)
        set_clauses.append(f"tags = ${len(values)}")

    if not set_clauses:
        return await get_repository_metadata(
            namespace,
            repo_name,
        )

    values.extend([namespace, repo_name])

    async with pool.acquire() as conn:
        await conn.execute(
            f"""
            UPDATE model_repositories
            SET {", ".join(set_clauses)},
                updated_at = NOW()
            WHERE namespace = ${len(values) - 1}
            AND repo_name = ${len(values)};
            """,
            *values,
        )

    return await get_repository_metadata(
        namespace,
        repo_name,
    )


# Revision Management Functions


async def get_or_create_revision(
    namespace: str,
    repo_name: str,
    revision_name: str = "main",
    commit_sha: str | None = None,
    etag: str | None = None,
) -> dict | None:
    """
    Get or create a revision for a repository.
    If commit_sha is provided and revision doesn't exist, create it.
    """
    pool = await get_db_pool()

    # Get repository ID
    async with pool.acquire() as conn:
        repo_row = await conn.fetchrow(
            """
            SELECT id FROM model_repositories
            WHERE namespace = $1 AND repo_name = $2;
            """,
            namespace,
            repo_name,
        )

    if not repo_row:
        return None

    repo_id = repo_row["id"]

    async with pool.acquire() as conn:
        # Try to get existing revision
        revision_row = await conn.fetchrow(
            """
            SELECT id, commit_sha, etag, created_at
            FROM model_revisions
            WHERE repository_id = $1 AND revision_name = $2;
            """,
            repo_id,
            revision_name,
        )

        if revision_row:
            return {
                "id": revision_row["id"],
                "revision": revision_name,
                "sha": revision_row["commit_sha"],
                "etag": revision_row["etag"],
                "created_at": revision_row["created_at"],
            }

        # Create new revision if commit_sha is provided
        if commit_sha:
            await conn.execute(
                """
                INSERT INTO model_revisions (repository_id, revision_name, commit_sha, etag)
                VALUES ($1, $2, $3, $4);
                """,
                repo_id,
                revision_name,
                commit_sha,
                etag,
            )

            revision_row = await conn.fetchrow(
                """
                SELECT id, commit_sha, etag, created_at
                FROM model_revisions
                WHERE repository_id = $1 AND revision_name = $2;
                """,
                repo_id,
                revision_name,
            )

            return {
                "id": revision_row["id"],
                "revision": revision_name,
                "sha": revision_row["commit_sha"],
                "etag": revision_row["etag"],
                "created_at": revision_row["created_at"],
            }

    return None


async def list_revisions(
    namespace: str,
    repo_name: str,
) -> list[dict]:
    """
    List all revisions for a repository.
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        repo_row = await conn.fetchrow(
            """
            SELECT id FROM model_repositories
            WHERE namespace = $1 AND repo_name = $2;
            """,
            namespace,
            repo_name,
        )

        if not repo_row:
            return []

        repo_id = repo_row["id"]

        revision_rows = await conn.fetch(
            """
            SELECT id, revision_name, commit_sha, etag, created_at
            FROM model_revisions
            WHERE repository_id = $1
            ORDER BY created_at DESC;
            """,
            repo_id,
        )

    return [
        {
            "id": row["id"],
            "revision": row["revision_name"],
            "sha": row["commit_sha"],
            "etag": row["etag"],
            "created_at": row["created_at"],
        }
        for row in revision_rows
    ]


# File Tracking Functions


async def track_file(
    namespace: str,
    repo_name: str,
    revision_name: str,
    file_path: str,
    file_size: int,
    content_sha: str,
    content_etag: str | None = None,
) -> bool:
    """
    Track a file in a revision.
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        # Get repository ID
        repo_row = await conn.fetchrow(
            """
            SELECT id FROM model_repositories
            WHERE namespace = $1 AND repo_name = $2;
            """,
            namespace,
            repo_name,
        )

        if not repo_row:
            return False

        repo_id = repo_row["id"]

        # Get revision ID
        revision_row = await conn.fetchrow(
            """
            SELECT id FROM model_revisions
            WHERE repository_id = $1 AND revision_name = $2;
            """,
            repo_id,
            revision_name,
        )

        if not revision_row:
            return False

        revision_id = revision_row["id"]

        # Insert or update file record
        await conn.execute(
            """
            INSERT INTO model_files (revision_id, file_path, file_size, content_sha, content_etag)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (revision_id, file_path)
            DO UPDATE SET
                file_size = EXCLUDED.file_size,
                content_sha = EXCLUDED.content_sha,
                content_etag = EXCLUDED.content_etag;
            """,
            revision_id,
            file_path,
            file_size,
            content_sha,
            content_etag,
        )

    return True


async def list_files_in_revision(
    namespace: str,
    repo_name: str,
    revision_name: str = "main",
) -> list[dict]:
    """
    List all files in a specific revision.
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        repo_row = await conn.fetchrow(
            """
            SELECT id FROM model_repositories
            WHERE namespace = $1 AND repo_name = $2;
            """,
            namespace,
            repo_name,
        )

        if not repo_row:
            return []

        repo_id = repo_row["id"]

        revision_row = await conn.fetchrow(
            """
            SELECT id FROM model_revisions
            WHERE repository_id = $1 AND revision_name = $2;
            """,
            repo_id,
            revision_name,
        )

        if not revision_row:
            return []

        revision_id = revision_row["id"]

        file_rows = await conn.fetch(
            """
            SELECT file_path, file_size, content_sha, content_etag, created_at
            FROM model_files
            WHERE revision_id = $1
            ORDER BY file_path;
            """,
            revision_id,
        )

    return [
        {
            "path": row["file_path"],
            "size": row["file_size"],
            "oid": row["content_sha"],
            "etag": row["content_etag"],
            "type": "file",
            "created_at": row["created_at"],
        }
        for row in file_rows
    ]
