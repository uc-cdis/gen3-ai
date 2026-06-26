"""
File tracking database operations.

Handles file metadata tracking within model revisions, including
adding, retrieving, and listing files associated with revisions.
"""

from gen3_ai_model_repo.database.db import get_db_pool


async def model_files_has_s3_key(conn) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'model_files'
              AND column_name = 's3_key';
            """
        )
    )


async def track_file(
    namespace: str,
    repo_name: str,
    revision_name: str,
    file_path: str,
    file_size: int,
    content_sha: str,
    content_etag: str | None = None,
    s3_key: str | None = None,
) -> bool:
    """
    Track a file in a revision.

    Adds or updates a file record in the database for a specific revision.
    If the file already exists in the revision, it will be updated.

    Args:
        namespace: The namespace/organization for the repository.
        repo_name: The name of the repository.
        revision_name: The name of the revision containing the file.
        file_path: The path to the file within the revision.
        file_size: The size of the file in bytes.
        content_sha: The SHA hash of the file content.
        content_etag: Optional ETag for the file.

    Returns:
        True if the file was successfully tracked, False otherwise.
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        has_s3_key = await model_files_has_s3_key(conn)
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
        if has_s3_key:
            await conn.execute(
                """
                INSERT INTO model_files (revision_id, file_path, file_size, content_sha, content_etag, s3_key)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (revision_id, file_path)
                DO UPDATE SET
                    file_size = EXCLUDED.file_size,
                    content_sha = EXCLUDED.content_sha,
                    content_etag = EXCLUDED.content_etag,
                    s3_key = EXCLUDED.s3_key;
                """,
                revision_id,
                file_path,
                file_size,
                content_sha,
                content_etag,
                s3_key,
            )
        else:
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

    Retrieves metadata for all files in a given revision, ordered by file path.

    Args:
        namespace: The namespace/organization for the repository.
        repo_name: The name of the repository.
        revision_name: The name of the revision (default: "main").

    Returns:
        A list of dictionaries containing file metadata (path, size, oid, etag, type, created_at),
        or an empty list if the repository or revision doesn't exist.
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


async def get_file_record(
    namespace: str,
    repo_name: str,
    revision_name: str,
    file_path: str,
) -> dict | None:
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        has_s3_key = await model_files_has_s3_key(conn)
        if has_s3_key:
            row = await conn.fetchrow(
                """
                SELECT
                    mf.file_path,
                    mf.file_size,
                    mf.content_sha,
                    mf.content_etag,
                    mf.s3_key
                FROM model_files mf
                JOIN model_revisions mr ON mr.id = mf.revision_id
                JOIN model_repositories repo ON repo.id = mr.repository_id
                WHERE repo.namespace = $1
                  AND repo.repo_name = $2
                  AND mr.revision_name = $3
                  AND mf.file_path = $4;
                """,
                namespace,
                repo_name,
                revision_name,
                file_path,
            )
        else:
            row = await conn.fetchrow(
                """
                SELECT
                    mf.file_path,
                    mf.file_size,
                    mf.content_sha,
                    mf.content_etag
                FROM model_files mf
                JOIN model_revisions mr ON mr.id = mf.revision_id
                JOIN model_repositories repo ON repo.id = mr.repository_id
                WHERE repo.namespace = $1
                  AND repo.repo_name = $2
                  AND mr.revision_name = $3
                  AND mf.file_path = $4;
                """,
                namespace,
                repo_name,
                revision_name,
                file_path,
            )

    if row is None:
        return None

    return {
        "path": row["file_path"],
        "size": row["file_size"],
        "sha": row["content_sha"],
        "etag": row["content_etag"],
        "s3_key": row["s3_key"] if has_s3_key else f"{namespace}/{repo_name}/{revision_name}/{file_path}",
    }


async def delete_file(
    namespace: str,
    repo_name: str,
    revision_name: str,
    file_path: str,
) -> bool:
    pool = await get_db_pool()
    assert pool is not None, "Database pool is not initialized"
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM model_files
            WHERE id IN (
                SELECT mf.id
                FROM model_files mf
                JOIN model_revisions mr ON mr.id = mf.revision_id
                JOIN model_repositories repo ON repo.id = mr.repository_id
                WHERE repo.namespace = $1
                  AND repo.repo_name = $2
                  AND mr.revision_name = $3
                  AND mf.file_path = $4
            );
            """,
            namespace,
            repo_name,
            revision_name,
            file_path,
        )
    return result == "DELETE 1"


async def delete_files_for_revision(
    namespace: str,
    repo_name: str,
    revision_name: str,
) -> bool:
    pool = await get_db_pool()
    assert pool is not None, "Database pool is not initialized"
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM model_files
            WHERE revision_id IN (
                SELECT mr.id
                FROM model_revisions mr
                JOIN model_repositories repo ON repo.id = mr.repository_id
                WHERE repo.namespace = $1
                  AND repo.repo_name = $2
                  AND mr.revision_name = $3
            );
            """,
            namespace,
            repo_name,
            revision_name,
        )
    return result.startswith("DELETE")
