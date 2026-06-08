"""
File tracking database operations.

Handles file metadata tracking within model revisions, including
adding, retrieving, and listing files associated with revisions.
"""

from gen3_ai_model_repo.database.db import get_db_pool


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
