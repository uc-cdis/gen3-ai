"""
Revision management database operations.

Handles CRUD operations for model revisions, including creating,
retrieving, and listing revisions with commit SHAs and ETags.
"""

from gen3_ai_model_repo.database.db import get_db_pool


async def get_or_create_revision(
    namespace: str,
    repo_name: str,
    revision_name: str = "main",
    commit_sha: str | None = None,
    etag: str | None = None,
) -> dict | None:
    """
    Get or create a revision for a repository.

    If the revision exists, returns it. If commit_sha is provided and the
    revision doesn't exist, creates a new revision.

    Args:
        namespace: The namespace/organization for the repository.
        repo_name: The name of the repository.
        revision_name: The name of the revision (default: "main").
        commit_sha: Optional commit SHA for creating a new revision.
        etag: Optional ETag for the revision.

    Returns:
        A dictionary containing revision details (id, revision, sha, etag, created_at),
        or None if the repository doesn't exist or revision couldn't be created.
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

    Retrieves all revisions for a given repository, ordered by creation date (newest first).

    Args:
        namespace: The namespace/organization for the repository.
        repo_name: The name of the repository.

    Returns:
        A list of dictionaries containing revision details, or an empty list if
        the repository doesn't exist.
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
