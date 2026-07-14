"""
Revision management database operations.

Handles CRUD operations for model revisions, including creating,
retrieving, and listing revisions with commit SHAs and ETags.
"""

from gen3_ai_model_repo.database.db import get_db_pool

GET_MODEL_ID_SQL = """
    SELECT id FROM models
    WHERE namespace = $1 AND model_name = $2;
"""


async def get_revision_identifier_column(conn) -> str:
    """
    Return the revision identifier column used by the live database.

    Older local databases used commit_sha while the current migration uses
    revision_identifier. Keeping this lookup here lets route and helper code work
    across already-created developer databases without hiding real SQL errors.
    """
    stmt = await conn.prepare(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'model_revisions'
          AND column_name IN ('revision_identifier', 'commit_sha')
        ORDER BY CASE column_name
            WHEN 'revision_identifier' THEN 1
            WHEN 'commit_sha' THEN 2
        END
        LIMIT 1;
        """
    )
    row = await stmt.fetchrow()
    if row is None:
        raise RuntimeError("model_revisions is missing revision identifier column")
    return row["column_name"]


async def create_revision(
    namespace: str,
    model_name: str,
    revision_name: str = "main",
    revision_identifier: str | None = None,
    etag: str | None = None,
) -> dict | None:
    """Create or update a revision for a repository."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        identifier_column = await get_revision_identifier_column(conn)
        model_stmt = await conn.prepare(GET_MODEL_ID_SQL)
        model_row = await model_stmt.fetchrow(namespace, model_name)
        if not model_row:
            return None
        model_id = model_row["id"]
        insert_stmt = await conn.prepare(
            f"""
            INSERT INTO model_revisions (model_id, revision_name, {identifier_column}, etag)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (model_id, revision_name)
            DO UPDATE SET {identifier_column} = EXCLUDED.{identifier_column},
                          etag = EXCLUDED.etag;
            """
        )
        await insert_stmt.fetch(model_id, revision_name, revision_identifier, etag)
    return await get_revision(namespace, model_name, revision_name)


async def get_revision(
    namespace: str,
    model_name: str,
    revision_name: str,
) -> dict | None:
    """Return the metadata for a specific revision, if present."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        identifier_column = await get_revision_identifier_column(conn)
        stmt = await conn.prepare(
            f"""
            SELECT mr.id, mr.revision_name, mr.{identifier_column} AS revision_identifier, mr.etag, mr.created_at
            FROM model_revisions mr
            JOIN models repo ON repo.id = mr.model_id
            WHERE repo.namespace = $1 AND repo.model_name = $2 AND mr.revision_name = $3;
            """
        )
        row = await stmt.fetchrow(namespace, model_name, revision_name)
    if not row:
        return None
    return {
        "id": row["id"],
        "revision": row["revision_name"],
        "sha": row["revision_identifier"],
        "etag": row["etag"],
        "created_at": row["created_at"],
    }


async def get_or_create_revision(
    namespace: str,
    model_name: str,
    revision_name: str = "main",
    commit_sha: str | None = None,
    etag: str | None = None,
) -> dict | None:
    """
    Get or create a revision for a repository.

    If the revision exists, returns it. If commit_sha is provided and the
    revision doesn't exist, creates a new revision.

    Args:
        namespace (str): The namespace/organization for the repository.
        model_name (str): The name of the model.
        revision_name (str): The name of the revision (default: "main").
        commit_sha (str | None): Optional commit SHA for creating a new revision.
        etag (str | None): Optional ETag for the revision.

    Returns:
        A dictionary containing revision details (id, revision, sha, etag, created_at),
        or None if the repository doesn't exist or revision couldn't be created.
    """
    pool = await get_db_pool()

    # Get model ID
    async with pool.acquire() as conn:
        model_stmt = await conn.prepare(GET_MODEL_ID_SQL)
        model_row = await model_stmt.fetchrow(namespace, model_name)

    if not model_row:
        return None

    model_id = model_row["id"]

    async with pool.acquire() as conn:
        identifier_column = await get_revision_identifier_column(conn)
        # Try to get existing revision
        revision_stmt = await conn.prepare(
            f"""
            SELECT id, {identifier_column} AS revision_identifier, etag, created_at
            FROM model_revisions
            WHERE model_id = $1 AND revision_name = $2;
            """
        )
        revision_row = await revision_stmt.fetchrow(model_id, revision_name)

        if revision_row:
            return {
                "id": revision_row["id"],
                "revision": revision_name,
                "sha": revision_row["revision_identifier"],
                "etag": revision_row["etag"],
                "created_at": revision_row["created_at"],
            }

        # Create new revision if commit_sha is provided
        if commit_sha:
            insert_stmt = await conn.prepare(
                f"""
                INSERT INTO model_revisions (model_id, revision_name, {identifier_column}, etag)
                VALUES ($1, $2, $3, $4);
                """
            )
            await insert_stmt.fetch(model_id, revision_name, commit_sha, etag)

            revision_stmt = await conn.prepare(
                f"""
                SELECT id, {identifier_column} AS revision_identifier, etag, created_at
                FROM model_revisions
                WHERE model_id = $1 AND revision_name = $2;
                """
            )
            revision_row = await revision_stmt.fetchrow(model_id, revision_name)

            return {
                "id": revision_row["id"],
                "revision": revision_name,
                "sha": revision_row["revision_identifier"],
                "etag": revision_row["etag"],
                "created_at": revision_row["created_at"],
            }

    return None


async def list_revisions(
    namespace: str,
    model_name: str,
) -> list[dict]:
    """
    List all revisions for a repository.

    Retrieves all revisions for a given repository, ordered by creation date (newest first).

    Args:
        namespace: The namespace/organization for the repository.
        model_name: The name of the model.

    Returns:
        A list of dictionaries containing revision details, or an empty list if
        the repository doesn't exist.
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        identifier_column = await get_revision_identifier_column(conn)
        model_stmt = await conn.prepare(GET_MODEL_ID_SQL)
        model_row = await model_stmt.fetchrow(namespace, model_name)

        if not model_row:
            return []

        model_id = model_row["id"]

        revision_stmt = await conn.prepare(
            f"""
            SELECT id, revision_name, {identifier_column} AS revision_identifier, etag, created_at
            FROM model_revisions
            WHERE model_id = $1
            ORDER BY created_at DESC;
            """
        )
        revision_rows = await revision_stmt.fetch(model_id)

    return [
        {
            "id": row["id"],
            "revision": row["revision_name"],
            "sha": row["revision_identifier"],
            "etag": row["etag"],
            "created_at": row["created_at"],
        }
        for row in revision_rows
    ]


async def delete_revision(
    namespace: str,
    model_name: str,
    revision_name: str,
) -> bool:
    """Delete a revision for a repository."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        stmt = await conn.prepare(
            """
            DELETE FROM model_revisions
            WHERE id IN (
                SELECT mr.id
                FROM model_revisions mr
                                JOIN models repo ON repo.id = mr.model_id
                WHERE repo.namespace = $1
                                    AND repo.model_name = $2
                  AND mr.revision_name = $3
                        )
                        RETURNING 1;
            """
        )
        deleted = await stmt.fetchval(namespace, model_name, revision_name)
    return deleted is not None
