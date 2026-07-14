"""
Repository metadata database operations.

Handles CRUD operations for repository metadata including
namespace, repository name, description, and tags.
"""

from gen3_ai_model_repo.database.db import get_db_pool
from gen3_ai_model_repo.models.schemas import RepositoryMetadataModel


async def create_model_metadata(
    namespace: str,
    model_name: str,
    description: str,
    tags: list[str] | None = None,
) -> RepositoryMetadataModel:
    """
    Insert repository metadata into database.

    Creates a new repository metadata entry or updates an existing one
    if a repository with the same namespace and repo_name already exists.

    Args:
        namespace: The namespace/organization for the repository.
        repo_name: The name of the repository.
        description: A description of the repository.
        tags: Optional list of tags for the repository.

    Returns:
        RepositoryMetadataModel: The created or updated repository metadata.
    """
    if tags is None:
        tags = []

    pool = await get_db_pool()

    async with pool.acquire() as conn:
        stmt = await conn.prepare(
            """
            INSERT INTO models (
                namespace,
                model_name,
                description,
                tags
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (namespace, model_name)
            DO UPDATE SET
                description = EXCLUDED.description,
                tags = EXCLUDED.tags,
                updated_at = NOW();
            """
        )
        await stmt.fetch(namespace, model_name, description, tags)

    return await get_model_metadata(
        namespace,
        model_name,
    )


async def get_model_metadata(
    namespace: str,
    model_name: str,
) -> RepositoryMetadataModel | None:
    """
    Retrieve repository metadata from database.

    Args:
        namespace (str): The namespace/organization for the repository.
        repo_name (str): The name of the repository.

    Returns:
        RepositoryMetadataModel if found, None otherwise.
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        stmt = await conn.prepare(
            """
            SELECT
                namespace,
                model_name AS repo_name,
                description,
                tags,
                created_at
            FROM models
            WHERE namespace = $1
            AND model_name = $2;
            """
        )
        row = await stmt.fetchrow(namespace, model_name)

    if row is None:
        return None

    return RepositoryMetadataModel(
        namespace=row["namespace"],
        repo=row["repo_name"],
        description=row["description"],
        tags=row["tags"],
        created_at=row["created_at"],
    )


async def delete_model_metadata(
    namespace: str,
    model_name: str,
) -> bool:
    """
    Delete repository metadata from database.

    Args:
        namespace: The namespace/organization for the repository.
        repo_name: The name of the repository.

    Returns:
        True if deletion was successful, False otherwise.
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        stmt = await conn.prepare(
            """
            DELETE FROM models
            WHERE namespace = $1
            AND model_name = $2
            RETURNING 1;
            """
        )
        deleted = await stmt.fetchval(namespace, model_name)

    return deleted is not None


async def model_exists(
    namespace: str,
    model_name: str,
) -> bool:
    """
    Check whether a repository exists in the database.

    Args:
        namespace: The namespace/organization for the repository.
        repo_name: The name of the repository.

    Returns:
        True if the repository exists, False otherwise.
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        stmt = await conn.prepare(
            """
            SELECT 1
            FROM models
            WHERE namespace = $1
            AND model_name = $2;
            """
        )
        row = await stmt.fetchrow(namespace, model_name)

    return row is not None


async def list_all_models() -> list[RepositoryMetadataModel]:
    """
    Return all repositories from the database.

    Returns:
        A list of RepositoryMetadataModel objects for all repositories.
    """
    pool = await get_db_pool()

    async with pool.acquire() as conn:
        stmt = await conn.prepare(
            """
            SELECT
                namespace,
                model_name AS repo_name,
                description,
                tags,
                created_at
            FROM models;
            """
        )
        rows = await stmt.fetch()

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


async def get_model(
    namespace: str,
    model_name: str,
) -> RepositoryMetadataModel | None:
    """Return repository metadata by namespace and repository name."""
    return await get_model_metadata(namespace, model_name)


async def list_models(
    namespace: str | None = None,
    tags: list[str] | None = None,
    search: str | None = None,
) -> list[RepositoryMetadataModel]:
    """List repositories, optionally filtered by namespace, tags, or free-text search."""
    pool = await get_db_pool()
    clauses = []
    values: list[object] = []
    if namespace:
        values.append(namespace)
        clauses.append(f"namespace = ${len(values)}")
    if tags:
        values.append(tags)
        clauses.append(f"tags && ${len(values)}")
    if search:
        values.append(f"%{search}%")
        clauses.append(f"(model_name ILIKE ${len(values)} OR description ILIKE ${len(values)})")
    sql = """
        SELECT namespace, model_name AS repo_name, description, tags, created_at
        FROM models
    """
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC;"
    async with pool.acquire() as conn:
        stmt = await conn.prepare(sql)
        rows = await stmt.fetch(*values)
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


async def update_model_metadata(
    namespace: str,
    model_name: str,
    description: str | None = None,
    tags: list[str] | None = None,
) -> RepositoryMetadataModel | None:
    """
    Update repository metadata.

    Selectively updates description and/or tags for an existing repository.
    Only provided fields will be updated.

    Args:
        namespace: The namespace/organization for the repository.
        repo_name: The name of the repository.
        description: Optional new description for the repository.
        tags: Optional new list of tags for the repository.

    Returns:
        RepositoryMetadataModel of the updated repository, or None if not found.
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
        return await get_model_metadata(
            namespace,
            model_name,
        )

    values.extend([namespace, model_name])

    async with pool.acquire() as conn:
        stmt = await conn.prepare(
            f"""
            UPDATE models
            SET {", ".join(set_clauses)},
                updated_at = NOW()
            WHERE namespace = ${len(values) - 1}
            AND model_name = ${len(values)};
            """
        )
        await stmt.fetch(*values)

    return await get_model_metadata(
        namespace,
        model_name,
    )
