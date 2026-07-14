from pydantic import BaseModel


class RepositoryCreateRequest(BaseModel):
    """
    Request payload for creating a repository.

    Attributes:
        description: Optional human-readable repository description.
        tags: Optional list of tags used for filtering and discovery.
    """

    description: str | None = None
    tags: list[str] = []


class MultipartUploadResponse(BaseModel):
    """
    Response payload for multipart model uploads.

    Attributes:
        status: Upload status string.
        repo: Repository identifier in namespace/repo form.
        revision: Revision name associated with the upload.
        files: Number of uploaded files.
        total_size: Total uploaded size in bytes.
    """

    status: str
    repo: str
    revision: str
    files: int
    total_size: int


async def repository_has_current_revision(conn) -> bool:
    """
    Check whether the models table has a current_revision column.

    Args:
        conn: Active database connection.

    Returns:
        True if the current_revision column exists, otherwise False.
    """

    stmt = await conn.prepare(
        """
        SELECT 1
        FROM information_schema.columns
                WHERE table_name = 'models'
          AND column_name = 'current_revision';
        """
    )
    return bool(await stmt.fetchval())


async def model_files_optional_columns(conn) -> tuple[bool, bool]:
    """
    Detect optional columns in the model_files table.

    Args:
        conn: Active database connection.

    Returns:
        Tuple[bool, bool]: Flags for s3_key and file_type column presence.
    """

    stmt = await conn.prepare(
        """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'model_files'
                    AND column_name IN ('s3_key', 'file_type');
                """
    )
    rows = await stmt.fetch()
    column_names = {row["column_name"] for row in rows}
    return "s3_key" in column_names, "file_type" in column_names
