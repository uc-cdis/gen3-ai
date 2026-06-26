from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.database.db import get_db_pool

# Re-export file tracking functions
from gen3_ai_model_repo.database.file_tracking import (
    delete_file,
    delete_files_for_revision,
    get_file_record,
    list_files_in_revision,
    track_file,
)

# Re-export repository metadata functions
from gen3_ai_model_repo.database.repo_metadata import (
    create_repository_metadata,
    delete_repository_metadata,
    get_repository,
    get_repository_metadata,
    list_all_repositories,
    list_repositories,
    repository_exists,
    update_repository_metadata,
)

# Re-export revision management functions
from gen3_ai_model_repo.database.revisions import (
    create_revision,
    delete_revision,
    get_or_create_revision,
    get_revision,
    list_revisions,
)

__all__ = [
    # Repository metadata
    "create_repository_metadata",
    "delete_repository_metadata",
    "get_repository",
    "get_repository_metadata",
    "list_all_repositories",
    "list_repositories",
    "repository_exists",
    "update_repository_metadata",
    # Revisions
    "create_revision",
    "delete_revision",
    "get_revision",
    "get_or_create_revision",
    "list_revisions",
    # File tracking
    "list_files_in_revision",
    "delete_file",
    "delete_files_for_revision",
    "get_file_record",
    "track_file",
    # Utilities
    "check_db_connection",
]


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
