from gen3_ai_model_repo.config import logging
from gen3_ai_model_repo.database.db import get_db_pool

__all__ = ["check_db_connection"]


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
