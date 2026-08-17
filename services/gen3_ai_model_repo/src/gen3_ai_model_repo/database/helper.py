from gen3_ai_model_repo.database.db import get_db_pool


async def check_db_connection():
    """
    Helper function to check database connection.
    """
    try:
        db_pool = await get_db_pool()
        async with db_pool.acquire() as connection:
            await connection.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"Database connection check failed: {e}")
        return False
