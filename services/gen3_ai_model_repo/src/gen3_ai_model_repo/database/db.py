import logging

import asyncpg

from gen3_ai_model_repo.config import (
    DB_DATABASE,
    DB_HOST,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
)

db_pool = None


async def connect_db():
    """
    Create asyncpg connection pool.
    """

    global db_pool

    logging.info("Connecting to PostgreSQL")

    db_pool = await asyncpg.create_pool(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_DATABASE,
        host=DB_HOST,
        port=DB_PORT,
        min_size=10,
        max_size=10,
    )

    logging.info("PostgreSQL connection pool initialized")


async def close_db():
    """
    Close asyncpg connection pool.
    """

    global db_pool

    if db_pool:
        logging.info("Closing PostgreSQL connection pool")

        await db_pool.close()


async def get_db_pool():
    """
    Return initialized database connection pool.
    """

    global db_pool

    if db_pool is None:
        await connect_db()

    return db_pool
