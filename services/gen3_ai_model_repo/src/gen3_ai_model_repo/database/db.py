"""Database connectivity helpers for the Gen3 AI model repo service."""

import asyncpg

from gen3_ai_model_repo.config import (
    DB_CONNECTION_STRING,
    PGDATABASE,
    PGHOST,
    PGPASSWORD,
    PGPOOL_MAX_SIZE,
    PGPOOL_MIN_SIZE,
    PGPORT,
    PGUSER,
    logging,
)

db_pool = None


async def connect_db():
    """
    Create asyncpg connection pool.
    """

    global db_pool

    logging.info("Connecting to PostgreSQL")

    connection_uri = str(DB_CONNECTION_STRING) if DB_CONNECTION_STRING else ""

    if connection_uri:
        db_pool = await asyncpg.create_pool(
            dsn=connection_uri,
            min_size=PGPOOL_MIN_SIZE,
            max_size=PGPOOL_MAX_SIZE,
        )
    else:
        db_pool = await asyncpg.create_pool(
            user=PGUSER,
            password=PGPASSWORD,
            database=PGDATABASE,
            host=PGHOST,
            port=PGPORT,
            min_size=PGPOOL_MIN_SIZE,
            max_size=PGPOOL_MAX_SIZE,
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
        db_pool = None


async def get_db_pool():
    """
    Return initialized database connection pool.

    Returns:
        asyncpg.Pool: The initialized database connection pool.
    """

    global db_pool

    if db_pool is None:
        await connect_db()

    return db_pool
