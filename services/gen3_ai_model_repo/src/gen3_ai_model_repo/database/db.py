"""Database connectivity helpers for the Gen3 AI model repo service."""

from dataclasses import dataclass
from datetime import datetime

import asyncpg

from gen3_ai_model_repo.config import (
    DB_CONNECTION_STRING,
    DB_DATABASE,
    DB_HOST,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
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
            min_size=10,
            max_size=10,
        )
    else:
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


@dataclass
class ModelRepository:
    """
    Dataclass representing a models table row.
    """

    id: int
    namespace: str
    model_name: str
    description: str | None
    tags: list[str]
    current_revision: str
    created_at: datetime
    updated_at: datetime
