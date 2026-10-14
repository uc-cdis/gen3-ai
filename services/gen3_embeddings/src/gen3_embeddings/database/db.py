"""
The entry point to the database: the connection pool factory and the composed data access layer.

OVERVIEW
--------

We're using asyncpg alongside FastAPI's dependency injection.

Database manipulation goes through a "data access layer" (DataAccessLayer / DAL) class, so
other areas of the code make simple `.create_*()`, `.list_*()`, `.search_*()` calls without
knowing how to manage connections or interact with the db directly. Connections are drawn
from an asyncpg connection pool, and FastAPI's dependency injection provides a DAL instance
per request.

Each DAL instance is bound to the caller's authz resources for RLS purposes, but it does
not resolve or interpret them: `gen3_embeddings.dependencies` resolves authz at the HTTP
boundary and hands the results in. Nothing in the DAL talks to anything but the database,
and nothing in it raises HTTP errors - see `database/errors.py`.

WHAT THIS MODULE OWNS
---------------------

- `create_pool()`, a factory for the asyncpg connection pool. This module holds no pool itself
    - It builds one from the DB URL in config. The app's lifespan handler calls it once at
      startup and keeps the result on `app.state.db_pool`, mirroring how the Arborist client
      is held, so the pool's lifetime is the app's rather than the module's

- `DataAccessLayer`, which isolates all database manipulations
    - All CRUD and search operations go through this interface instead of
      leaking raw SQL into the higher-level web app endpoint code
    - It is composed from the read, write and search mixins in `database/dal/`. They share
      `DataAccessLayerBase` (`database/dal/base.py`), which holds the pool and the caller's
      authz context and runs every query through `_with_rls()`. How Row Level Security (RLS)
      is wired up, and which policies read which setting, is documented there
    - DAL methods use prepared statements where appropriate as a security and
      efficiency measure
    - Multiple vector types are isolated domains. Collections store a `vector_type`
      (e.g., 'vector', 'halfvec'), each vector type has its own embeddings table (e.g.,
      `embeddings_vector`, `embeddings_halfvec`), and DAL methods route all embedding CRUD
      and search operations to the table for the collection's `vector_type`
    - Search methods use pgvector operators and functions and expose
      a uniform interface with configurable distance metrics, min/max thresholds,
      and filters on metadata

WHERE THE REST LIVES
--------------------

- `database/models.py`: lightweight dataclasses for Collections and Embeddings. They mirror
  rows from the database and provide `.from_record()` helpers to convert from
  asyncpg.Record objects
- `database/errors.py`: the domain errors DAL methods raise. The mapping to status codes
  lives in `gen3_embeddings.error_handlers`
- `gen3_embeddings.dependencies`: the FastAPI dependencies that build a DAL. They live
  outside this package because resolving authz requires a network call to the Gen3 policy
  engine, and the DAL should only ever talk to the database
"""

import asyncpg
from pgvector.asyncpg import register_vector

from gen3_embeddings import config
from gen3_embeddings.config import logging
from gen3_embeddings.database.dal.reads import ReadMixin
from gen3_embeddings.database.dal.search import SearchMixin
from gen3_embeddings.database.dal.writes import WriteMixin


async def create_pool() -> asyncpg.Pool:
    """
    Create the pool of connections.

    Called once per app, from the lifespan handler, which owns the returned pool and holds it
    on `app.state.db_pool`. Nothing here is cached: a second call builds a second pool, so
    request paths must read the one on the app state rather than calling this.

    We have a special initialization to support pgvector columns efficiently.

    See https://github.com/pgvector/pgvector-python

    The `register_vector` adds a custom codec for the `vector` and `halfvec` column types.

    This ensures that when we read from/write to pgvector, it happens
    at the binary level rather than as a string for maximum efficiency.

    Without this, asyncpg defaults to treating it like a string - which is incredibly
    inefficient b/c that's not how it's stored.

    Returns:
        asyncpg.Pool: A new pool, which the caller is responsible for closing.
    """
    logging.info(
        "Initializing connection pool... pool min=%d, pool max=%d", config.PGPOOL_MIN_SIZE, config.PGPOOL_MAX_SIZE
    )
    return await asyncpg.create_pool(
        str(config.DB_CONNECTION_STRING),
        min_size=config.PGPOOL_MIN_SIZE,
        max_size=config.PGPOOL_MAX_SIZE,
        init=register_vector,
    )


class DataAccessLayer(ReadMixin, WriteMixin, SearchMixin):
    """
    Database interface for collections and embeddings, scoped to one caller's authz.

    The methods live in mixins split by what they do to the data: `ReadMixin`
    (`database/dal/reads.py`), `WriteMixin` (`database/dal/writes.py`) and `SearchMixin`
    (`database/dal/search.py`). All three inherit the constructor, the caller's authz context
    and `_with_rls` from `DataAccessLayerBase` (`database/dal/base.py`), so there is still
    exactly one place a connection is taken from the pool.
    """
