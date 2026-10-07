"""
This file houses the database logic.

OVERVIEW
--------

We're using asyncpg alongside FastAPI's dependency injection.

This file contains the logic for database manipulation in a "data access layer"
(DataAccessLayer / DAL) class, such that other areas of the code have simple
`.create_*()`, `.list_*()`, `.search_*()` calls which won't require knowledge
of how to manage connections or interact with the db directly. Connections are
managed via an asyncpg connection pool and FastAPI's dependency injection
provides a DAL instance per-request.

Each DAL instance is bound to the caller's authz resources for RLS purposes, but it does
not resolve or interpret them: `gen3_embeddings.dependencies` resolves authz at the HTTP
boundary and hands the results in. Nothing in this file talks to anything but the database,
and nothing here raises HTTP errors - see `database/errors.py`.

DETAILS
-------

What do we do in this file?

- We provide a factory for an asyncpg connection pool, but hold no pool ourselves
    - `create_pool()` builds one from the DB URL in config. The app's lifespan handler calls
      it once at startup and keeps the result on `app.state.db_pool`, mirroring how the
      Arborist client is held, so the pool's lifetime is the app's rather than the module's

- We define lightweight dataclasses for Collections and Embeddings
    - These mirror rows from the database and provide `.from_record()` helpers
      to convert from asyncpg.Record objects

- We define a DataAccessLayer class which isolates all database manipulations
    - All CRUD and search operations go through this interface instead of
      leaking raw SQL into the higher-level web app endpoint code
    - The class is composed from read, write and search mixins in `database/dal/`,
      which share one base holding the pool, the authz context and `_with_rls()`
    - DAL methods use prepared statements where appropriate as a security and
      efficiency measure
    - Each DAL instance carries a per-request `allowed_authz` list, derived
      from the current user's Arborist authz mapping

- We are deliberately free of HTTP and authz concerns
    - DAL methods raise the domain errors in `database/errors.py`; the mapping to status
      codes lives in `gen3_embeddings.error_handlers`
    - The FastAPI dependencies that build a DAL live in `gen3_embeddings.dependencies`,
      because resolving authz requires a network call to the Gen3 policy engine and this
      layer should only ever talk to the database

- We implement Row Level Security (RLS) integration for every table we own
    - Before each logical operation, `_with_rls()` sets two per-transaction
      PostgreSQL parameters:
          SELECT set_config('app.allowed_authz', $1, true);
          SELECT set_config('app.allowed_collection_names', $2, true);
      where `$1` is a text representation of the user's allowed authz resources
      and `$2` is those same grants reduced to collection names
    - Because `set_config(..., true)` uses a local/transaction-scoped setting,
      each request runs with its own authz context even when using a pooled
      connection
    - The embeddings tables (e.g., `embeddings_vector`, `embeddings_halfvec`)
      define RLS policies that consult `current_setting('app.allowed_authz', true)`
      and compare it to each row's `authz` column
    - `collections` has no `authz` column, because a collection's authz identity IS
      its name: the resource path is derived from it by convention. So its policy
      consults `current_setting('app.allowed_collection_names', true)` and compares
      it to `collection_name`

- We support multiple vector types as isolated domains
    - Collections store a `vector_type` (e.g., 'vector', 'halfvec')
    - Each vector type has its own embeddings table (e.g., `embeddings_vector`,
      `embeddings_halfvec`)
    - DAL methods route all embedding CRUD and search operations to the
      appropriate table based on the collection's `vector_type`
    - Search methods use pgvector operators and functions and expose
      a uniform interface with configurable distance metrics, min/max thresholds,
      and filters on metadata

- We carry the caller's authz context on the DAL instance, not on each call
    - `allowed_authz` and `allowed_collection_names` are constructor arguments, resolved
      once per request by `gen3_embeddings.dependencies`. This layer hands them to Postgres
      but never decides what belongs in them; empty means "nothing", which is fail-closed
    - Some methods additionally short-circuit in Python on the same set before issuing a
      query. Postgres is the enforcement point; those checks only save a round trip, or
      turn "not authorized" into a specific error rather than an empty result. The two
      cannot disagree, because both read the same field
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
