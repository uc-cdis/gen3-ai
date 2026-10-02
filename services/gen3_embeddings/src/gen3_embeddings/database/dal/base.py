"""
The state and transaction handling every data access mixin builds on.

`DataAccessLayerBase` holds the caller's authz context and is the only place that takes a
connection from the pool. The read, write and search mixins all inherit from it, so every
query they issue runs through `_with_rls` and its RLS settings.
"""

from collections.abc import Awaitable, Callable
from typing import Any

import asyncpg
from asyncpg.exceptions import InsufficientPrivilegeError

from gen3_embeddings import config
from gen3_embeddings.database.errors import RowLevelSecurityDeniedError
from gen3_embeddings.database.index_discovery import VectorIndex

# Postgres reports a policy violation and a missing table GRANT identically: both are
# SQLSTATE 42501 (asyncpg's InsufficientPrivilegeError), and neither populates `table_name`
# or any other distinguishing field. So the message text is the only discriminator, and it
# has to be, because the two mean opposite things -- one is the caller's fault, the other is
# a broken deployment.
_RLS_VIOLATION_TEXT = "violates row-level security policy"


def _rls_denial_or_reraise(exc: InsufficientPrivilegeError) -> Exception:
    """
    Classify an insufficient-privilege error as a caller denial or a deployment fault.

    Args:
        exc (InsufficientPrivilegeError): The error Postgres raised.

    Returns:
        Exception: A `RowLevelSecurityDeniedError` to raise in place of `exc`, when a policy
        rejected the row the caller asked to write.

    Raises:
        InsufficientPrivilegeError: Re-raises `exc` unchanged for anything else, most
            importantly `permission denied for table ...` -- a missing GRANT is a broken
            deployment and must stay a 500 rather than being reported as the caller's fault.
    """
    # `str(exc)` rather than `exc.message`: asyncpg declares `message` on PostgresMessage but
    # only populates it for server-raised errors, so it is None for a locally constructed one
    # and is untyped either way. For server errors the two are identical strings.
    if _RLS_VIOLATION_TEXT not in str(exc):
        raise exc

    return RowLevelSecurityDeniedError(
        "Not authorized to store a row under the requested authz. The authz value must be a "
        "resource you hold this action on."
    )


class DataAccessLayerBase:
    """
    Connection pool, authz context and the RLS transaction wrapper shared by the DAL mixins.

    Each instance carries the authz context the caller holds for the current request, in the
    two forms the database's policies key on: `allowed_authz` for the embeddings tables,
    whose rows carry an `authz` column, and `allowed_collection_names` for `collections`,
    whose rows are identified by name. Both are resolved by
    `gen3_embeddings.dependencies` and simply handed to Postgres by `_with_rls`; nothing in
    this class interprets or extends them.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        allowed_authz: list[str] | None = None,
        allowed_collection_names: set[str] | None = None,
        vector_indexes: dict[int, list[VectorIndex]] | None = None,
    ):
        """
        Bind the DAL to a connection pool and the caller's authz context.

        Args:
            pool (asyncpg.Pool): Shared connection pool.
            allowed_authz (list[str] | None): Authz resource paths the caller may act on.
                Omitted or empty means no resources are allowed, which is a valid
                fail-closed state rather than "allow everything".
            allowed_collection_names (set[str] | None): The same grants reduced to
                collection names. Omitted or empty means no collections, also fail-closed.
            vector_indexes (dict[int, list[VectorIndex]] | None): Vector indexes discovered at
                startup, keyed by collection id. Omitted means none are known and every search
                emits its unindexed form.
        """
        self.pool = pool
        # Empty means "nothing allowed", which is valid (and safe) for RLS
        self.allowed_authz = allowed_authz or []
        self.allowed_collection_names = allowed_collection_names or set()
        # Which vector indexes exist, discovered at startup. Empty means none are known, and
        # search emits its unindexed query -- slow but correct, which is the right default for
        # a collection whose index was never built.
        self.vector_indexes = vector_indexes or {}

    async def _with_rls(
        self,
        fn: Callable[..., Awaitable[Any]],
        *args: Any,
        ef_search: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run a DB operation inside a transaction carrying the caller's RLS context.

        Both settings are written for every operation regardless of which tables `fn`
        touches. Setting only the one a query "needs" would mean a later query added to the
        same operation, or a policy added to another table, silently ran with a setting that
        had never been set. Writing both makes the transaction's authz context complete.

        `is_local=true` scopes them to this transaction, so a pooled connection cannot carry
        one caller's context into another caller's query.

        The search settings ride along in the same statement, which is also the only place
        they can go: asyncpg's pool runs `RESET ALL` when a connection is released, so
        anything set in the pool's `init` survives exactly one request and then silently
        reverts. That failure is invisible -- searches quietly return at most `ef_search`
        (default 40) rows, and `statement_timeout` goes back to "no limit" -- so it has to be
        re-applied per transaction rather than per connection. Doing it here costs no extra
        round trip.

        Args:
            fn: Coroutine function taking an open connection as its first argument.
            *args: Passed through to `fn`.
            ef_search: Raise `hnsw.ef_search` to at least this for the transaction, and do not
                pass it on to `fn`. Searches that rescore a pool of binary-quantized candidates
                set it to the pool size, since an HNSW scan never returns more candidates than
                `ef_search` and would otherwise truncate the pool without saying so.
            **kwargs: Passed through to `fn`.

        Returns:
            Whatever `fn` returns.

        Raises:
            RowLevelSecurityDeniedError: If an RLS policy rejected a row `fn` tried to write.
            InsufficientPrivilegeError: If `fn` hit any other privilege error, such as a missing
                GRANT, which is a deployment fault rather than the caller's.
        """  # noqa: DOC501
        # DOC501 reads `raise _rls_denial_or_reraise(...)` as raising a type named after the
        # helper; the exceptions it actually produces are documented under Raises above.
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    SELECT
                        set_config('app.allowed_authz', $1::text[]::text, true),
                        set_config('app.allowed_collection_names', $2::text[]::text, true),
                        set_config('hnsw.ef_search', $3::text, true),
                        set_config('hnsw.iterative_scan', $4::text, true),
                        set_config('statement_timeout', $5::text, true)
                    """,
                    self.allowed_authz,
                    list(self.allowed_collection_names),
                    # An HNSW scan will not return more than `ef_search` candidates, so a
                    # binary rescore pool wider than it is silently truncated: ask for 200 and
                    # get 40. Raise it to cover the pool whenever one is in play.
                    str(max(config.HNSW_EF_SEARCH, ef_search or 0)),
                    config.HNSW_ITERATIVE_SCAN,
                    f"{config.DB_STATEMENT_TIMEOUT_MS}ms",
                )
                try:
                    return await fn(conn, *args, **kwargs)
                except InsufficientPrivilegeError as exc:
                    raise _rls_denial_or_reraise(exc) from exc
