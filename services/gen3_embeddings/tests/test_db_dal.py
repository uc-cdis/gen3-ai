"""
Unit tests for the data access layer in database/db.py.

These are the db.py tests that need no Postgres. Two kinds of thing live in that module and
both are observable without a server:

- the pure batch-shaping helpers -- `_prepare_bulk_write`, `_bulk_write_results` and
  `_rls_denial_or_reraise` -- which decide what counts as a duplicate, how written rows line
  back up with the caller's inputs, and whether a privilege error is the caller's fault or a
  broken deployment
- the DAL methods, whose SQL, bound parameters, error translation and fail-closed
  short-circuits are all visible through a fake pool

What Postgres does with the context these methods set up (which rows a policy actually hides)
is enforced by the server and covered against a real one in test_db_rls.py. What is covered
here is the half the application owns: that every operation runs inside a transaction carrying
both RLS settings, that parameters are bound in the order the SQL expects, that server errors
become the right domain errors, and that the paths which cannot possibly need the database --
an unauthorized collection name, an empty batch, a malformed vector -- never take a connection
out of the pool.
"""

import re
from uuid import UUID, uuid4

import asyncpg
import pytest
from asyncpg.exceptions import InsufficientPrivilegeError, UniqueViolationError
from pgvector.asyncpg import register_vector

from gen3_embeddings import config
from gen3_embeddings.database import db as db_module
from gen3_embeddings.database import hashing
from gen3_embeddings.database.db import (
    DataAccessLayer,
    _bulk_write_results,
    _prepare_bulk_write,
    _rls_denial_or_reraise,
)
from gen3_embeddings.database.errors import (
    CollectionAlreadyExistsError,
    CollectionCreateFailedError,
    CollectionNameNotAllowedError,
    DuplicateEmbeddingError,
    EmbeddingDimensionMismatchError,
    EmbeddingNotRepresentableError,
    EmbeddingsAlreadyExistError,
    EmbeddingWriteInconsistencyError,
    InvalidCollectionNameError,
    MetadataLengthMismatchError,
    RowLevelSecurityDeniedError,
)
from gen3_embeddings.database.models import Collection
from gen3_embeddings.models.schemas import DistanceMetric, VectorType

DOCS_AUTHZ = "/vectorstore/collections/docs"

# what Postgres says when a policy rejects a row: the caller asked for something it does not
# hold, which is a 403
RLS_VIOLATION_MESSAGE = 'new row violates row-level security policy for table "embeddings_vector"'
# what Postgres says when the role is missing a GRANT: nothing the caller did, so it must stay
# a 500 rather than being reported as their fault
MISSING_GRANT_MESSAGE = 'permission denied for table "embeddings_vector"'


# ---------------------------------------------------------------------------
# fakes
#
# A pool/connection pair faithful to the parts of asyncpg the DAL uses: `pool.acquire()` as an
# async context manager, `conn.transaction()` likewise, `conn.execute()` returning a command
# tag, and `conn.prepare()` returning something with `fetch`/`fetchrow`. Rows are plain dicts,
# which support both `row["col"]` and `dict(row)` -- the only two things `from_record` does
# with an asyncpg.Record.
# ---------------------------------------------------------------------------


class FakeTransaction:
    """Async context manager standing in for `Connection.transaction()`."""

    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        self._conn.transactions += 1
        self._conn.in_transaction = True
        return self

    async def __aexit__(self, *exc_info):
        self._conn.in_transaction = False
        return False


class FakeStatement:
    """Prepared statement that records its parameters and returns a queued result."""

    def __init__(self, conn, sql):
        self._conn = conn
        self.sql = sql

    async def fetch(self, *params):
        return self._conn._run(self.sql, params)

    async def fetchrow(self, *params):
        rows = self._conn._run(self.sql, params)
        return rows[0] if rows else None


class FakeConnection:
    """
    Connection that records every query and hands back queued results.

    Args:
        results: One entry per query, consumed in order. A list of rows is returned; an
            Exception instance is raised, which is how server errors (unique violations,
            privilege denials) are injected. Queries past the end of the queue see no rows.
        execute_result: Command tag returned by `execute`, e.g. "DELETE 1".
    """

    def __init__(self, results=None, execute_result="DELETE 0"):
        self._results = list(results) if results is not None else []
        self.execute_result = execute_result
        # (allowed_authz, allowed_collection_names) exactly as bound by _with_rls
        self.rls_context = None
        # the set_config statement itself, so a setting can be checked to actually read its
        # parameter rather than just to have been passed one
        self.rls_sql = None
        # (sql, params) per query, in order, excluding the RLS set_config
        self.queries = []
        self.transactions = 0
        self.in_transaction = False

    def transaction(self):
        return FakeTransaction(self)

    async def execute(self, sql, *params):
        if "set_config" in sql:
            assert self.in_transaction, "the RLS settings must be set inside the transaction"
            self.rls_context = params
            self.rls_sql = sql
            return "SELECT 1"
        self._run(sql, params)
        return self.execute_result

    async def prepare(self, sql):
        assert self.in_transaction, "a statement was prepared outside the RLS transaction"
        return FakeStatement(self, sql)

    def _run(self, sql, params):
        assert self.in_transaction, "a query ran outside the RLS transaction"
        assert self.rls_context is not None, "a query ran before the RLS context was set"
        self.queries.append((sql, list(params)))
        if not self._results:
            return []
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    @property
    def sql(self):
        """The SQL of the single non-RLS query this connection ran."""
        assert len(self.queries) == 1, f"expected exactly one query, got {len(self.queries)}"
        return self.queries[0][0]

    @property
    def params(self):
        """The bound parameters of the single non-RLS query this connection ran."""
        assert len(self.queries) == 1, f"expected exactly one query, got {len(self.queries)}"
        return self.queries[0][1]


class FakeAcquire:
    """Async context manager standing in for `Pool.acquire()`."""

    def __init__(self, pool):
        self._pool = pool

    async def __aenter__(self):
        return self._pool.conn

    async def __aexit__(self, *exc_info):
        return False


class FakePool:
    """Pool that hands out one connection and counts how often it was asked for it."""

    def __init__(self, conn):
        self.conn = conn
        self.acquired = 0

    def acquire(self):
        self.acquired += 1
        return FakeAcquire(self)


def make_dal(
    results=None,
    execute_result="DELETE 0",
    allowed_authz=(DOCS_AUTHZ,),
    allowed_collection_names=("docs",),
):
    """Build a DAL over a fake pool, returning (dal, connection, pool)."""
    conn = FakeConnection(results=results, execute_result=execute_result)
    pool = FakePool(conn)
    dal = DataAccessLayer(
        pool,
        allowed_authz=list(allowed_authz),
        allowed_collection_names=set(allowed_collection_names),
    )
    return dal, conn, pool


def collection_row(**overrides):
    """A row of `collections`, as asyncpg would return it."""
    row = {
        "id": 1,
        "collection_name": "docs",
        "description": "a description",
        "ai_model_name": None,
        "dimensions": 3,
        "vector_type": "vector",
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def make_collection(**overrides) -> Collection:
    """A Collection built the same way the DAL builds one, so `vector_type` is the enum."""
    return Collection.from_record(collection_row(**overrides))


def embedding_row(**overrides):
    """A row of one of the `embeddings_*` tables, as asyncpg would return it."""
    row = {
        "collection_id": 1,
        "embedding_id": uuid4(),
        "embedding": [1.0, 2.0, 3.0],
        "authz": DOCS_AUTHZ,
        "metadata": {},
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def written_row(vector, metadata, dimensions=3, vector_type=VectorType.vector, **overrides):
    """
    A row as a bulk write's RETURNING clause emits it, carrying the hashes of its content.

    `_bulk_write_results` matches on those hashes, so they have to be the real ones for the
    given content rather than arbitrary uuids.
    """
    return embedding_row(
        embedding=vector,
        metadata=metadata,
        embedding_hash_v2=hashing.hash_vector(vector, vector_type, dimensions),
        metadata_hash_v2=hashing.hash_metadata(metadata),
        **overrides,
    )


def normalized(sql: str) -> str:
    """Collapse whitespace so SQL can be matched without depending on its formatting."""
    return re.sub(r"\s+", " ", sql).strip()


# ---------------------------------------------------------------------------
# create_pool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_pool_registers_the_pgvector_codec(monkeypatch):
    """
    The pool is built with `register_vector` as its per-connection init.

    Without it asyncpg treats a `vector` column as text, so every read and write round-trips
    the whole vector as a decimal string. That is a silent, purely-performance regression:
    nothing fails, so only this assertion would notice the init going missing.
    """
    captured = {}

    async def fake_create_pool(dsn, **kwargs):
        captured["dsn"] = dsn
        captured.update(kwargs)
        return "the-pool"

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(config, "DB_CONNECTION_STRING", "postgresql://localhost:5432/test_db")
    monkeypatch.setattr(config, "PGPOOL_MIN_SIZE", 2)
    monkeypatch.setattr(config, "PGPOOL_MAX_SIZE", 9)

    assert await db_module.create_pool() == "the-pool"

    assert captured["dsn"] == "postgresql://localhost:5432/test_db"
    assert captured["min_size"] == 2
    assert captured["max_size"] == 9
    assert captured["init"] is register_vector


# ---------------------------------------------------------------------------
# _rls_denial_or_reraise
# ---------------------------------------------------------------------------


def test_policy_violation_becomes_an_rls_denial():
    """A row a policy rejected is the caller asking for authz it does not hold."""
    exc = InsufficientPrivilegeError(RLS_VIOLATION_MESSAGE)

    result = _rls_denial_or_reraise(exc)

    assert isinstance(result, RowLevelSecurityDeniedError)
    # returned rather than raised, so the caller decides how to chain it
    assert isinstance(result, Exception)


def test_missing_table_grant_is_re_raised_unchanged():
    """
    A missing GRANT is a broken deployment, not a caller error.

    Postgres reports it with the same SQLSTATE as a policy violation and populates no field
    that distinguishes the two, so the message text is the only discriminator there is. Get
    it wrong in this direction and an operator's misconfiguration is reported to every user
    as "you are not authorized", with nothing in the logs saying otherwise.
    """
    exc = InsufficientPrivilegeError(MISSING_GRANT_MESSAGE)

    with pytest.raises(InsufficientPrivilegeError) as raised:
        _rls_denial_or_reraise(exc)

    assert raised.value is exc


def test_classification_reads_the_message_via_str():
    """
    The text is read with `str(exc)`, which is populated for a locally built error too.

    asyncpg only fills `exc.message` in for server-raised errors, so keying on that attribute
    would classify every locally constructed error -- including the ones in these tests -- as
    a deployment fault.
    """
    exc = InsufficientPrivilegeError(RLS_VIOLATION_MESSAGE)
    assert exc.message is None
    assert isinstance(_rls_denial_or_reraise(exc), RowLevelSecurityDeniedError)


# ---------------------------------------------------------------------------
# _prepare_bulk_write
# ---------------------------------------------------------------------------


def test_absent_metadata_becomes_an_empty_object_per_row():
    """None means "no metadata for any row", which is stored as {} rather than NULL."""
    batch = _prepare_bulk_write(make_collection(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], None)

    assert batch.metadata_json == ["{}", "{}"]
    assert batch.row_count == 2


def test_metadata_of_the_wrong_length_is_rejected():
    """Metadata is positional, so a shorter list would silently misattribute it."""
    with pytest.raises(MetadataLengthMismatchError):
        _prepare_bulk_write(make_collection(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [{"a": 1}])


def test_identical_rows_collapse_and_every_input_keeps_a_slot():
    """
    A repeat inside one request is one row, but the caller still gets a result per input.

    Responses are indexed by position, so `original_to_unique` has to stay as long as the
    input even though the batch is shorter.
    """
    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]
    batch = _prepare_bulk_write(make_collection(), vectors, None)

    assert batch.row_count == 2
    assert batch.has_duplicates is True
    assert batch.original_to_unique == [0, 1, 0]
    # only the surviving rows are bound, row-major
    assert batch.flat_vectors == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_a_batch_without_repeats_maps_one_to_one():
    """Nothing to collapse means the identity mapping and the cheap result path."""
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    batch = _prepare_bulk_write(make_collection(), vectors, None)

    assert batch.has_duplicates is False
    assert batch.original_to_unique == [0, 1, 2]
    assert batch.row_count == 3


def test_metadata_key_order_does_not_defeat_deduplication():
    """
    Dedup is on canonical JSON, so key order cannot smuggle a second copy of a row in.

    The hashed text is the same text that gets stored, so this row's hash also matches what
    the single-row update path would compute for the same metadata.
    """
    vectors = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    batch = _prepare_bulk_write(make_collection(), vectors, [{"a": 1, "b": 2}, {"b": 2, "a": 1}])

    assert batch.row_count == 1
    assert batch.metadata_json == ['{"a":1,"b":2}']


def test_same_vector_with_different_metadata_is_two_rows():
    """The dedup key is the pair, so metadata alone is enough to keep rows apart."""
    vectors = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    batch = _prepare_bulk_write(make_collection(), vectors, [{"a": 1}, {"a": 2}])

    assert batch.row_count == 2
    assert batch.embedding_hashes[0] == batch.embedding_hashes[1]
    assert batch.metadata_hashes[0] != batch.metadata_hashes[1]


def test_vectors_differing_below_storage_precision_are_one_row_on_halfvec():
    """
    Hashes are taken at storage precision, so 1.0 and 1.0001 are the same halfvec row.

    float16 spacing near 1.0 is ~0.001. These two inputs land on byte-identical stored
    vectors, so treating them as distinct would write two rows holding the same vector --
    which is exactly what the old hash-the-caller's-JSON-text approach did.
    """
    vectors = [[1.0, 2.0, 3.0], [1.0001, 2.0, 3.0]]
    batch = _prepare_bulk_write(make_collection(vector_type="halfvec"), vectors, None)

    assert batch.row_count == 1
    assert batch.original_to_unique == [0, 0]


def test_the_same_vectors_stay_distinct_on_a_vector_collection():
    """
    The same pair does *not* collapse at float32, which is what makes the case above precision
    rather than rounding-everything.
    """
    vectors = [[1.0, 2.0, 3.0], [1.0001, 2.0, 3.0]]
    batch = _prepare_bulk_write(make_collection(), vectors, None)

    assert batch.row_count == 2


def test_a_wrong_length_vector_is_rejected_before_any_binding():
    """
    A vector of the wrong length cannot be bound at all.

    The INSERT binds the batch as one flat float4[] and slices row `i` by the collection's
    dimensionality, so a short vector would shift every following row's slice rather than
    fail: row 2 would be stored holding the tail of row 1.
    """
    with pytest.raises(EmbeddingDimensionMismatchError):
        _prepare_bulk_write(make_collection(), [[1.0, 2.0, 3.0], [1.0, 2.0]], None)


def test_a_value_the_column_cannot_hold_is_rejected():
    """float16 stops at ~65504, and overflowing it locally beats a Postgres cast error."""
    with pytest.raises(EmbeddingNotRepresentableError):
        _prepare_bulk_write(make_collection(vector_type="halfvec"), [[1e6, 2.0, 3.0]], None)


def test_an_empty_batch_produces_no_rows():
    """Nothing to write is a valid batch, which the write methods turn into a no-op."""
    batch = _prepare_bulk_write(make_collection(), [], None)

    assert batch.row_count == 0
    assert batch.flat_vectors == []
    assert batch.original_to_unique == []


def test_row_keys_pair_the_hashes_positionally():
    """
    `row_keys` is the identity each RETURNING row is matched back on.

    It must pair each row's own two hashes; crossing them would still be unique across the
    batch, so the mismatch would only show up as results attributed to the wrong input.
    """
    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    batch = _prepare_bulk_write(make_collection(), vectors, [{"a": 1}, {"a": 2}])

    assert batch.row_keys == [
        (batch.embedding_hashes[0], batch.metadata_hashes[0]),
        (batch.embedding_hashes[1], batch.metadata_hashes[1]),
    ]
    assert len(set(batch.row_keys)) == batch.row_count


def test_hashes_match_the_single_row_helpers():
    """
    A batch hashes a row to the same value `hash_vector`/`hash_metadata` do.

    The backfill script and the single-row update path use those helpers, so a batch that
    disagreed would write rows the unique index could not recognize as duplicates.
    """
    batch = _prepare_bulk_write(make_collection(), [[0.1, 0.2, 0.3]], [{"b": 1, "a": 2}])

    assert batch.embedding_hashes == [hashing.hash_vector([0.1, 0.2, 0.3], VectorType.vector, 3)]
    assert batch.metadata_hashes == [hashing.hash_metadata({"b": 1, "a": 2})]


# ---------------------------------------------------------------------------
# _bulk_write_results
# ---------------------------------------------------------------------------


def test_returned_rows_are_matched_by_content_not_by_position():
    """
    Rows are lined up by their hash pair, so the order Postgres emits them does not matter.

    RETURNING order is not something Postgres promises. Matching by position happens to work
    most of the time, which is what makes it a bad thing to rely on: it would break as a
    silent mis-attribution of results, under a plan change nobody made.
    """
    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    metadata_list = [{"i": 0}, {"i": 1}]
    batch = _prepare_bulk_write(make_collection(), vectors, metadata_list)

    rows = [written_row(vector, metadata) for vector, metadata in zip(vectors, metadata_list)]
    results = _bulk_write_results(list(reversed(rows)), batch)

    assert [r.metadata for r in results] == [{"i": 0}, {"i": 1}]


def test_deduplicated_inputs_all_receive_the_row_they_collapsed_onto():
    """Each input gets a result, and the repeats share the single row that was written."""
    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]
    batch = _prepare_bulk_write(make_collection(), vectors, None)

    rows = [written_row([1.0, 2.0, 3.0], {}), written_row([4.0, 5.0, 6.0], {})]
    results = _bulk_write_results(rows, batch)

    assert len(results) == 3
    assert results[0] is results[2]
    assert results[0] is not results[1]


def test_a_short_returning_set_is_an_inconsistency():
    """Fewer rows back than asked for means a row was dropped, which is never the caller's fault."""
    batch = _prepare_bulk_write(make_collection(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], None)

    with pytest.raises(EmbeddingWriteInconsistencyError):
        _bulk_write_results([written_row([1.0, 2.0, 3.0], {})], batch)


def test_a_row_that_matches_no_input_is_an_inconsistency():
    """
    The right *number* of rows is not enough; each one has to be an input's row.

    Reporting an unrelated row's id back as the caller's would hand them a pointer to a row
    they never wrote, so this fails rather than returning whatever came back.
    """
    batch = _prepare_bulk_write(make_collection(), [[1.0, 2.0, 3.0]], None)

    with pytest.raises(EmbeddingWriteInconsistencyError):
        _bulk_write_results([written_row([9.0, 9.0, 9.0], {})], batch)


# ---------------------------------------------------------------------------
# construction and _with_rls
# ---------------------------------------------------------------------------


def test_an_unscoped_dal_is_fail_closed():
    """
    Omitted authz means "nothing allowed", not "everything".

    This is the default a caller gets by forgetting an argument, so it has to be the safe one.
    """
    dal = DataAccessLayer(FakePool(FakeConnection()))

    assert dal.allowed_authz == []
    assert dal.allowed_collection_names == set()


@pytest.mark.asyncio
async def test_every_operation_carries_both_rls_settings():
    """
    Both settings are written for every operation, from the instance's own fields.

    `collections` keys on `app.allowed_collection_names` and the embeddings tables key on
    `app.allowed_authz`. Setting only the one the current query "needs" means the next query
    added to the operation runs under a setting that was never set -- which denies everything
    and reads as data loss rather than as a bug.
    """
    dal, conn, _ = make_dal(
        results=[[collection_row()]],
        allowed_authz=[DOCS_AUTHZ, "/programs/foo"],
        allowed_collection_names=["docs"],
    )

    await dal.get_collection_by_name("docs")

    authz, collection_names = conn.rls_context
    assert authz == [DOCS_AUTHZ, "/programs/foo"]
    # the set is bound as a list, because asyncpg has no encoder for a set
    assert collection_names == ["docs"]
    assert isinstance(collection_names, list)
    assert conn.transactions == 1

    rls_sql = normalized(conn.rls_sql)
    assert "set_config('app.allowed_authz', $1::text[]::text, true)" in rls_sql
    assert "set_config('app.allowed_collection_names', $2::text[]::text, true)" in rls_sql


@pytest.mark.asyncio
async def test_the_rls_context_is_set_before_the_query_runs():
    """
    Settings land inside the transaction and ahead of the work.

    `set_config(..., is_local => true)` is transaction-scoped, which is the whole reason a
    pooled connection cannot leak one caller's context into another's query -- but only if
    the transaction is already open. The fake connection asserts both orderings, so a query
    prepared or run before the settings fails here.
    """
    dal, conn, pool = make_dal(results=[[collection_row()]])

    await dal.get_collection_by_name("docs")

    assert pool.acquired == 1
    assert conn.rls_context is not None
    assert len(conn.queries) == 1


@pytest.mark.asyncio
async def test_with_rls_passes_arguments_through_and_returns_the_result():
    """`_with_rls` is a wrapper: it adds the context and otherwise stays out of the way."""
    dal, conn, _ = make_dal()
    seen = {}

    async def op(connection, value, *, keyword):
        seen["connection"] = connection
        seen["value"] = value
        seen["keyword"] = keyword
        return "returned"

    assert await dal._with_rls(op, 7, keyword="k") == "returned"
    assert seen == {"connection": conn, "value": 7, "keyword": "k"}


@pytest.mark.asyncio
async def test_a_policy_denial_inside_an_operation_becomes_a_domain_error():
    """A row rejected by a policy surfaces as the DAL's own error, not asyncpg's."""
    dal, _, _ = make_dal(results=[InsufficientPrivilegeError(RLS_VIOLATION_MESSAGE)])

    with pytest.raises(RowLevelSecurityDeniedError):
        await dal.create_embeddings_bulk(make_collection(), [[1.0, 2.0, 3.0]], DOCS_AUTHZ, None)


@pytest.mark.asyncio
async def test_a_missing_grant_inside_an_operation_stays_a_server_error():
    """The deployment fault propagates untranslated, so it cannot be answered as a 403."""
    dal, _, _ = make_dal(results=[InsufficientPrivilegeError(MISSING_GRANT_MESSAGE)])

    with pytest.raises(InsufficientPrivilegeError):
        await dal.create_embeddings_bulk(make_collection(), [[1.0, 2.0, 3.0]], DOCS_AUTHZ, None)


# ---------------------------------------------------------------------------
# collections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creating_a_collection_the_caller_may_not_name_never_reaches_the_database():
    """
    An unauthorized name is refused in Python, without a connection.

    The table's WITH CHECK would refuse it too, so this changes the error rather than the
    outcome: a specific "not authorized to use this name" instead of a bare privilege error,
    and no round trip to find that out.
    """
    dal, _, pool = make_dal(allowed_collection_names=["docs"])

    with pytest.raises(CollectionNameNotAllowedError):
        await dal.create_collection("other", "d", 3)

    assert pool.acquired == 0


@pytest.mark.asyncio
async def test_creating_a_collection_binds_its_columns_in_order():
    """The insert's parameters line up with its column list, vector type included."""
    dal, conn, _ = make_dal(results=[[collection_row(vector_type="halfvec")]])

    collection = await dal.create_collection("docs", "a description", 3, "some-model", VectorType.halfvec)

    assert conn.params == ["docs", "a description", "some-model", 3, "halfvec"]
    assert isinstance(collection, Collection)
    assert collection.vector_type == VectorType.halfvec


@pytest.mark.asyncio
async def test_a_taken_collection_name_is_reported_as_already_existing():
    """The unique violation on `collection_name` is the only way to learn the name is taken."""
    dal, _, _ = make_dal(results=[UniqueViolationError("duplicate key value violates unique constraint")])

    with pytest.raises(CollectionAlreadyExistsError):
        await dal.create_collection("docs", None, 3)


@pytest.mark.asyncio
async def test_an_insert_that_returns_no_row_is_a_failure_not_a_none():
    """
    RETURNING with no row means the write did not happen, which must not read as success.

    Returning None here would have the route answer 200 with an empty body for a collection
    that does not exist.
    """
    dal, _, _ = make_dal(results=[[]])

    with pytest.raises(CollectionCreateFailedError):
        await dal.create_collection("docs", None, 3)


@pytest.mark.asyncio
async def test_looking_up_a_collection_normalizes_the_name_first():
    """
    The name is normalized before it is bound, because this runs on raw path parameters.

    Without it, "DOCS" would miss the row that "docs" created -- and, worse, would miss the
    RLS setting too, since the policy compares against the same normalized names.
    """
    dal, conn, _ = make_dal(results=[[collection_row()]])

    await dal.get_collection_by_name("  DOCS  ")

    assert conn.params == ["docs"]


@pytest.mark.asyncio
async def test_a_malformed_collection_name_is_rejected_without_a_connection():
    """A name that cannot be valid is the caller's error, not something to ask Postgres about."""
    dal, _, pool = make_dal()

    with pytest.raises(InvalidCollectionNameError):
        await dal.get_collection_by_name("not a valid name!")

    assert pool.acquired == 0


@pytest.mark.asyncio
async def test_a_hidden_or_absent_collection_is_indistinguishable():
    """
    No row means None, whether it does not exist or RLS hid it.

    Keeping the two cases identical is deliberate: a caller must not be able to probe for the
    names of collections it cannot see.
    """
    dal, _, _ = make_dal(results=[[]])

    assert await dal.get_collection_by_name("docs") is None


@pytest.mark.asyncio
async def test_looking_up_a_collection_by_id_binds_the_key():
    """The by-id lookup is the same shape, keyed on the primary key."""
    dal, conn, _ = make_dal(results=[[collection_row(id=42)]])

    collection = await dal.get_collection_by_id(42)

    assert conn.params == [42]
    assert collection.id == 42


@pytest.mark.asyncio
async def test_updating_a_collection_with_nothing_to_set_only_reads():
    """
    `description=None` updates nothing, so the row is read rather than written.

    Issuing `SET updated_at = NOW()` with no other assignment would make a no-op request look
    like a modification to anything watching that column.
    """
    dal, conn, _ = make_dal(results=[[collection_row()]])

    collection = await dal.update_collection("docs", None)

    assert "UPDATE" not in normalized(conn.sql)
    assert conn.params == ["docs"]
    assert collection.collection_name == "docs"


@pytest.mark.asyncio
async def test_updating_a_description_touches_the_timestamp_and_keys_on_the_name():
    """
    The name is a predicate here, never an assignment.

    The collections policy keys on `collection_name`, so assigning it would move the row to a
    different authz resource -- a rename that grants itself. There is no API path to do that,
    and this is the assertion that keeps it that way.
    """
    dal, conn, _ = make_dal(results=[[collection_row(description="new")]])

    await dal.update_collection("docs", "new")

    sql = normalized(conn.sql)
    assert "SET description = $2::text, updated_at = NOW()" in sql
    assert "WHERE collection_name = $1::text" in sql
    assert "SET collection_name" not in sql
    assert conn.params == ["docs", "new"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tag, expected", [("DELETE 1", True), ("DELETE 0", False)])
async def test_deleting_a_collection_reports_whether_a_row_went(tag, expected):
    """
    The command tag's count decides, not its verb.

    "DELETE 0" is what both a nonexistent collection and one hidden by RLS produce, and both
    have to come back False so the route can answer 404 instead of a cheerful 200.
    """
    dal, _, _ = make_dal(execute_result=tag)

    assert await dal.delete_collection("docs") is expected


@pytest.mark.asyncio
async def test_listing_collections_with_no_grants_skips_the_round_trip():
    """
    A caller allowed nothing gets an empty list without a query.

    RLS would return nothing anyway; this only saves the connection. That the two agree is
    structural, since both read the same field.
    """
    dal, _, pool = make_dal(allowed_authz=[], allowed_collection_names=[])

    assert await dal.list_collections() == []
    assert pool.acquired == 0


@pytest.mark.asyncio
async def test_listing_collections_binds_offset_and_limit_to_their_own_placeholders():
    """
    The SQL reads `LIMIT $2 OFFSET $1` while the call passes (offset, limit).

    The placeholders are deliberately out of source order, so this pins the pairing: swapping
    them silently turns "page 5 of 10" into "the first 5 rows, skipping 10".
    """
    dal, conn, _ = make_dal(results=[[collection_row(id=1), collection_row(id=2, collection_name="images")]])

    collections = await dal.list_collections(offset=10, limit=5)

    assert conn.params == [10, 5]
    sql = normalized(conn.sql)
    assert "LIMIT $2::int OFFSET $1::int" in sql
    # no name predicate: duplicating the policy in the WHERE clause would be a second,
    # invisible authorization rule that could disagree with it
    assert "collection_name = ANY" not in sql
    assert [c.id for c in collections] == [1, 2]


@pytest.mark.asyncio
async def test_bulk_collection_lookup_with_no_grants_skips_the_round_trip():
    """Same fail-closed short-circuit as the paged list."""
    dal, _, pool = make_dal(allowed_authz=[], allowed_collection_names=[])

    assert await dal.get_collection_by_id_bulk([1, 2, 3]) == []
    assert pool.acquired == 0


@pytest.mark.asyncio
async def test_bulk_collection_lookup_binds_the_ids_as_one_array():
    """Ids go as an array parameter, so the statement is the same shape for any batch size."""
    dal, conn, _ = make_dal(results=[[collection_row(id=1)]])

    collections = await dal.get_collection_by_id_bulk([1, 2, 3])

    assert conn.params == [[1, 2, 3]]
    # unauthorized ids simply return no row, so a short result is normal
    assert [c.id for c in collections] == [1]


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_writing_an_empty_batch_never_reaches_the_database():
    """Nothing to write is a no-op, not an INSERT with an empty array."""
    dal, _, pool = make_dal()

    assert await dal.create_embeddings_bulk(make_collection(), [], DOCS_AUTHZ, None) == []
    assert await dal.upsert_embeddings_bulk(make_collection(), [], DOCS_AUTHZ, None) == []
    assert pool.acquired == 0


@pytest.mark.asyncio
async def test_a_bulk_write_binds_the_batch_in_the_order_the_insert_slices_it():
    """
    The seven parameters are what let one statement write a whole batch in binary.

    $2 is the flat float4[] and $3 the stride the INSERT slices it by, so the vectors never
    become text on the way to the server; $5-$7 are the parallel per-row arrays. Everything
    here is positional, and a wrong stride or a swapped pair of arrays would store rows made
    of other rows' halves rather than fail.
    """
    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    metadata_list = [{"i": 0}, {"i": 1}]
    rows = [written_row(vector, metadata) for vector, metadata in zip(vectors, metadata_list)]
    dal, conn, _ = make_dal(results=[rows])

    results = await dal.create_embeddings_bulk(make_collection(id=7), vectors, DOCS_AUTHZ, metadata_list)

    collection_id, flat_vectors, dimensions, authz, metadata_json, embedding_hashes, metadata_hashes = conn.params
    assert collection_id == 7
    assert flat_vectors == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert dimensions == 3
    assert authz == DOCS_AUTHZ
    assert metadata_json == ['{"i":0}', '{"i":1}']
    assert all(isinstance(h, UUID) for h in embedding_hashes + metadata_hashes)
    assert [r.metadata for r in results] == metadata_list


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vector_type, table",
    [(VectorType.vector, "embeddings_vector"), (VectorType.halfvec, "embeddings_halfvec")],
)
async def test_a_bulk_write_targets_the_table_for_the_collections_vector_type(vector_type, table):
    """
    Each vector type is its own table, chosen from the collection rather than the request.

    Writing a halfvec collection's rows into `embeddings_vector` would store them at the
    wrong precision, where they would be invisible to every read of the collection.
    """
    collection = make_collection(vector_type=vector_type.value)
    rows = [written_row([1.0, 2.0, 3.0], {}, vector_type=vector_type)]
    dal, conn, _ = make_dal(results=[rows])

    await dal.create_embeddings_bulk(collection, [[1.0, 2.0, 3.0]], DOCS_AUTHZ, None)

    sql = normalized(conn.sql)
    assert f"INSERT INTO {table} (" in sql
    assert f"::{vector_type.value}" in sql


@pytest.mark.asyncio
async def test_a_bulk_write_dual_writes_the_legacy_hash_columns():
    """
    The pre-sha256 md5 columns get the new value too.

    They are still NOT NULL and still carry a unique constraint until the contract migration
    drops them, so a write that skipped them would fail outright. Writing the same value means
    anything that would collide on the legacy index collides on the v2 one as well.
    """
    dal, conn, _ = make_dal(results=[[written_row([1.0, 2.0, 3.0], {})]])

    await dal.create_embeddings_bulk(make_collection(), [[1.0, 2.0, 3.0]], DOCS_AUTHZ, None)

    sql = normalized(conn.sql)
    assert "embedding_hash, metadata_hash, embedding_hash_v2, metadata_hash_v2" in sql
    assert sql.count("raw.embedding_hash,") + sql.count("raw.embedding_hash ") >= 2


@pytest.mark.asyncio
async def test_a_bulk_create_of_existing_content_is_reported_as_already_existing():
    """
    A create does not upsert: the conflict is the answer, and no row was written.

    The message says so, because a partially-applied batch would be the alternative reading
    and callers act on it (they retry with PUT).
    """
    dal, _, _ = make_dal(results=[UniqueViolationError("duplicate key value violates unique constraint")])

    with pytest.raises(EmbeddingsAlreadyExistError):
        await dal.create_embeddings_bulk(make_collection(), [[1.0, 2.0, 3.0]], DOCS_AUTHZ, None)


@pytest.mark.asyncio
async def test_an_upsert_conflicts_on_the_v2_index_and_only_touches_the_timestamp():
    """
    An upsert of existing content updates that row rather than failing.

    The conflict target is the v2 index, and the DO UPDATE deliberately assigns nothing but
    `updated_at`: the conflicting columns are the content itself, so there is nothing else
    about the row that could have changed.
    """
    rows = [written_row([1.0, 2.0, 3.0], {"a": 1})]
    dal, conn, _ = make_dal(results=[rows])

    results = await dal.upsert_embeddings_bulk(make_collection(), [[1.0, 2.0, 3.0]], DOCS_AUTHZ, [{"a": 1}])

    sql = normalized(conn.sql)
    assert "ON CONFLICT (collection_id, embedding_hash_v2, metadata_hash_v2, authz)" in sql
    assert "DO UPDATE SET updated_at = NOW()" in sql
    assert len(results) == 1


@pytest.mark.asyncio
async def test_an_upsert_expands_deduplicated_rows_back_to_the_caller():
    """Repeats inside an upsert get a result each, the same way a create does."""
    vectors = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    dal, conn, _ = make_dal(results=[[written_row([1.0, 2.0, 3.0], {})]])

    results = await dal.upsert_embeddings_bulk(make_collection(), vectors, DOCS_AUTHZ, None)

    # one row written, two results
    assert conn.params[1] == [1.0, 2.0, 3.0]
    assert len(results) == 2
    assert results[0] is results[1]


@pytest.mark.asyncio
async def test_reading_one_embedding_keys_on_collection_and_id():
    """
    The collection is part of the predicate, not just context.

    An embedding id is a uuid, but it is only meaningful within its collection: matching on
    the id alone would read a row out of a collection the caller did not name.
    """
    row = embedding_row(embedding_id=uuid4())
    dal, conn, _ = make_dal(results=[[row]])

    embedding = await dal.get_embedding_by_collection_and_id(make_collection(id=7), row["embedding_id"])

    assert conn.params == [7, row["embedding_id"]]
    assert "embeddings_vector" in conn.sql
    assert embedding.embedding_id == row["embedding_id"]


@pytest.mark.asyncio
async def test_a_missing_embedding_is_none():
    """No row means None here too, for the same reason it does for collections."""
    dal, _, _ = make_dal(results=[[]])

    assert await dal.get_embedding_by_collection_and_id(make_collection(), uuid4()) is None


@pytest.mark.asyncio
async def test_an_update_with_nothing_to_change_only_reads():
    """All-None means "read it back", not an UPDATE that bumps `updated_at`."""
    dal, conn, _ = make_dal(results=[[embedding_row()]])

    embedding = await dal.update_embedding(make_collection(), uuid4(), None, None, None)

    assert "UPDATE" not in normalized(conn.sql)
    assert embedding is not None


@pytest.mark.asyncio
async def test_updating_a_vector_rewrites_both_hash_columns_from_one_parameter():
    """
    The vector's new hash goes to the v2 column and the legacy one, bound once.

    Sharing the placeholder is what guarantees the two cannot drift: the legacy column's
    unique constraint is still live, so a stale value there would reject the next write of
    genuinely new content.
    """
    embedding_id = uuid4()
    dal, conn, _ = make_dal(results=[[embedding_row()]])

    await dal.update_embedding(make_collection(id=7), embedding_id, [4.0, 5.0, 6.0], None, None)

    sql = normalized(conn.sql)
    assert "embedding = $3::vector" in sql
    assert "embedding_hash = $4::uuid" in sql
    assert "embedding_hash_v2 = $4::uuid" in sql
    assert "updated_at = NOW()" in sql
    assert conn.params == [
        7,
        embedding_id,
        [4.0, 5.0, 6.0],
        hashing.hash_vector([4.0, 5.0, 6.0], VectorType.vector, 3),
    ]


@pytest.mark.asyncio
async def test_updating_metadata_stores_the_same_text_that_was_hashed():
    """
    Canonical JSON is both stored and hashed, so this row matches a bulk write of the same
    metadata.

    These two paths used to canonicalize differently -- one hashed `json.dumps` output, the
    other Postgres's jsonb rendering -- which meant they could disagree about what a duplicate
    was, and a PUT could create the row a POST would have rejected.
    """
    dal, conn, _ = make_dal(results=[[embedding_row()]])

    await dal.update_embedding(make_collection(), uuid4(), None, {"b": 2, "a": 1}, None)

    sql = normalized(conn.sql)
    assert "metadata = $3::jsonb" in sql
    assert "metadata_hash = $4::uuid" in sql
    assert "metadata_hash_v2 = $4::uuid" in sql
    assert conn.params[2] == '{"a":1,"b":2}'
    assert conn.params[3] == hashing.hash_metadata({"a": 1, "b": 2})


@pytest.mark.asyncio
async def test_updating_authz_alone_leaves_the_content_columns_untouched():
    """Moving a row to another resource does not rewrite (or rehash) what it holds."""
    dal, conn, _ = make_dal(results=[[embedding_row()]])

    await dal.update_embedding(make_collection(), uuid4(), None, None, "/programs/foo")

    sql = normalized(conn.sql)
    assert "authz = $3::text" in sql
    assert "embedding =" not in sql
    assert "metadata =" not in sql
    assert conn.params[2] == "/programs/foo"


@pytest.mark.asyncio
async def test_an_update_that_would_duplicate_another_row_is_reported_as_such():
    """
    The unique constraint is the only definition of duplicate, so its violation is the signal.

    (collection_id, embedding_hash_v2, metadata_hash_v2, authz) has to stay unique; an update
    can collide with a row that was already there.
    """
    dal, _, _ = make_dal(results=[UniqueViolationError("duplicate key value violates unique constraint")])

    with pytest.raises(DuplicateEmbeddingError):
        await dal.update_embedding(make_collection(), uuid4(), [1.0, 2.0, 3.0], None, None)


@pytest.mark.asyncio
async def test_a_bad_vector_in_an_update_is_caught_before_a_connection_is_taken():
    """
    Hashing happens before the transaction opens, so a bad vector costs no connection.

    Under load the pool is the scarce thing; a malformed request should not be able to occupy
    one of its slots to find out it was malformed.
    """
    dal, _, pool = make_dal()

    with pytest.raises(EmbeddingDimensionMismatchError):
        await dal.update_embedding(make_collection(), uuid4(), [1.0, 2.0], None, None)

    assert pool.acquired == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("tag, expected", [("DELETE 1", True), ("DELETE 0", False)])
async def test_deleting_an_embedding_reports_whether_a_row_went(tag, expected):
    """A row hidden by RLS deletes nothing, which has to be distinguishable from success."""
    dal, _, _ = make_dal(execute_result=tag)

    assert await dal.delete_embedding(make_collection(), uuid4()) is expected


@pytest.mark.asyncio
async def test_listing_embeddings_pages_on_a_unique_ordering():
    """
    Paging orders by `embedding_id`.

    `created_at` is not unique -- a bulk write gives a whole batch the same value -- so
    ordering by it lets rows move between pages, which shows up as a paging client seeing one
    row twice and never seeing another.
    """
    dal, conn, _ = make_dal(results=[[embedding_row(), embedding_row()]])

    embeddings = await dal.list_embeddings_in_collection(make_collection(id=7), offset=10, limit=5)

    assert conn.params == [7, 10, 5]
    sql = normalized(conn.sql)
    assert "ORDER BY embedding_id" in sql
    assert "OFFSET $2::int LIMIT $3::int" in sql
    assert len(embeddings) == 2


@pytest.mark.asyncio
async def test_a_bulk_read_of_a_known_vector_type_queries_only_that_table():
    """Given the type, there is exactly one table an id could be in."""
    dal, conn, _ = make_dal(results=[[embedding_row()]])

    embeddings = await dal.get_embeddings_bulk([uuid4()], VectorType.halfvec)

    assert len(conn.queries) == 1
    assert "embeddings_halfvec" in conn.sql
    assert len(embeddings) == 1


@pytest.mark.asyncio
async def test_a_bulk_read_without_a_vector_type_queries_both_tables():
    """
    With no type to narrow it, both tables are searched and the results combined.

    Callers reach this when they hold ids but not the collections they came from, so missing a
    table would silently drop every halfvec row from the answer.
    """
    dal, conn, _ = make_dal(results=[[embedding_row()], [embedding_row()]])

    embeddings = await dal.get_embeddings_bulk([uuid4()], None)

    tables = [sql for sql, _ in conn.queries]
    assert any("embeddings_vector" in sql for sql in tables)
    assert any("embeddings_halfvec" in sql for sql in tables)
    assert len(embeddings) == 2
    # both tables were read inside the one RLS transaction
    assert conn.transactions == 1


@pytest.mark.asyncio
async def test_a_bulk_read_can_be_narrowed_to_one_collection():
    """The collection predicate is added when one is given, and absent when it is not."""
    dal, conn, _ = make_dal(results=[[embedding_row()]])
    await dal.get_embeddings_bulk([uuid4()], VectorType.vector, collection_id=7)
    assert "collection_id = 7" in normalized(conn.sql)

    dal, conn, _ = make_dal(results=[[embedding_row()]])
    await dal.get_embeddings_bulk([uuid4()], VectorType.vector)
    assert "collection_id" not in normalized(conn.sql)


@pytest.mark.asyncio
async def test_narrowing_applies_to_both_tables_when_the_type_is_unknown():
    """
    The two-table read narrows both of its queries, not just the first.

    The predicate is appended per table, so a caller asking for specific ids within one
    collection would otherwise get that collection's vector rows plus every halfvec row it
    could see with a matching id.
    """
    dal, conn, _ = make_dal(results=[[embedding_row()], [embedding_row()]])

    await dal.get_embeddings_bulk([uuid4()], None, collection_id=7)

    assert len(conn.queries) == 2
    assert all("collection_id = 7" in normalized(sql) for sql, _ in conn.queries)


@pytest.mark.asyncio
async def test_an_ordered_bulk_read_of_nothing_never_reaches_the_database():
    """An empty id list has no answer to look up."""
    dal, _, pool = make_dal()

    assert await dal.get_embeddings_bulk_from_collection_ordered([], make_collection()) == []
    assert pool.acquired == 0


@pytest.mark.asyncio
async def test_an_ordered_bulk_read_returns_each_rows_position_in_the_request():
    """
    Rows come back tagged with the zero-based index of the id that asked for them.

    Rows the caller cannot see are simply absent, so position in the result says nothing about
    position in the request -- the ordinality does. It is 1-based in SQL and 0-based here,
    which is the off-by-one this pins.
    """
    ids = [uuid4(), uuid4(), uuid4()]
    # the middle id is hidden or absent, so only two rows come back
    rows = [
        embedding_row(embedding_id=ids[0], ord=1),
        embedding_row(embedding_id=ids[2], ord=3),
    ]
    dal, conn, _ = make_dal(results=[rows])

    results = await dal.get_embeddings_bulk_from_collection_ordered(ids, make_collection(id=7))

    assert conn.params == [ids, 7]
    assert [index for index, _ in results] == [0, 2]
    assert [embedding.embedding_id for _, embedding in results] == [ids[0], ids[2]]


@pytest.mark.asyncio
async def test_counting_embeddings_returns_the_rls_filtered_count():
    """The count is what this caller can read, which is the number the API reports."""
    dal, conn, _ = make_dal(results=[[{"cnt": 12}]])

    assert await dal.count_available_embeddings_in_collection(make_collection(id=7)) == 12
    assert conn.params == [7]


@pytest.mark.asyncio
async def test_counting_embeddings_with_no_row_is_zero():
    """A COUNT always returns a row, but a missing one must not be reported as None."""
    dal, _, _ = make_dal(results=[[]])

    assert await dal.count_available_embeddings_in_collection(make_collection()) == 0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_collection_search_binds_its_first_three_parameters_in_order():
    """
    $1/$2/$3 are the collection, the query vector and top_k, and the metric's extras follow.

    `build_search_sql` numbers filter and threshold parameters from $4 on the assumption that
    those three are already taken, so the two have to agree about the prefix.
    """
    dal, conn, _ = make_dal(results=[[embedding_row()]])

    await dal.search_embeddings_in_collection(
        make_collection(id=7),
        [1.0, 0.0, 0.0],
        top_k=5,
        min_value=None,
        max_value=0.5,
        distance_metric=DistanceMetric.cosine_distance,
        filters={"kind": "doc"},
    )

    assert conn.params == [7, [1.0, 0.0, 0.0], 5, "kind", "doc", 0.5]
    sql = normalized(conn.sql)
    assert "FROM embeddings_vector" in sql
    assert "embedding <=> $2::vector" in sql


@pytest.mark.asyncio
async def test_single_collection_search_casts_the_query_vector_to_the_collections_type():
    """
    A halfvec collection is searched with a halfvec query.

    pgvector will not use a halfvec index for a `vector` operand, so the wrong cast is a
    sequential scan that still returns the right rows -- correct and arbitrarily slow.
    """
    dal, conn, _ = make_dal(results=[[embedding_row()]])

    await dal.search_embeddings_in_collection(
        make_collection(vector_type="halfvec"),
        [1.0, 0.0, 0.0],
        top_k=5,
        min_value=None,
        max_value=None,
        distance_metric=DistanceMetric.l2_distance,
        filters=None,
    )

    sql = normalized(conn.sql)
    assert "FROM embeddings_halfvec" in sql
    assert "embedding <-> $2::halfvec" in sql


@pytest.mark.asyncio
async def test_searching_no_collections_never_reaches_the_database():
    """Nowhere to look means no hits, without a query."""
    dal, _, pool = make_dal()

    rows = await dal.search_embeddings_across_collections(
        [],
        [1.0, 0.0, 0.0],
        top_k=5,
        min_value=None,
        max_value=None,
        distance_metric=DistanceMetric.l2_distance,
        filters=None,
    )

    assert rows == []
    assert pool.acquired == 0


@pytest.mark.asyncio
async def test_a_cross_collection_search_only_queries_the_collections_that_could_match():
    """
    Collections of another vector type or another dimensionality are dropped.

    They are queried as one statement against one table, so a mismatched collection cannot be
    included: a different dimensionality makes the distance operator error out, which would
    fail the whole search over a collection that could not have held a hit anyway.
    """
    collections = [
        make_collection(id=1, dimensions=3, vector_type="vector"),
        make_collection(id=2, dimensions=4, vector_type="vector"),
        make_collection(id=3, dimensions=3, vector_type="halfvec"),
        make_collection(id=4, dimensions=3, vector_type="vector"),
    ]
    dal, conn, _ = make_dal(results=[[embedding_row()]])

    await dal.search_embeddings_across_collections(
        collections,
        [1.0, 0.0, 0.0],
        top_k=5,
        min_value=None,
        max_value=None,
        distance_metric=DistanceMetric.l2_distance,
        filters=None,
        vector_type=VectorType.vector,
    )

    assert conn.params[0] == [1, 4]
    assert "FROM embeddings_vector" in normalized(conn.sql)


@pytest.mark.asyncio
async def test_a_cross_collection_search_with_no_matching_collection_is_empty_not_an_error():
    """
    Every collection filtered out means no hits, which is a result rather than a failure.

    A caller searching a mixed set of collections with a 3-dimensional vector is asking a
    well-formed question; the answer for the halfvec ones is just that they hold nothing that
    could match.
    """
    collections = [make_collection(id=3, dimensions=3, vector_type="halfvec")]
    dal, _, pool = make_dal()

    rows = await dal.search_embeddings_across_collections(
        collections,
        [1.0, 0.0, 0.0],
        top_k=5,
        min_value=None,
        max_value=None,
        distance_metric=DistanceMetric.l2_distance,
        filters=None,
        vector_type=VectorType.vector,
    )

    assert rows == []
    assert pool.acquired == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_vector_type", [VectorType.halfvec, "halfvec"])
@pytest.mark.parametrize("requested_vector_type", [VectorType.halfvec, "halfvec"])
async def test_the_vector_type_filter_compares_by_value(stored_vector_type, requested_vector_type):
    """
    Matching is by value, so either side may be the enum or its string.

    `Collection.from_record` coerces the stored side, but a Collection built in code (a
    script, a test, a future caller) can hold the plain string, and `vector_type` arrives as
    whatever the caller passed. `VectorType` is a StrEnum precisely so all four combinations
    compare equal. Comparing against `vector_type.value` instead -- which is what this filter
    used to do -- handles the stored side fine and then breaks as an AttributeError the moment
    the argument itself is a string.
    """
    collections = [Collection(**collection_row(id=3, vector_type=stored_vector_type))]
    dal, conn, _ = make_dal(results=[[embedding_row()]])

    await dal.search_embeddings_across_collections(
        collections,
        [1.0, 0.0, 0.0],
        top_k=5,
        min_value=None,
        max_value=None,
        distance_metric=DistanceMetric.l2_distance,
        filters=None,
        vector_type=requested_vector_type,
    )

    assert conn.params[0] == [3]
    assert "FROM embeddings_halfvec" in normalized(conn.sql)
