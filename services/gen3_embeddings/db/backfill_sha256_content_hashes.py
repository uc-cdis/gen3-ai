"""
Data migration: populate `embedding_hash_v2` / `metadata_hash_v2` for existing rows.

Companion to db/migrations/20260826120000_sha256_content_hashes.sql, which adds the columns
and their unique index but cannot fill them. The old hashes were md5 over the JSON text the
app happened to send (`json.dumps` of float64s, e.g. `[0.1, 0.2]`); Postgres cannot reproduce
that text from a stored float4 vector (`[0.1,0.2]`), so the new hash has to be computed
outside the database, from the same code path the app now uses.

Run it after deploying the app code that writes the v2 columns:

    cd services/gen3_embeddings
    uv run python db/backfill_sha256_content_hashes.py            # uses DB_CONNECTION_STRING
    uv run python db/backfill_sha256_content_hashes.py --dry-run
    uv run python db/backfill_sha256_content_hashes.py --dsn postgresql://...

Until it completes, rows written before that deploy have NULL v2 hashes, which means they sit
outside the new unique index and a re-POST of their content inserts a duplicate instead of
returning 409.

Properties:

- Idempotent. Only rows with a NULL `embedding_hash_v2` are touched, so an interrupted run
  can simply be re-run.
- Incremental. Rows are processed in keyset-paginated batches within their own transaction,
  so it does not hold one long transaction open against a live table and can be stopped at
  any point.
- Non-destructive. The legacy md5 columns are never written; the app keeps them satisfied on
  its own until the contract migration drops them.
- Honest about collisions. Rows whose content is genuinely identical (which the md5 hashes
  failed to notice, most likely on a halfvec collection where two vectors differ only below
  float16 precision) cannot both take the same v2 hash. The batch's UPDATE would fail, so
  those rows are retried one at a time, left NULL, and printed at the end for a human to
  resolve. Nothing is deleted.

Connects with the credentials in DB_CONNECTION_STRING by default. That role must be able to
read and write the embeddings tables; note that these tables FORCE row level security, so a
non-superuser role that owns them is subject to the authz policies and would see no rows. Run
as the admin/superuser used for migrations.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

import asyncpg
from asyncpg.exceptions import UniqueViolationError
from pgvector.asyncpg import register_vector

# allow running as a plain script from the service directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gen3_embeddings import config  # noqa: E402
from gen3_embeddings.database import hashing  # noqa: E402
from gen3_embeddings.models.schemas import VectorType  # noqa: E402

# Rows per transaction. Each row carries a full vector, so this is a memory bound as much as
# a lock-duration one: 1000 x 8192 float32 is ~32MB.
BATCH_SIZE = 500

TABLES: dict[str, VectorType] = {
    "embeddings_vector": VectorType.vector,
    "embeddings_halfvec": VectorType.halfvec,
}


def row_hashes(embedding, metadata, vector_type: VectorType) -> tuple[UUID, UUID]:
    """
    Compute the v2 hashes for one existing row.

    The vector arrives from pgvector's binary codec as a Vector/HalfVector holding exactly the
    stored float32/float16 values, and its numpy view is the same bytes the app hashes when
    writing a new row. So a backfilled row and a fresh write of identical content agree,
    which is the whole point: the unique index has to see them as one.

    Args:
        embedding: Vector or HalfVector as decoded by pgvector.
        metadata: The row's metadata; jsonb comes back as text unless a codec is registered.
        vector_type (VectorType): Which table's storage type this row uses.

    Returns:
        tuple[UUID, UUID]: (embedding_hash_v2, metadata_hash_v2).
    """
    array = embedding.to_numpy().reshape(1, -1)
    embedding_hash = hashing.hash_rows(array)[0]

    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    # NULL metadata hashes as {}, matching how the write paths treat a missing metadata field
    metadata_hash = hashing.hash_metadata(metadata)

    return embedding_hash, metadata_hash


async def backfill_table(conn: asyncpg.Connection, table: str, vector_type: VectorType, dry_run: bool) -> dict:
    """
    Backfill one embeddings table.

    Args:
        conn (asyncpg.Connection): Connection with pgvector codecs registered.
        table (str): Table to backfill.
        vector_type (VectorType): That table's storage type.
        dry_run (bool): Compute hashes and report, but write nothing.

    Returns:
        dict: Counts, plus the embedding_ids of any rows left unhashed because their content
            collides with another row's.
    """
    scanned = 0
    updated = 0
    collisions: list[tuple[int, UUID]] = []
    # keyset pagination on the table's (collection_id, embedding_id) unique key, so progress
    # cannot be lost or repeated as rows are updated out from under the scan
    cursor: tuple[int, UUID] | None = None

    while True:
        where_cursor = "AND (collection_id, embedding_id) > ($1::bigint, $2::uuid)" if cursor else ""
        select_sql = f"""
            SELECT collection_id, embedding_id, embedding, metadata
            FROM {table}
            WHERE embedding_hash_v2 IS NULL
            {where_cursor}
            ORDER BY collection_id, embedding_id
            LIMIT {BATCH_SIZE}
        """
        rows = await conn.fetch(select_sql, *cursor) if cursor else await conn.fetch(select_sql)
        if not rows:
            break

        scanned += len(rows)
        cursor = (rows[-1]["collection_id"], rows[-1]["embedding_id"])

        computed = [
            (row["collection_id"], row["embedding_id"], *row_hashes(row["embedding"], row["metadata"], vector_type))
            for row in rows
        ]
        if dry_run:
            updated += len(computed)
            print(f"  {table}: would update {len(computed)} rows (running total {updated})")
            continue

        update_sql = f"""
            UPDATE {table} AS t
            SET embedding_hash_v2 = source.embedding_hash,
                metadata_hash_v2 = source.metadata_hash
            FROM unnest($1::bigint[], $2::uuid[], $3::uuid[], $4::uuid[])
                AS source(collection_id, embedding_id, embedding_hash, metadata_hash)
            WHERE t.collection_id = source.collection_id
              AND t.embedding_id = source.embedding_id
        """
        columns = list(zip(*computed))
        try:
            async with conn.transaction():
                await conn.execute(update_sql, *[list(column) for column in columns])
            updated += len(computed)
        except UniqueViolationError:
            # Somewhere in this batch two rows (or a row and an already-backfilled row) hold
            # the same content. Redo it row by row to isolate them; the rest still lands.
            for collection_id, embedding_id, embedding_hash, metadata_hash in computed:
                try:
                    async with conn.transaction():
                        await conn.execute(
                            update_sql,
                            [collection_id],
                            [embedding_id],
                            [embedding_hash],
                            [metadata_hash],
                        )
                    updated += 1
                except UniqueViolationError:
                    collisions.append((collection_id, embedding_id))

        print(f"  {table}: updated {updated} rows, {len(collisions)} collisions so far")

    return {"table": table, "scanned": scanned, "updated": updated, "collisions": collisions}


async def main(dsn: str, dry_run: bool) -> int:
    """
    Backfill both embeddings tables and report.

    Args:
        dsn (str): Postgres connection string.
        dry_run (bool): Compute hashes and report, but write nothing.

    Returns:
        int: Process exit code; non-zero if any row was left unhashed.
    """
    conn = await asyncpg.connect(dsn)
    await register_vector(conn)
    try:
        results = [await backfill_table(conn, table, vector_type, dry_run) for table, vector_type in TABLES.items()]
    finally:
        await conn.close()

    print("\n=== backfill summary" + (" (DRY RUN, nothing written)" if dry_run else "") + " ===")
    exit_code = 0
    for result in results:
        print(f"{result['table']}: scanned {result['scanned']}, updated {result['updated']}")
        if result["collisions"]:
            exit_code = 1
            print(
                f"  {len(result['collisions'])} row(s) left with NULL hashes because another row holds identical\n"
                f"  content (same stored vector, metadata, and authz). The md5 hashes did not catch these.\n"
                f"  Decide which to keep, delete the others, and re-run. (collection_id, embedding_id):"
            )
            for collection_id, embedding_id in result["collisions"]:
                print(f"    ({collection_id}, {embedding_id})")

    if not exit_code and not dry_run:
        print("no rows left with a NULL embedding_hash_v2")

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", default=str(config.DB_CONNECTION_STRING), help="Postgres connection string")
    parser.add_argument("--dry-run", action="store_true", help="compute and report, write nothing")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args.dsn, args.dry_run)))
