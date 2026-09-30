# Vector index migrations (operator-owned)

This folder holds the vector indexes **you** decide to build for your deployment. The service
never creates vector indexes itself. Which collections deserve an index, and which kind, depends
on your data size, your memory budget, and the metrics your users search with.

Index changes are made as [dbmate](https://github.com/amacneil/dbmate) migrations so that every
change to the database is written down and replayable:

* Each index is a file with an up (create) and a down (drop) step.
* dbmate records every applied file in its own table, `index_migrations`, separate from the
  service's `schema_migrations`. `SELECT * FROM index_migrations;` shows what was applied.
* These migrations are never dumped into `db/schema.sql`. That file describes the schema every
  deployment shares, and your indexes are specific to your deployment.

The official migrations in `../migrations` never run these files, and these recipes never run
those. dbmate only reads `.sql` files directly inside the directory it is given, not
subdirectories.

## Where your files live

By default the recipes read this folder. Collection ids differ between deployments, so an
index file only makes sense for the database it was written for. Keep your files in your
deployment's own repository and point the recipes at it:

```bash
export INDEX_MIGRATIONS_DIR=/path/to/your/deployment/index_migrations
```

`*.sql` files in this folder are git-ignored in the upstream repository. Everything under
`db/` is copied into the service image, so a committed index file would ship to every
deployment and build an index on whatever collection happens to have that id there.

## Workflow

```bash
# 1. find the collection's id, dimensions and vector_type (run as the admin/owner role)
#    collections.id is the value that goes in the index's WHERE collection_id = <id>
psql -c "SELECT id, collection_name, dimensions, vector_type FROM collections;"

# 2. create an empty migration file
just db_index_new "hnsw cosine for my_collection"

# 3. write the SQL (examples below) into the new file

# 4. apply it. This blocks until the index is built, which can take days on a large
#    collection -- see "Long builds" below before running it against real data
just db_index_migrate

# 5. once it has finished, restart the gen3_embeddings pods -- see "Restart the service"

# check what has been applied, or undo the latest one
just db_index_status
just db_index_rollback
```

The recipes connect with the `PG*` settings from `services/gen3_embeddings/.env`, the same
as `just db_migrate`. Building an index requires owning the table, so use the admin/owner
role, not the app's limited role.

## List existing indexes

This lists every vector index on the embeddings tables, including ones built outside these
migrations:

```sql
SELECT i.relname AS index_name,
       t.relname AS table_name,
       substring(pg_get_expr(x.indpred, x.indrelid)
                 FROM '^\(collection_id = (\d+)\)$')::bigint AS collection_id,
       am.amname AS method,
       opc.opcname AS opclass,
       x.indisvalid AS valid,
       pg_size_pretty(pg_relation_size(i.oid)) AS size,
       pg_get_indexdef(i.oid) AS definition
FROM pg_index x
JOIN pg_class i ON i.oid = x.indexrelid
JOIN pg_class t ON t.oid = x.indrelid
JOIN pg_am am ON am.oid = i.relam
JOIN pg_opclass opc ON opc.oid = x.indclass[0]
WHERE t.relname IN ('embeddings_vector', 'embeddings_halfvec')
  AND am.amname IN ('hnsw', 'ivfflat')
ORDER BY collection_id, index_name;
```

Search ignores any row where `valid` is false, because the build failed or is still
running. It also ignores any row where `collection_id` is empty, because the `WHERE` clause
is not exactly `collection_id = <id>`.

## Writing an index the service will use

Search only uses an index whose shape it recognizes. On startup it reads the Postgres catalog
and matches each index against the SQL it emits. An index with any other shape is ignored,
and searches on that collection fall back to a full sequential scan. That is slow, but the
results are still correct. The rules:

1. **One statement per file, with `transaction:false` on both the up and down lines.**
   `CREATE INDEX CONCURRENTLY` cannot run inside a transaction, and dbmate sends a
   multi-statement file as one transaction. That includes `SET` statements, so do not add any.
2. **`CONCURRENTLY`** keeps the table readable and writable while the index builds.
3. **One collection per index:** the `WHERE` clause is exactly `collection_id = <id>`, with a
   literal integer. `IN (...)` or extra `AND` terms are not recognized.
4. **The indexed expression matches the collection exactly:**

   | Collection `vector_type` | Table                | Expression                                | Operator classes                                                                 |
   | ------------------------ | -------------------- | ----------------------------------------- | -------------------------------------------------------------------------------- |
   | `vector`                 | `embeddings_vector`  | `(embedding::vector(<dims>))`             | `vector_cosine_ops`, `vector_l2_ops`, `vector_ip_ops`, `vector_l1_ops`           |
   | `halfvec`                | `embeddings_halfvec` | `(embedding::halfvec(<dims>))`            | `halfvec_cosine_ops`, `halfvec_l2_ops`, `halfvec_ip_ops`, `halfvec_l1_ops`       |
   | either                   | same as above        | `(binary_quantize(embedding)::bit(<dims>))` | `bit_hamming_ops` (binary index)                                               |

   `<dims>` is the collection's `dimensions`. The access method is `hnsw` or `ivfflat`.

### Which operator class

* **A direct index serves only its own metric.** An index built with `vector_cosine_ops`
  serves `cosine_distance` and `cosine_similarity` searches, and nothing else. Build one
  index per metric your users search with.
* **A binary index serves every metric.** It picks candidates by Hamming distance, and the
  service then re-ranks them exactly by the requested metric. It is much smaller than a
  direct index, at some cost in recall. The candidate pool size is set with
  `BINARY_RESCORE_MULTIPLIER`, `BINARY_RESCORE_MIN` and `BINARY_RESCORE_MAX`.
* **When a collection has both, the direct index wins** for its metric, and the binary index
  serves the rest.

pgvector's HNSW limits decide what is possible for large embeddings. A `vector` index supports
up to 2,000 dimensions and a `halfvec` index up to 4,000. Above that, only a binary index can
be built.

### Example: HNSW, cosine, on a `vector` collection

```sql
-- migrate:up transaction:false
CREATE INDEX CONCURRENTLY IF NOT EXISTS embeddings_vector_c42_hnsw_cosine
ON embeddings_vector USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
WHERE collection_id = 42;

-- migrate:down transaction:false
DROP INDEX CONCURRENTLY IF EXISTS embeddings_vector_c42_hnsw_cosine;
```

### Example: HNSW, L2, on a `halfvec` collection

```sql
-- migrate:up transaction:false
CREATE INDEX CONCURRENTLY IF NOT EXISTS embeddings_halfvec_c7_hnsw_l2
ON embeddings_halfvec USING hnsw ((embedding::halfvec(3072)) halfvec_l2_ops)
WHERE collection_id = 7;

-- migrate:down transaction:false
DROP INDEX CONCURRENTLY IF EXISTS embeddings_halfvec_c7_hnsw_l2;
```

### Example: binary-quantized HNSW (serves every metric)

```sql
-- migrate:up transaction:false
CREATE INDEX CONCURRENTLY IF NOT EXISTS embeddings_vector_c42_hnsw_binary
ON embeddings_vector USING hnsw ((binary_quantize(embedding)::bit(1536)) bit_hamming_ops)
WHERE collection_id = 42;

-- migrate:down transaction:false
DROP INDEX CONCURRENTLY IF EXISTS embeddings_vector_c42_hnsw_binary;
```

HNSW parameters such as `WITH (m = 16, ef_construction = 64)` may be added after the
expression. They do not affect whether the service recognizes the index.

## Restart the service

The service reads the index catalog **once, at startup**. It does not notice an index created
or dropped while it is running, so restart the gen3_embeddings pods after
`just db_index_migrate` or `just db_index_rollback` finishes.

Finishing and "the index is built" are the same moment. `CREATE INDEX CONCURRENTLY` returns
only once the build has completed and the index is valid, so a migration that finished
without an error has a usable index. Keep these in mind:

* **Restarting during a build does no harm, but it does not help.** The service only
  counts valid indexes, and an index under construction is not valid yet. Restart again
  once the migration finishes.
* **A failed migration leaves nothing to restart for.** The half-built index is `INVALID`,
  the service ignores it, and searches keep their current behavior. See
  "Long builds and failures" below.
* **If dbmate disconnected mid-build,** there is no "finished" to wait for. Restart once the
  index shows as valid, as described under "If the client disconnects mid-build".
* **Restart after a rollback too.** Otherwise the service keeps emitting SQL for the dropped
  index, and searches on that collection run as sequential scans until the restart.

The startup log reports how many indexes it found:

```text
Discovered 1 usable vector index(es) across 1 collection(s)
```

A count lower than expected means an index does not have the shape above. To confirm a
specific search uses the index, `EXPLAIN` the query and look for your index name.

## Long builds and failures

`just db_index_migrate` returns only when the `CREATE INDEX CONCURRENTLY` statement returns,
which is when the index is fully built. On a large collection that can take hours or days.
dbmate applies pending files one after another, so a long build also holds up every file
after it. Apply one large index per run.

### Session settings

A migration file cannot `SET` anything (rule 1), so the index recipes pass Postgres session
settings in the connection URL instead. By default they send `statement_timeout=0`, so a
build is not killed by a server-side timeout. Replace the defaults with `INDEX_PG_PARAMS`,
and keep `statement_timeout=0` in the list:

```bash
INDEX_PG_PARAMS="statement_timeout=0&maintenance_work_mem=8GB&max_parallel_maintenance_workers=4" \
  just db_index_migrate
```

HNSW builds are much faster when the graph fits in `maintenance_work_mem`. See the
[pgvector notes on index build time](https://github.com/pgvector/pgvector#index-build-time).

### Run it somewhere that stays up

A build lasting days should not depend on a laptop, a VPN, or a `kubectl port-forward`.
Run it from a host or pod close to the database that stays up for the whole build: a
`tmux`/`screen` session on a bastion host, or a pod in the cluster. The gen3_embeddings
image already contains `dbmate` and this folder. The equivalent of the recipe, inside a pod,
with your files mounted at `/index_migrations`:

```bash
dbmate --url "postgresql://$PGUSER:$PGPASSWORD@$PGHOST:$PGPORT/$PGDATABASE?sslmode=require&statement_timeout=0" \
  --migrations-dir /index_migrations --migrations-table index_migrations --no-dump-schema \
  migrate
```

### Watch progress

From another session:

```sql
SELECT phase, round(100.0 * blocks_done / nullif(blocks_total, 0), 1) AS "%"
FROM pg_stat_progress_create_index;
```

### If the client disconnects mid-build

Killing or disconnecting dbmate does **not** stop the build. The server keeps running the
statement to the end, but dbmate never records the migration in `index_migrations`.

1. Wait until the build has finished: `pg_stat_progress_create_index` no longer shows it.
   **Do not rerun the migration while the build is still running.** The rerun waits on the
   running build's table lock, silently, for as long as the build takes. When it gets the
   lock, `IF NOT EXISTS` finds the index and records the migration as applied without
   checking it. If the build failed, the history then says applied while the index is
   `INVALID`.
2. Check the result:

   ```sql
   SELECT c.relname, x.indisvalid
   FROM pg_index x JOIN pg_class c ON c.oid = x.indexrelid
   WHERE c.relname = 'embeddings_vector_c42_hnsw_cosine';
   ```

3. If `indisvalid` is true, run `just db_index_migrate` again. `IF NOT EXISTS` skips the
   build, and the migration gets recorded. Then restart the service.
4. If `indisvalid` is false, the build failed. Handle it as in the next section.

### Failed builds

A failed or cancelled `CREATE INDEX CONCURRENTLY` leaves an `INVALID` index behind. The
service ignores it, but `IF NOT EXISTS` would skip rebuilding it and record the migration as
applied. Drop it first, then run `just db_index_migrate` again:

```sql
DROP INDEX CONCURRENTLY IF EXISTS embeddings_vector_c42_hnsw_cosine;
```

### Deleted collections

Deleting a collection removes its rows but not its index. Add a migration that drops the
index. Collection ids are not reused, so a stale index is only wasted space, never a wrong
result.
