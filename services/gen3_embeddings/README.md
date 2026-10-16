# Gen3 Embeddings

An authorization-scoped vector store for Gen3, built on PostgreSQL and
[pgvector](https://github.com/pgvector/pgvector). Callers store embeddings in collections,
read them back, and run similarity search, and only ever see rows their Gen3 grants cover.

The endpoints are documented in the [API reference](https://uc-cdis.github.io/gen3-ai/api.html).

## Concepts

* **Collection**: a named group of embeddings. Two things are fixed when it is created and
  cannot change: `dimensions`, which every vector in it must have, and `vector_type`, which is
  `vector` (float32) or `halfvec` (float16, half the storage). Names are lower-cased on the way
  in and may only contain `a-z`, `0-9`, `-` and `_`.
* **Embedding**: a vector, optional JSON `metadata`, and an `authz` resource path that decides
  who can see it. A collection holds at most one embedding with a given vector, metadata and
  `authz`.
* **Search**: nearest neighbours to a query vector, within one collection or across several,
  with a choice of distance metric, distance thresholds and metadata filters. Without a vector
  index, search scans the collection exactly. See [Vector indexes](#vector-indexes).

## Authorization

Every request is checked twice: against the Gen3 policy engine (Arborist), and by
PostgreSQL row-level security (RLS) on every row it touches.

* A collection's resource path is `/vectorstore/collections/{collection_name}`. Policies have
  to name each collection, in lowercase, exactly as it is stored.
* An embedding's `authz` defaults to its collection's path. A write can set it to any resource
  path the caller holds the same action on.
* Actions are `read`, `create`, `update` and `delete` on the service `gen3-embeddings`.
* Reads return only what the caller's grants cover, so "not found" and "not authorized" look
  the same.

A minimal Arborist `user.yaml` that grants one user full access to one collection:

```yaml
authz:
  resources:
  - name: vectorstore
    subresources:
    - name: collections
      subresources:
      - name: public

  roles:
  - id: gen3-embeddings-user
    permissions:
    - id: embeddings_reader
      action: {service: gen3-embeddings, method: read}
    - id: embeddings_creator
      action: {service: gen3-embeddings, method: create}
    - id: embeddings_updater
      action: {service: gen3-embeddings, method: update}
    - id: embeddings_deleter
      action: {service: gen3-embeddings, method: delete}

  policies:
  - id: gen3-embeddings-public
    role_ids: [gen3-embeddings-user]
    resource_paths: [/vectorstore/collections/public]

users:
  someone@example.org:
    policies: [gen3-embeddings-public]
```

## Database

The service needs PostgreSQL with pgvector 0.8 or later, and two roles:

| Role | Used by | Needs |
| --- | --- | --- |
| Migration role (`DB_MIGRATION_USER`) | `just db_*` recipes, dbmate | Owns the database and its tables. Superuser, unless the `vector` extension already exists. |
| Service role (`PGUSER`) | the running service | `SELECT, INSERT, UPDATE, DELETE` on `collections`, `embeddings_vector` and `embeddings_halfvec`. Must not be `SUPERUSER` or `BYPASSRLS`, except under `DEBUG_SKIP_AUTH` (see [Running locally](#running-locally)). |

At startup the service refuses to run if its role can bypass RLS, or if RLS is not enabled
and forced on those three tables.

[db/app_user.sql](db/app_user.sql) creates the service role and grants it exactly that. It is
idempotent and runs as the migration role:

```bash
psql -v ON_ERROR_STOP=1 -v app_user="$PGUSER" -v app_password="$PGPASSWORD" -f db/app_user.sql
```

From the repository root:

```bash
just db_setup gen3_embeddings     # create the database, owned by DB_MIGRATION_USER
just db_load gen3_embeddings      # load db/schema.sql into an empty database
just db_migrate gen3_embeddings   # or: apply pending migrations from db/migrations
just db_app_user gen3_embeddings  # create PGUSER and grant it access (db_load and db_migrate run this)
```

Schema migrations live in [db/migrations](db/migrations) and follow an expand/contract pattern,
so they can run while the previous release is still serving.
[db/backfill_sha256_content_hashes.py](db/backfill_sha256_content_hashes.py) fills content
hashes for rows written before the `20260826120000_sha256_content_hashes` migration. Its
docstring explains when to run it.

### Vector indexes

The service does not create vector indexes. Operators choose which collections to index and add
each index as a dbmate migration in a deployment-owned folder, applied with
`just db_index_migrate`. Search picks a new index up on the next restart. See
[db/index_migrations/README.md](db/index_migrations/README.md) for the workflow and the index
shapes search can use.

## Configuration

Set as environment variables, or in `services/gen3_embeddings/.env`. Defaults are in
[src/gen3_embeddings/config.py](src/gen3_embeddings/config.py) and
[libraries/common/src/common/config.py](../../libraries/common/src/common/config.py).

| Variable | Default | Purpose |
| --- | --- | --- |
| `PGHOST`, `PGPORT`, `PGDATABASE` | `localhost`, `5432`, `gen3embeddings` | Database location. |
| `PGUSER`, `PGPASSWORD` | `postgres`, none | The service role. |
| `DB_CONNECTION_STRING` | built from the `PG*` variables | Overrides them all. |
| `DB_MIGRATION_USER`, `DB_MIGRATION_PASSWORD` | `PGUSER`, `PGPASSWORD` | The migration role. Read by the `just db_*` recipes only. |
| `PGPOOL_MIN_SIZE`, `PGPOOL_MAX_SIZE` | `1`, `5` | Connection pool size per process. |
| `DB_STATEMENT_TIMEOUT_MS` | `60000` | Cancels any query that runs longer. `0` disables. |
| `HNSW_EF_SEARCH` | `100` | HNSW candidate list size: higher is better recall, slower search. |
| `HNSW_ITERATIVE_SCAN` | `relaxed_order` | Keeps an HNSW scan going until RLS and metadata filters leave `top_k` rows. |
| `ARBORIST_URL` | `http://arborist-service` | Gen3 policy engine. |
| `GEN3_EMBEDDINGS_URL_PREFIX` | empty | Path prefix the service is mounted under. |
| `DEBUG_SKIP_AUTH` | `False` | Development only. See [Running locally](#running-locally). |

Request size limits (`MAX_*`) are documented next to their definitions in `config.py`.
Traces, metrics and logs are covered in [docs/observability.md](../../docs/observability.md).

## Running locally

Start a pgvector-enabled Postgres:

```bash
docker run --name pgvector -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:pg18-trixie
```

Create `services/gen3_embeddings/.env`. `PGUSER` is the service role, which
`just db_app_user` creates, so pick any name and password for it:

```bash
PGHOST=localhost
PGPORT=5432
PGDATABASE=gen3_embeddings
PGUSER=gen3_embeddings_app
PGPASSWORD=gen3_embeddings_app
DB_MIGRATION_USER=postgres
DB_MIGRATION_PASSWORD=postgres

ARBORIST_URL=http://localhost:4280

ENABLE_OPENTELEMETRY_TRACES=False
ENABLE_METRICS=False
```

Then, from the repository root, set up the database, load the test collections, and start the
service:

```bash
just db_setup gen3_embeddings && just db_load gen3_embeddings && just db_seed gen3_embeddings
just run gen3_embeddings
```

`just db_seed` writes seven test collections (`public`, `internal`, `noaccess`, `d3vector`,
`d200vector`, `d3halfvec`, `d3000halfvec`) through the service's own data access layer as
`PGUSER`. Collections that already exist are skipped; `just db_seed gen3_embeddings --reset`
reloads them. The list and sizes are in [db/seed.py](db/seed.py).

Requests need a token from a Gen3 deployment whose Arborist grants access to those collections
(see [Authorization](#authorization)). With a local Helm deployment, forward its Arborist to
the `ARBORIST_URL` above:

```bash
kubectl port-forward svc/arborist-service 4280:80
```

`DEBUG_SKIP_AUTH=True` lets the service start without Arborist and accept requests with no
token. Such a request holds no grants, so under RLS every read would come back empty and every
write would be rejected. With it set, the service therefore refuses to start unless its role
bypasses RLS (`SUPERUSER` or `BYPASSRLS`, such as the migration role) -- the reverse of what a
deployment requires -- and the startup RLS check only warns. Never set it in a deployment.

Some requests to try:

```bash
export TOKEN=...
BASE=http://localhost:8000

curl "$BASE/vectorstore/collections" -H "Authorization: Bearer $TOKEN"

curl "$BASE/vectorstore/collections/public/embeddings?page=2&page_size=200" -H "Authorization: Bearer $TOKEN"

curl -X POST "$BASE/vectorstore/collections" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"collection_name": "scratch", "description": "scratch collection", "dimensions": 3, "vector_type": "halfvec"}'

curl -X POST "$BASE/vectorstore/collections/scratch/embeddings" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"embeddings": [{"embedding": [0.1, 0.2, 0.3], "metadata": {"source": "some_file.md"}}]}'

curl -X POST "$BASE/vectorstore/collections/d3halfvec/search" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"input": [1.0, 0.0, 0.0], "top_k": 5, "distance_metric": "l2_distance"}'

curl -X POST "$BASE/vectorstore/search?collections=public,d3vector" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"input": [1.0, 0.0, 0.0], "top_k": 5}'
```

## Tests

```bash
# from the repo root; this is what CI runs
just test gen3_embeddings

# or from this directory
uv run pytest -n auto . -vv
```

The tests create and drop their own database and roles, so they connect as a Postgres
superuser (`postgres`/`postgres` on localhost by default). To use something else, copy
[tests/.env.example](tests/.env.example) to `tests/.env`. They also need `dbmate` and `psql` on
`PATH`.

Every run measures coverage of `src/gen3_embeddings` and fails below 90%, separately from
whether the tests passed: the last two lines of the output tell the two apart. Turn it off when
running a subset, which would otherwise fail the gate:

```bash
uv run pytest tests/test_db_dal.py -q --no-cov

# a browsable line-by-line report in htmlcov/
uv run pytest -n auto . --cov-report=html
```
