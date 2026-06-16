# Gen3 Embeddings

Service which provides vector indices and embedding support.

## Implementation Details and Contraints

We use [pgvector](https://github.com/pgvector/pgvector) in a PostgreSQL database.

By default:

* We use a variable `vector`-type column which supports vectors of up to 2,000 dimensions
* We create a HNSW index for optimizing L2 distance for every service-level "Vector Index"
    * A "Vector Index" practically is a metadata table + a group of similarly dimensioned vectors
      in the embedding table
* Modifications to queries are possible through the API (e.g. sacrificing time for more accuracy), see the API specification for more details

> If you need to support indexing >2000 dimension vectors: the service will need to be modified. This is a limitation of pgvector. The `vector` column allows higher dimensionality, but the indexing can't go beyond 2000.
> https://github.com/pgvector/pgvector?tab=readme-ov-file#what-if-i-want-to-index-vectors-with-more-than-2000-dimensions
> The best bet for up to 4000 would be to use half-precision INDEXING

## Startup

* Log current index size and available memory:
    * SHOW config_file;
    * SHOW shared_buffers;
    * SELECT pg_size_pretty(pg_relation_size('index_name'));
    * https://github.com/pgvector/pgvector?tab=readme-ov-file#do-indexes-need-to-fit-into-memory


## Querying

Support setting this via query param: https://github.com/pgvector/pgvector?tab=readme-ov-file#query-options


## Misc

Consider exposing indexing progress somehow?
https://github.com/pgvector/pgvector?tab=readme-ov-file#indexing-progress

```sql
SELECT phase, round(100.0 * blocks_done / nullif(blocks_total, 0), 1) AS "%" FROM pg_stat_progress_create_index;
```

## Install with Helm

Right now in order to make the gen3_embeddings service work in your local deployment, the pgvector Postgres image is used for all gen3 services. (pending fix: [ticket](https://ctds-planx.atlassian.net/browse/GPE-2406?visitedUserSeg=true))

Prepare a values.yaml file with at least the following configuration:
```yaml
postgresql:
  image:
    registry: docker.io
    repository: pgvector/pgvector
    tag: pg18-trixie
    pullPolicy: IfNotPresent
  primary:
    containerSecurityContext:
      runAsUser: 0

gen3-embeddings:
  enabled: true
  debug: true
  image:
    repository: quay.io/cdis/gen3_embeddings
    pullPolicy: Always
    tag: feat_embedding
```

Download Helm charts and install the services
```bash
git clone --branch feat/add-gen3-embeddings --single-branch https://github.com/uc-cdis/gen3-helm.git

cd helm

helm dependency update ./gen3

helm install gen3-test ./gen3 -f /PTAH_TO_YOUR/values-gen3-embeddings.yaml
```
Now you can try (change the hostname):
```bash
curl -X GET "https://markx.dev.planx-pla.net/ai/vectorstore/collections/" -H "Authorization: Bearer $TOKEN"
```

If you got errors like this one on your local Helm:
```
fastapi.exceptions.HTTPException: 403: Cannot fetch pubkey from issuer https://markx.dev.planx-pla.net/user: All connection attempts failed
```
Follow [this](https://ctds-planx.atlassian.net/wiki/spaces/PD/pages/3755474945/Setting+up+a+local+dev+env+using+Helm#Ingress%3A) to fix it.

## Running and testing locally

### Create a pgvector database, create app db user, load test datasets

Run this under the gen3_embeddings folder to access the migration file
```bash
docker run --name pgvector \
    -e POSTGRES_USER=adminuser \
    -e POSTGRES_PASSWORD=adminpass \
    -e POSTGRES_DB=gen3embeddings \
    -p 5432:5432 \
    -v $(pwd)/db_migrations/0/0.sql:/docker-entrypoint-initdb.d/0.sql \
    -d pgvector/pgvector:pg18-trixie
```

Create an app user with limited permissions. A superuser can bypass RLS, and the app won't allow a superuser. for example create an app user and load some test data:

```bash
PGPASSWORD=adminpass psql -h localhost -p 5432 -U adminuser -d gen3embeddings
```
```sql
-- Create an app_user with limited permissions. A superuser can bypass RLS.
CREATE ROLE embeddings_user
  LOGIN
  PASSWORD 'embeddings_pass'
  NOSUPERUSER
  NOCREATEDB
  NOCREATEROLE
  NOINHERIT;

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE collections        TO embeddings_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE embeddings_vector  TO embeddings_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE embeddings_halfvec TO embeddings_user;
```
```sql
-- For the following, make sure you update the collection_id according to the ids from creating indices outputs
INSERT INTO collections (collection_name, description, ai_model_name, dimensions, vector_type)
VALUES
  ('noaccess', 'noaccess collection', 'test-model', 3, 'vector'),
  ('public',   'public collection', 'test-model', 3, 'vector'),
  ('internal',   'internal collection', 'test-model', 3, 'vector'),
  ('d3vector', 'd3 vector collection', 'test-model', 3, 'vector'),
  ('d200vector',    'd200 vector collection', 'test-model', 200, 'vector'),
  ('d3000halfvec',  'd5 halfvec collection', 'test-model', 3000, 'halfvec'),
  ('d3halfvec', 'd3 halfvec collection', 'test-model', 3, 'halfvec')
RETURNING *;

INSERT INTO embeddings_vector (collection_id, embedding, authz, metadata)
SELECT
    1 AS collection_id,  -- noaccess
    FORMAT('[%s,%s,%s]', g.i, g.i + 1, g.i + 2)::vector AS embedding,
    ARRAY['/vectorstore/collections/noaccess']::text[]  AS authz,
    jsonb_build_object(
        'name', FORMAT('no_access_%s', g.i),
        'i', g.i
    ) AS metadata
FROM generate_series(1, 10000) AS g(i);

INSERT INTO embeddings_vector (collection_id, embedding, authz, metadata)
SELECT
    2 AS collection_id,  -- public
    FORMAT('[%s,%s,%s]', g.i, g.i + 10, g.i + 20)::vector AS embedding,
    ARRAY['/vectorstore/collections/public']::text[]  AS authz,
    jsonb_build_object(
        'name', FORMAT('public_%s', g.i),
        'i', g.i
    ) AS metadata
FROM generate_series(1, 5000) AS g(i);

INSERT INTO embeddings_vector (collection_id, embedding, authz, metadata)
SELECT
    3 AS collection_id,  -- internal
    FORMAT('[%s,%s,%s]', g.i * 2, g.i * 2 + 1, g.i * 2 + 2)::vector AS embedding,
    ARRAY['/vectorstore/collections/internal']::text[]  AS authz,
    jsonb_build_object(
        'name', FORMAT('internal_%s', g.i),
        'i', g.i
    ) AS metadata
FROM generate_series(1, 3000) AS g(i);

INSERT INTO embeddings_vector (collection_id, embedding, authz, metadata)
SELECT
    4 AS collection_id,  -- d3vector
    FORMAT('[%s,%s,%s]', g.i, g.i * 3, g.i * 5)::vector AS embedding,
    ARRAY['/vectorstore/collections/d3vector']::text[] AS authz,
    jsonb_build_object(
        'name', FORMAT('d3vector_%s', g.i),
        'i', g.i
    ) AS metadata
FROM generate_series(1, 2000) AS g(i);

INSERT INTO embeddings_vector (collection_id, embedding, authz, metadata)
SELECT
    5 AS collection_id,  -- d200vector
    (
        '[' ||
        string_agg((g.i + offs)::text, ',' ORDER BY offs)
        || ']'
    )::vector AS embedding,
    ARRAY['/vectorstore/collections/d200vector']::text[] AS authz,
    jsonb_build_object(
        'name', FORMAT('d200vector_%s', g.i),
        'i', g.i
    ) AS metadata
FROM generate_series(1, 500) AS g(i)
CROSS JOIN LATERAL generate_series(0, 199) AS offs
GROUP BY g.i;

INSERT INTO embeddings_halfvec (collection_id, embedding, authz, metadata)
SELECT
    6 AS collection_id,  -- d3000halfvec
    (
        '[' ||
        string_agg(
            (g.i::float8 / (offs + 1))::text,
            ',' ORDER BY offs
        ) ||
        ']'
    )::halfvec AS embedding,
    ARRAY['/vectorstore/collections/d3000halfvec']::text[] AS authz,
    jsonb_build_object(
        'name', FORMAT('d3000halfvec_%s', g.i),
        'i', g.i
    ) AS metadata
FROM generate_series(1, 50) AS g(i)
CROSS JOIN LATERAL generate_series(0, 2999) AS offs
GROUP BY g.i;

INSERT INTO embeddings_halfvec (collection_id, embedding, authz, metadata)
SELECT
    7 AS collection_id,  -- d3halfvec
    FORMAT('[%s.%s,%s.%s,%s.%s]',
           g.i, 1,
           g.i + 1, 2,
           g.i + 2, 3)::halfvec AS embedding,
    ARRAY['/vectorstore/collections/d3halfvec']::text[] AS authz,
    jsonb_build_object(
        'name', FORMAT('d3halfvec_%s', g.i),
        'i', g.i
    ) AS metadata
FROM generate_series(1, 1000) AS g(i);


```

### Authz and prepare Arborist server
Launch Arborist server using gen3-helm, you can use the following example values.yaml file, update it accordingly. After launch run `kubectl port-forward svc/arborist-service -n default 4280:80`

```yaml
global:
  hostname: markx.dev.planx-pla.net
  dev: true
  tls:
    cert:
      CERT
    key: |
      KEY
fence:
  FENCE_CONFIG:
    MOCK_AUTH: true
  USER_YAML: |
    cloud_providers: {}
    authz:
      # policies automatically given to anyone, even if they are not authenticated
      anonymous_policies:
      - open_data_reader

      # policies automatically given to authenticated users (in addition to their other policies)
      all_users_policies: []

      groups:
      # can CRUD programs and projects and upload data files
      - name: data_submitters
        policies:
        - services.sheepdog-admin
        - data_upload
        - MyFirstProject_submitter
        users:
        - username1@gmail.com
        - test

      # can create/update/delete indexd records
      - name: indexd_admins
        policies:
        - indexd_admin
        users:
        - username1@gmail.com
        - test

      resources:
      - name: workspace
      - name: data_file
      - name: services
        subresources:
        - name: sheepdog
          subresources:
          - name: submission
            subresources:
            - name: program
            - name: project
        - name: 'indexd'
          subresources:
            - name: 'admin'
        - name: audit
          subresources:
            - name: presigned_url
            - name: login
      - name: vectorstore
        subresources:
        - name: collections
          subresources:
          - name: noaccess
          - name: public
          - name: internal
          - name: d3vector
          - name: d200vector
          - name: d3000halfvec
          - name: d3halfvec
      - name: open
      - name: programs
        subresources:
        - name: MyFirstProgram
          subresources:
          - name: projects
            subresources:
            - name: MyFirstProject

      policies:
      - id: services.gen3-embeddings-admin
        description: CRUD access to embeddings
        role_ids:
        - gen3-embeddings-admin
        resource_paths:
        - /vectorstore/collections/noaccess
        - /vectorstore/collections/public
        - /vectorstore/collections/internal
        - /vectorstore/collections/d3vector
        - /vectorstore/collections/d200vector
        - /vectorstore/collections/d3000halfvec
        - /vectorstore/collections/d3halfvec
      - id: services.gen3-embeddings-user
        description: CRUD access to embeddings
        role_ids:
        - gen3-embeddings-user
        resource_paths:
        - /vectorstore/collections/public
        - /vectorstore/collections/internal
        - /vectorstore/collections/d3vector
        - /vectorstore/collections/d200vector
        - /vectorstore/collections/d3000halfvec
        - /vectorstore/collections/d3halfvec
      - id: workspace
        description: be able to use workspace
        resource_paths:
        - /workspace
        role_ids:
        - workspace_user
      - id: data_upload
        description: upload raw data files to S3
        role_ids:
        - file_uploader
        resource_paths:
        - /data_file
      - id: services.sheepdog-admin
        description: CRUD access to programs and projects
        role_ids:
          - sheepdog_admin
        resource_paths:
          - /services/sheepdog/submission/program
          - /services/sheepdog/submission/project
      - id: indexd_admin
        description: full access to indexd API
        role_ids:
          - indexd_admin
        resource_paths:
          - /programs
      - id: open_data_reader
        role_ids:
          - peregrine_reader
          - guppy_reader
          - fence_storage_reader
        resource_paths:
        - /open
      - id: all_programs_reader
        role_ids:
        - peregrine_reader
        - guppy_reader
        - fence_storage_reader
        resource_paths:
        - /programs
      - id: MyFirstProject_submitter
        role_ids:
        - reader
        - creator
        - updater
        - deleter
        - storage_reader
        - storage_writer
        resource_paths:
        - /programs/MyFirstProgram/projects/MyFirstProject

      roles:
      - id: 'gen3-embeddings-user'
        description: ''
        permissions:
        - id: 'embeddings_reader'
          action:
            method: read
            service: 'gen3-embeddings'
        - id: 'embeddings_creator'
          action:
            method: create
            service: 'gen3-embeddings'
        - id: 'embeddings_updater'
          action:
            method: update
            service: 'gen3-embeddings'
        - id: 'embeddings_deleter'
          action:
            method: delete
            service: 'gen3-embeddings'
      - id: 'gen3-embeddings-admin'
        description: ''
        permissions:
        - id: 'embeddings_reader'
          action:
            method: read
            service: 'gen3-embeddings'
        - id: 'embeddings_creator'
          action:
            method: create
            service: 'gen3-embeddings'
        - id: 'embeddings_updater'
          action:
            method: update
            service: 'gen3-embeddings'
        - id: 'embeddings_deleter'
          action:
            method: delete
            service: 'gen3-embeddings'
      - id: file_uploader
        permissions:
        - id: file_upload
          action:
            service: fence
            method: file_upload
      - id: workspace_user
        permissions:
        - id: workspace_access
          action:
            service: jupyterhub
            method: access
      - id: sheepdog_admin
        description: CRUD access to programs and projects
        permissions:
        - id: sheepdog_admin_action
          action:
            service: sheepdog
            method: '*'
      - id: indexd_admin
        description: full access to indexd API
        permissions:
        - id: indexd_admin
          action:
            service: indexd
            method: '*'
      - id: admin
        permissions:
          - id: admin
            action:
              service: '*'
              method: '*'
      - id: creator
        permissions:
          - id: creator
            action:
              service: '*'
              method: create
      - id: reader
        permissions:
          - id: reader
            action:
              service: '*'
              method: read
      - id: updater
        permissions:
          - id: updater
            action:
              service: '*'
              method: update
      - id: deleter
        permissions:
          - id: deleter
            action:
              service: '*'
              method: delete
      - id: storage_writer
        permissions:
          - id: storage_creator
            action:
              service: '*'
              method: write-storage
      - id: storage_reader
        permissions:
          - id: storage_reader
            action:
              service: '*'
              method: read-storage
      - id: peregrine_reader
        permissions:
        - id: peregrine_reader
          action:
            method: read
            service: peregrine
      - id: guppy_reader
        permissions:
        - id: guppy_reader
          action:
            method: read
            service: guppy
      - id: fence_storage_reader
        permissions:
        - id: fence_storage_reader
          action:
            method: read-storage
            service: fence

    clients:
      wts:
        policies:
        - all_programs_reader
        - open_data_reader

    users:
      username1@gmail.com: {}
      username2:
        tags:
          name: John Doe
          email: johndoe@gmail.com
        policies:
        - MyFirstProject_submitter
      test:
        policies:
        - workspace
        - services.gen3-embeddings-user

    cloud_providers: {}
    groups: {}

postgresql:
  primary:
    persistence:
      # -- (bool) Option to persist the dbs data.
      enabled: false

# Use a prebuilt portal image if you're deploying to a laptop, less resources consumed by gen3
portal:
  resources:
    requests:
      cpu: "0.2"
      memory: 100Mi
  image:
    repository: quay.io/cdis/data-portal-prebuilt
    tag: dev
```

Create `.env` file under gen3_embeddings folder
```bash
PGHOST=localhost
PGPORT=5432
PGUSER=embeddings_user
PGPASSWORD=embeddings_pass
PGDATABASE=gen3embeddings

DEBUG=True
ARBORIST_URL="http://localhost:4280"
VERBOSE_INTERNAL_LOGS=True
```

### Start gen3_embeddings server
run this under gen3-ai folder
```bash
uv run --directory "./services/gen3_embeddings" \
  gunicorn \
  gen3_embeddings.main:app_instance \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:4142 \
  --access-logfile - \
  --error-logfile -
  ```

### Run tests

```bash
uv run pytest -n auto . -vv

uv pip install pytest-cov

uv run pytest -n auto . -vv \
  --cov=gen3_embeddings \
  --cov-report=term-missing \
  --cov-report=html

uv run pytest -n auto . --maxfail=1 --disable-warnings \
  --cov=gen3_embeddings.database.db \
  --cov=gen3_embeddings.database.helpers \
  --cov-report=term-missing
```

### Sample manual tests
```bash
export TOKEN=...
curl -X GET "http://localhost:4142/vectorstore/collections/public/embeddings" -H "Authorization: Bearer $TOKEN"

curl -X GET "http://localhost:4142/vectorstore/collections/public/embeddings?page=2&page_size=200" -H "Authorization: Bearer $TOKEN"

curl -X GET "http://localhost:4142/vectorstore/collections/public/embeddings?no_embeddings_info=true" -H "Authorization: Bearer $TOKEN"

curl -X GET "http://localhost:4142/vectorstore/collections/public/embeddings/e3c5cfe0-20f8-4270-8c3d-30e73adbe83c" -H "Authorization: Bearer $TOKEN"

curl -X DELETE "http://localhost:4142/vectorstore/collections/internal" -H "Authorization: Bearer $TOKEN"

curl -X POST "http://localhost:4142/vectorstore/collections" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "collection_name": "internal",
    "description": "internal index",
    "dimensions": 3,
    "vector_type": "halfvec"
  }'

curl -X GET "http://localhost:4142/vectorstore/collections" -H "Authorization: Bearer $TOKEN"

curl -X GET "http://localhost:4142/vectorstore/collections/internal" -H "Authorization: Bearer $TOKEN"

curl -X PATCH "http://localhost:4142/vectorstore/collections/internal" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"description": "Updated description"}'

curl -X POST "http://localhost:4142/vectorstore/collections/internal/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "embeddings": [
      {
        "embedding": [0.1, 0.2, 0.3],
        "metadata": {
          "source": "some_file.md",
          "chunk_size": "1000"
        }
      },
      {
        "embedding": [0.4, 0.2, 0.3],
        "metadata": {
          "source": "some_file1.md",
          "chunk_size": "10001"
        }
      }
    ]
  }'

curl -X POST "http://localhost:4142/vectorstore/collections/internal/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "authz": ["/vectorstore/collections/d3halfvec", "/vectorstore/collections/d200vector"],
    "embeddings": [
      {
        "embedding": [0.1, 0.2, 0.3],
        "metadata": {
          "source": "some_file.md",
          "chunk_size": "1000"
        }
      },
      {
        "embedding": [0.4, 0.2, 0.3],
        "metadata": {
          "source": "some_file1.md",
          "chunk_size": "10001"
        }
      }
    ]
  }'

curl -X PUT "http://localhost:4142/vectorstore/collections/internal/embeddings/07fc788b-4f54-478f-a821-1c4235a8369e" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "authz": ["/vectorstore/collections/d3halfvec", "/vectorstore/collections/internal"]
  }'


curl -X PUT "http://localhost:4142/vectorstore/collections/internal/embeddings/44618c99-fd7a-4fe3-8056-309f40d8bbc4" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "embedding": [0.5, 0.5, 0.5]
  }'

curl -X PUT "http://localhost:4142/vectorstore/collections/internal/embeddings/44618c99-fd7a-4fe3-8056-309f40d8bbc4" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "metadata": {
      "source": "some_file1.md",
      "chunk_size": "10001",
      "new_tag": "tag"
    }
  }'

curl -X PUT "http://localhost:4142/vectorstore/collections/internal/embeddings/44618c99-fd7a-4fe3-8056-309f40d8bbc4" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "embedding": [1, 0.5, 0.5],
    "metadata": {
      "source": "some_file1.md",
      "chunk_size": "10001",
      "new_tag": "tag"
    }
  }'

curl -X DELETE "http://localhost:4142/vectorstore/collections/internal/embeddings/44618c99-fd7a-4fe3-8056-309f40d8bbc4" -H "Authorization: Bearer $TOKEN"

curl -X POST "http://localhost:4142/embeddings/bulk" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '[
    "2368bc06-5cec-4c40-a41e-c84aee3216d9",
    "d4ade0e7-4194-4cb4-9a59-dcf586d35283",
    "2edd2b42-f072-4b43-a5e9-47d1a31866e5"
  ]'

curl -X POST "http://localhost:4142/embeddings/bulk?no_embeddings_info=true" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '[
    "2368bc06-5cec-4c40-a41e-c84aee3216d9",
    "d4ade0e7-4194-4cb4-9a59-dcf586d35283",
    "2edd2b42-f072-4b43-a5e9-47d1a31866e5"
  ]'

curl -X POST "http://localhost:4142/vectorstore/collections/d200vector/embeddings/bulk" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '[
    "2368bc06-5cec-4c40-a41e-c84aee3216d9",
    "d4ade0e7-4194-4cb4-9a59-dcf586d35283",
    "2edd2b42-f072-4b43-a5e9-47d1a31866e5"
  ]'

curl -X POST "http://localhost:4142/vectorstore/collections/d3halfvec/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 2,
    "filters": null,
    "distance_metric": "l2_distance"
  }'

curl -X POST "http://localhost:4142/vectorstore/collections/d3halfvec/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 2,
    "filters": null,
    "distance_metric": "l2_distance",
    "min_value": 4,
    "max_value": 10
  }'

curl -X POST "http://localhost:4142/vectorstore/collections/d3halfvec/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 2,
    "filters": null,
    "distance_metric": "l2_distance",
    "min_value": 10,
    "max_value": 0
  }'


curl -X POST "http://localhost:4142/vectorstore/collections/d3halfvec/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 2,
    "filters": null
  }'

curl -X POST "http://localhost:4142/vectorstore/collections/d3halfvec/search?no_embeddings_info=true" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 2
  }'

curl -X POST "http://localhost:4142/vectorstore/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 5,
    "filters": null
  }'

curl -X POST "http://localhost:4142/vectorstore/search?vector_type=halfvec" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 5,
    "filters": null
  }'

curl -X POST "http://localhost:4142/vectorstore/search?collections=public,d3vector" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 5
  }'
```

## TODO
- ai model
- don't print out detailed errors at client side
- support DEBUG_SKIP_AUTH True for RLS
- sanitize collection name
- add .info logs for embedding reads (e.g. any time someone is auth-ed and successfully reads data, we need an info log saying what user read what data - can just be embedding IDs)
- add support for index
- set app user
- output page_size value use the actual output size or the defined page size?
- delete functions need some work
- the unauthorized error got 500
- move get_allowed_authz_for_request logic out of db.py
- Creating a duplicate collections should 409 instead of 400 imo. Right now if I try to recreate the same one I get a 400
- when attempting to create a collection that I do not have the authorization to create, I would expect a 401. right now getting a 400 bad request
- The type used is in the response, so if the user doesn't supply vector_type, we should default based on the dimensions, if they provide vector_type=vector AND try to exceed, then error, yes. but if they don't provide, we should just pick vector or halfvec based on the dimensions size
- no duplcicate embeddings in the same collection
