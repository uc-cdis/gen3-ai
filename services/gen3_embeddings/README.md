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

> If you need to support >2000 dimension vectors: the service will need to be modified. This is a limitation of the pgvector `vector` column. However, if you need to create different indexes, you can do so.

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


## Running and testing locally

### Create a pgvector database, create app db user, load test datasets

```bash
docker run --name pgvector \
    -e POSTGRES_USER=testuser \
    -e POSTGRES_PASSWORD=testpass \
    -e POSTGRES_DB=testdb \
    -p 5432:5432 \
    -v $(pwd)/db_migrations/0/0.sql:/docker-entrypoint-initdb.d/0.sql \
    -d pgvector/pgvector:pg18-trixie


PGPASSWORD=testpass psql -h localhost -p 5432 -U testuser -d testdb
```
For the following, make sure you update the vector_index_id according to the ids from creating indices outputs
```sql
CREATE ROLE app_user
  LOGIN
  PASSWORD 'app_user_pass'
  NOSUPERUSER
  NOCREATEDB
  NOCREATEROLE
  NOINHERIT;

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE embeddings TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE vector_indices TO app_user;


INSERT INTO vector_indices (vector_index_name, description, ai_model_name, dimensions)
VALUES
  ('noaccess', 'RLS test index', 'test-model', 3),
  ('public', 'RLS test index', 'test-model', 3),
  ('team42', 'RLS test index', 'test-model', 3),
  ('internal', 'RLS test index', 'test-model', 3),
  ('team7', 'RLS test index', 'test-model', 3)
RETURNING id;


INSERT INTO embeddings (vector_index_id, embedding, authz_version, authz, metadata)
VALUES
  (1, '[4,5,6]'::vector, 0, ARRAY['/vector/indices/noaccess'],                  '{"name": "no_access"}'),
  (2, '[1,2,3]'::vector, 0, ARRAY['/vector/indices/public'],                  '{"name": "public_only"}'),
  (3, '[2,3,4]'::vector, 0, ARRAY['/vector/indices/team42', '/vector/indices/internal'],     '{"name": "team_42_internal"}'),
  (5, '[5,4,5]'::vector, 0, ARRAY['/vector/indices/team7'],                  '{"name": "team_7_only"}'),
  (5, '[3,4,5]'::vector, 0, ARRAY['/vector/indices/team7'],                  '{"name": "team_7_only"}');
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
      - name: vector
        subresources:
        - name: indices
          subresources:
          - name: public
          - name: internal
          - name: team42
          - name: team7
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
        - /vector/indices/public
        - /vector/indices/team42
        - /vector/indices/internal
        - /vector/indices/team7
      - id: services.gen3-embeddings-user
        description: CRUD access to embeddings
        role_ids:
        - gen3-embeddings-user
        resource_paths:
        - /vector/indices/public
        - /vector/indices/team7
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
        - services.gen3-embeddings-admin

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

### Start gen3_embeddings server
Create `.env` file under gen3_embeddings folder
```bash
DB_HOST=localhost
DB_PORT=5432
DB_USER=app_user
DB_PASSWORD=app_user_pass
DB_DATABASE=testdb
DEBUG=True
ARBORIST_URL="http://localhost:4280"
```

run `just run gen3_embeddings` under gen3-ai folder

### Sample tests
```bash
export TOKEN=...
curl -X GET "http://localhost:4142/vector/indices/team7/embeddings" -H "Authorization: Bearer $TOKEN"

curl -X GET "http://localhost:4142/vector/indices/team7/embeddings?no_embeddings_info=true" -H "Authorization: Bearer $TOKEN"

curl -X GET "http://localhost:4142/vector/indices/team7/embeddings/f9090a1c-7a74-4253-a3fe-ef930b613caa" -H "Authorization: Bearer $TOKEN"

curl -X GET "http://localhost:4142/embeddings/dc4696bf-0aa3-42a5-b0ac-84b6ebf74c65" -H "Authorization: Bearer $TOKEN"

curl -X DELETE "http://localhost:4142/vector/indices/internal" -H "Authorization: Bearer $TOKEN"

curl -X POST "http://localhost:4142/vector/indices" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "index_name": "internal",
    "description": "internal index",
    "dimensions": 3
  }'

curl -X GET "http://localhost:4142/vector/indices" -H "Authorization: Bearer $TOKEN"

curl -X GET "http://localhost:4142/vector/indices/team42" -H "Authorization: Bearer $TOKEN"

curl -X PATCH "http://localhost:4142/vector/indices/team42" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"description": "Updated description"}'

curl -X POST "http://localhost:4142/vector/indices/team42/embeddings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '[
    [0.1, 0.2, 0.3],
    [0.9, 0.8, 0.7]
  ]'


curl -X GET "http://localhost:4142/embeddings/22761de5-6b0f-4bb3-acc6-4cc5a5db6de9" -H "Authorization: Bearer $TOKEN"

curl -X PUT "http://localhost:4142/vector/indices/team42/embeddings/830e2d61-2205-4207-9b83-a0341013623e" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "embedding": [0.5, 0.5, 0.5]
  }'

curl -X DELETE "http://localhost:4142/vector/indices/team42/embeddings/830e2d61-2205-4207-9b83-a0341013623e" -H "Authorization: Bearer $TOKEN"

curl -X POST "http://localhost:4142/embeddings/bulk" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '[
    "da1bcbcd-1164-4fb2-a613-1351521d96cf",
    "dc4696bf-0aa3-42a5-b0ac-84b6ebf74c65"
  ]'

curl -X POST "http://localhost:4142/embeddings/bulk?no_embeddings_info=true" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '[
    "da1bcbcd-1164-4fb2-a613-1351521d96cf",
    "dc4696bf-0aa3-42a5-b0ac-84b6ebf74c65"
  ]'

curl -X POST "http://localhost:4142/vector/indices/team7/embeddings/bulk" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '[
    "da1bcbcd-1164-4fb2-a613-1351521d96cf",
    "dc4696bf-0aa3-42a5-b0ac-84b6ebf74c65"
  ]'

curl -X POST "http://localhost:4142/vector/indices/team7/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 2,
    "filters": null
  }'

curl -X POST "http://localhost:4142/vector/indices/team7/search?no_embeddings_info=true" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 2
  }'

curl -X POST "http://localhost:4142/vector/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 5,
    "filters": null
  }'

curl -X POST "http://localhost:4142/vector/search?vector_indices=team7" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.0, 0.0, 0.0],
    "top_k": 5
  }'
```

## TODO
- db op error handling: duplication when creating, search with diff dim
- ai model
- diff dim between indices in searching
- table init need this?: ALTER TABLE accounts FORCE ROW LEVEL SECURITY;
- don't print out detailed errors at client side
