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
