# How vectors move through the service

A vector passes through three representations between the HTTP request and the stored row,
and each layer uses the one that suits its job. This page describes which type is used
where, why, and which code owns each step.

| Layer | Type | Owned by |
|---|---|---|
| Web (request and response bodies) | Python `list[float]` | FastAPI / pydantic |
| Code (validation, hashing, deduplication) | numpy array, little-endian, at storage precision | `database/hashing.py` |
| Database (binding parameters, decoding rows) | pgvector `Vector` / `HalfVector` | pgvector-python's asyncpg codec |

## Web layer: Python floats

Request bodies are JSON, so the framework hands us `list[list[float]]`. Nothing at this layer
cares about precision or byte order. The routes validate shape and pass the lists to the data
access layer.

## Code layer: little-endian numpy at storage precision

`hashing.to_storage_array` converts a batch into one `(n, dimensions)` numpy array **once per
request**. That array then feeds both the content hashes and the values that get written, so
the two cannot disagree.

**Storage precision.** The array uses the precision of the target column: float32 for
`vector`, float16 for `halfvec`. Postgres stores at that precision, so two inputs that differ
only below it are the same stored vector, and they need to hash alike. Converting here also
catches values that overflow float16 (above ~65504) before anything is sent.

**Little-endian, pinned explicitly.** The dtypes are `<f4` and `<f2` (`_STORAGE_DTYPE` in
`hashing.py`), not native `np.float32`/`np.float16`.

- Hashes are taken over the raw bytes of each row (`hashing.hash_rows`), so those bytes have
  to be the same on every host, or a hash computed on one machine would not match a hash
  stored by another. Pinning the order makes them stable.
- Little-endian specifically because it is the native order of the x86 and ARM hosts we run
  on, so the conversion costs nothing there.

**The byte order is part of the stored data.** `embedding_hash_v2` values already in the
table were computed over little-endian bytes. Changing `_STORAGE_DTYPE` would make new hashes
disagree with existing rows, and identical content would stop being recognized as a
duplicate. Treat it as fixed.

## Database layer: pgvector objects

`hashing.to_pgvector_rows` wraps each surviving row in a pgvector `Vector` (for `vector`
collections) or `HalfVector` (for `halfvec`). These objects are what get bound to the query.

### Bulk writes bind one array per column

`create_embeddings_bulk` and `upsert_embeddings_bulk` (`database/dal/writes.py`) write a
whole batch in one statement. Each column is one array parameter, and `unnest` zips them back
into rows:

```sql
INSERT INTO embeddings_vector (collection_id, embedding, authz, metadata, ...)
SELECT $1::bigint, raw.embedding, $3::text, raw.metadata::jsonb, ...
FROM unnest($2::vector[], $4::text[], $5::uuid[], $6::uuid[])
    AS raw(embedding, metadata, embedding_hash, metadata_hash)
```

`$2` is a `list[Vector]` (or `list[HalfVector]`, bound as `halfvec[]`), and asyncpg encodes
each element with the pgvector codec. The `::vector[]` / `::halfvec[]` placeholder comes from
`get_embeddings_table_and_cast`, which matches the collection's table.

### Single-row writes

`update_embedding` binds the caller's `list[float]` directly, and the codec converts it to a
`Vector`/`HalfVector` itself. It is hashed with `hashing.hash_vector`, which goes through the
same numpy conversion as the bulk path. pgvector's list conversion and numpy's agree bit for
bit at both precisions, including on values exactly halfway between two float16 values, so
the stored hash describes the stored vector.

## Reading vectors back

The pool's codec decodes `vector`/`halfvec` columns into `Vector`/`HalfVector` objects.

- JSON responses turn them back into Python lists.
- The binary bulk-read endpoints (`routes/embeddings_bulk.py`) return each vector as
  base64-encoded little-endian floats, labeled with its precision.
  `models.helpers.embedding_to_binary_result` gets those bytes from `to_numpy()`, then pins
  them with `storage_dtype_for_precision`. The pin is needed because `to_numpy()` returns
  native byte order. On little-endian hosts it is a no-op and the zero-copy view is kept.

The byte order clients receive and the byte order hashes are computed over therefore come
from the same table in `hashing.py`.
