"""
Content hashes that define when two embeddings are "the same".

Every embedding row carries a hash of its vector and a hash of its metadata. The unique
constraint on (collection_id, embedding_hash, metadata_hash, authz) is the only definition
of duplicate the service has: it is what makes a repeated POST fail as a conflict and what
PUT conflicts on to turn an insert into an update. Two properties matter.

1. The digest is computed over caller-controlled bytes, so it needs collision resistance
   against a caller who is trying. The original implementation used md5, for which
   chosen-prefix collisions are cheap; two crafted payloads that hash alike collapse into
   one row, so a POST of genuinely new content can be rejected as a duplicate and a PUT can
   overwrite an unrelated row. sha256 (truncated to the 128 bits a uuid column holds)
   removes that. It also keeps md5 out of the image for FIPS-mode and scanner purposes.

2. The digest is computed over what Postgres STORES, not over what the caller sent. pgvector
   stores `vector` as float32 and `halfvec` as float16, so inputs that differ only below the
   storage precision land on byte-identical stored vectors. Hashing the caller's
   full-precision JSON text let those in as separate rows that the constraint could not
   distinguish -- most visibly on halfvec collections, where float16 has ~3 decimal digits.

Hashing here (rather than as `md5(...)` inside the INSERT) also means the vector never has
to be serialized to text for the database to hash it, which is what makes the binary
float4[] write path in `db.py` possible.
"""

import hashlib
import json
from uuid import UUID

import numpy as np

from gen3_embeddings.database.errors import (
    EmbeddingDimensionMismatchError,
    EmbeddingNotRepresentableError,
)
from gen3_embeddings.models.schemas import VectorPrecision, VectorType

# `embedding_hash`/`metadata_hash` are uuid columns and hold 128 bits; sha256 produces 256.
# Truncating a sha256 digest to its leading half is the standard construction (sha256/128)
# and leaves a ~2^64 birthday bound, far above any row count these tables will reach.
DIGEST_BYTES = 16

# Byte order is pinned explicitly: `np.float32` is native-endian, which would make a hash
# computed on a big-endian host disagree with one computed on the x86/arm hosts that wrote
# the rest of the table.
#
# Keyed on precision rather than vector type because precision is what decides the byte
# layout, and because the binary read endpoints have a precision rather than a collection to
# hand. That makes this the single place the wire byte order of a vector is decided, for both
# the hashes stored in the table and the bytes handed to clients.
_STORAGE_DTYPE: dict[VectorPrecision, str] = {
    VectorPrecision.float32: "<f4",
    VectorPrecision.float16: "<f2",
}


def storage_dtype_for_precision(precision: VectorPrecision) -> np.dtype:
    """
    Return the numpy dtype for vectors stored at this precision.

    Args:
        precision (VectorPrecision): Floating-point precision of the stored vectors.

    Returns:
        np.dtype: Little-endian float32 for `float32`, little-endian float16 for `float16`.

    Raises:
        ValueError: If the precision has no known storage dtype.
    """
    try:
        return np.dtype(_STORAGE_DTYPE[precision])
    except KeyError:
        raise ValueError(f"Unsupported vector precision: {precision}") from None


def storage_dtype(vector_type: VectorType) -> np.dtype:
    """
    Return the numpy dtype matching how pgvector stores this vector type on disk.

    Args:
        vector_type (VectorType): Storage type of the target collection.

    Returns:
        np.dtype: Little-endian float32 for `vector`, little-endian float16 for `halfvec`.

    Raises:
        ValueError: If the vector type has no known storage dtype.
    """
    # pgvector `vector` stores float32, `halfvec` stores float16.
    return storage_dtype_for_precision(vector_type.precision)


def to_storage_array(
    vectors: list[list[float]],
    vector_type: VectorType,
    dimensions: int,
) -> np.ndarray:
    """
    Convert a batch of vectors into one (n, dimensions) array at storage precision.

    The array is the input to both the row hashes and the flat float4[] the bulk INSERT
    binds, so it is built once per request.

    Args:
        vectors (list[list[float]]): Vectors to convert; all must have `dimensions` elements.
        vector_type (VectorType): Storage type of the target collection.
        dimensions (int): Dimensionality the collection declares.

    Returns:
        np.ndarray: C-contiguous array of shape (len(vectors), dimensions), whose rows are
        byte-for-byte what Postgres will store.

    Raises:
        EmbeddingDimensionMismatchError: If any vector's length is not `dimensions`. The
            route layer already checks this per request, but the bulk INSERT binds these
            vectors as one flat array sliced by `dimensions`, so a wrong length here would
            silently shift every following row rather than fail. This check is what makes
            that flattening safe.
        EmbeddingNotRepresentableError: If a value overflows the storage type (only reachable
            for halfvec, whose float16 range stops at ~65504).
    """
    dtype = storage_dtype(vector_type)

    for index, vector in enumerate(vectors):
        if len(vector) != dimensions:
            raise EmbeddingDimensionMismatchError(
                f"Embedding at index {index} has {len(vector)} dimensions, expected {dimensions} for this collection"
            )

    # float64 -> float16 can overflow to inf; ignore the warning and report the offending
    # row ourselves, which is clearer than the Postgres cast error and catches it before
    # the round trip.
    with np.errstate(over="ignore"):
        array = np.asarray(vectors, dtype=dtype)

    if not np.isfinite(array).all():
        rows = np.flatnonzero(~np.isfinite(array).all(axis=1))
        raise EmbeddingNotRepresentableError(
            f"Embedding at index {int(rows[0])} has a value outside the range storable as {vector_type.value}"
        )

    return np.ascontiguousarray(array)


def hash_rows(array: np.ndarray) -> list[UUID]:
    """
    Hash each row of a storage-precision array from `to_storage_array`.

    Args:
        array (np.ndarray): C-contiguous (n, dimensions) array at storage precision.

    Returns:
        list[UUID]: One hash per row, in row order.
    """
    return [UUID(bytes=hashlib.sha256(row.tobytes()).digest()[:DIGEST_BYTES]) for row in array]


def flatten_rows(array: np.ndarray, row_indices: list[int]) -> list[float]:
    """
    Flatten the selected rows into the row-major float list bound as the INSERT's float4[].

    The bulk INSERT binds one flat array for the whole batch and slices out row `i` with
    `arr[((i - 1) * dimensions + 1):(i * dimensions)]`, which is what keeps vectors off the
    text-serialization path: asyncpg encodes float4[] in binary, so no float ever gets
    formatted as a string.

    Args:
        array (np.ndarray): Storage-precision array from `to_storage_array`.
        row_indices (list[int]): Ascending indices of the rows to keep, after deduplication.

    Returns:
        list[float]: Concatenated rows, in `row_indices` order.
    """
    if len(row_indices) == len(array):
        # every row survived deduplication, so skip the fancy-index copy
        selected = array
    else:
        selected = array[row_indices]
    return selected.reshape(-1).tolist()


def hash_vector(vector: list[float], vector_type: VectorType, dimensions: int) -> UUID:
    """
    Hash a single vector the same way `hash_rows` hashes a batch.

    Args:
        vector (list[float]): The vector to hash.
        vector_type (VectorType): Storage type of the target collection.
        dimensions (int): Dimensionality the collection declares.

    Returns:
        UUID: The vector's content hash.

    Raises:
        EmbeddingDimensionMismatchError: If the vector's length is not `dimensions`.
        EmbeddingNotRepresentableError: If a value overflows the storage type.
    """
    return hash_rows(to_storage_array([vector], vector_type, dimensions))[0]


def canonical_metadata_json(metadata: dict | None) -> str:
    """
    Render metadata as the canonical JSON text that its hash is taken over.

    Sorted keys and no whitespace mean two dicts that differ only in key order or in how the
    caller formatted their JSON hash alike. This is also the text bound to the INSERT, so the
    hash and the stored jsonb always come from the same bytes. It replaces the previous
    `md5(metadata::text)`, which relied on Postgres's jsonb text rendering staying stable
    across server versions.

    Args:
        metadata (dict | None): Metadata to render; None is treated as an empty object, the
            same way the write paths treat a missing metadata field.

    Returns:
        str: Canonical JSON text.
    """
    return json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))


def hash_metadata_json(metadata_json: str) -> UUID:
    """
    Hash already-canonicalized metadata text from `canonical_metadata_json`.

    Args:
        metadata_json (str): Canonical JSON text.

    Returns:
        UUID: The metadata's content hash.
    """
    return UUID(bytes=hashlib.sha256(metadata_json.encode("utf-8")).digest()[:DIGEST_BYTES])


def hash_metadata(metadata: dict | None) -> UUID:
    """
    Hash a metadata dict.

    Args:
        metadata (dict | None): Metadata to hash; None is treated as an empty object.

    Returns:
        UUID: The metadata's content hash.
    """
    return hash_metadata_json(canonical_metadata_json(metadata))
