"""Tests for the content hashes that define embedding uniqueness (database/hashing.py)."""

import hashlib
import json

import numpy as np
import pytest

from gen3_embeddings.database import hashing
from gen3_embeddings.database.errors import (
    EmbeddingDimensionMismatchError,
    EmbeddingNotRepresentableError,
)
from gen3_embeddings.models.schemas import VectorType


def test_embedding_hash_is_truncated_sha256_of_the_stored_bytes():
    """The hash is sha256 over the float32 bytes Postgres stores, cut to a uuid's 128 bits."""
    vector = [1.5, -2.25, 0.0]

    expected = hashlib.sha256(np.asarray(vector, dtype="<f4").tobytes()).digest()[:16]

    assert hashing.hash_vector(vector, VectorType.vector, 3).bytes == expected


def test_metadata_hash_is_truncated_sha256_of_canonical_json():
    """The metadata hash is sha256 over sorted-key, whitespace-free JSON."""
    metadata = {"b": 1, "a": [2, 3]}

    expected = hashlib.sha256(b'{"a":[2,3],"b":1}').digest()[:16]

    assert hashing.hash_metadata(metadata).bytes == expected


def test_no_md5_length_digests():
    """A full md5 digest is also 16 bytes, so assert the value is not the md5 of the old input."""
    vector = [0.1, 0.2, 0.3]

    legacy = hashlib.md5(json.dumps(vector).encode()).digest()

    assert hashing.hash_vector(vector, VectorType.vector, 3).bytes != legacy


def test_hashes_are_deterministic_across_equal_inputs():
    """Equal content hashes equally, which is what makes the unique constraint work."""
    assert hashing.hash_vector([1.0, 2.0], VectorType.vector, 2) == hashing.hash_vector(
        [1.0, 2.0], VectorType.vector, 2
    )
    assert hashing.hash_metadata({"a": 1}) == hashing.hash_metadata({"a": 1})


def test_different_content_hashes_differently():
    """Distinguishable content stays distinguishable."""
    assert hashing.hash_vector([1.0, 2.0], VectorType.vector, 2) != hashing.hash_vector(
        [2.0, 1.0], VectorType.vector, 2
    )
    assert hashing.hash_metadata({"a": 1}) != hashing.hash_metadata({"a": 2})


def test_metadata_hash_ignores_key_order():
    """Key order is not content; two dicts that differ only in order are the same metadata."""
    assert hashing.hash_metadata({"a": 1, "b": 2}) == hashing.hash_metadata({"b": 2, "a": 1})


def test_metadata_hash_treats_none_as_empty():
    """Missing metadata and empty metadata are the same row, as the write paths assume."""
    assert hashing.hash_metadata(None) == hashing.hash_metadata({})


def test_vector_hash_collapses_differences_below_float32_precision():
    """
    Two inputs that store as the same float32 vector must hash alike.

    They are the same row once written, so a hash that told them apart would let a duplicate
    past the unique constraint.
    """
    assert hashing.hash_vector([1.0, 2.0], VectorType.vector, 2) == hashing.hash_vector(
        [1.0 + 1e-12, 2.0], VectorType.vector, 2
    )


def test_vector_hash_collapses_differences_below_float16_precision_for_halfvec():
    """
    Same rule on halfvec, where float16 leaves only ~3 decimal digits.

    This is the case the old md5-of-JSON-text hash missed most easily: 1.0 and 1.0001 are
    different text, so they hashed differently, but they store as the same halfvec.
    """
    coarse = hashing.hash_vector([1.0, 2.0], VectorType.halfvec, 2)

    assert hashing.hash_vector([1.0001, 2.0], VectorType.halfvec, 2) == coarse
    # ... while a difference float16 CAN hold is still a different hash
    assert hashing.hash_vector([1.01, 2.0], VectorType.halfvec, 2) != coarse


def test_the_same_vector_hashes_differently_per_storage_type():
    """
    A vector and a halfvec collection hash the same input differently.

    They store different bytes, and each table's constraint only ever compares within itself.
    """
    assert hashing.hash_vector([0.1, 0.2], VectorType.vector, 2) != hashing.hash_vector(
        [0.1, 0.2], VectorType.halfvec, 2
    )


def test_to_storage_array_rejects_a_wrong_length_vector():
    """
    A wrong-length vector has to fail here.

    The bulk INSERT binds the batch as one flat array sliced by the collection's
    dimensionality, so a short or long row would shift every row after it instead of erroring.
    """
    with pytest.raises(EmbeddingDimensionMismatchError, match="index 1"):
        hashing.to_storage_array([[1.0, 2.0], [1.0, 2.0, 3.0]], VectorType.vector, 2)


def test_to_storage_array_rejects_values_the_storage_type_cannot_hold():
    """float16 tops out around 65504, so a larger value is reported rather than sent."""
    with pytest.raises(EmbeddingNotRepresentableError, match="index 0"):
        hashing.to_storage_array([[70000.0, 1.0]], VectorType.halfvec, 2)


def test_to_storage_array_allows_large_values_within_float32():
    """The same value is fine on a float32 collection."""
    array = hashing.to_storage_array([[70000.0, 1.0]], VectorType.vector, 2)

    assert array.shape == (1, 2)
    assert array[0][0] == pytest.approx(70000.0)


def test_flatten_rows_concatenates_selected_rows_in_order():
    """The flat float4[] the INSERT slices is the selected rows, row-major."""
    array = hashing.to_storage_array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], VectorType.vector, 2)

    assert hashing.flatten_rows(array, [0, 1, 2]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert hashing.flatten_rows(array, [0, 2]) == [1.0, 2.0, 5.0, 6.0]


def test_hash_rows_matches_hash_vector_row_by_row():
    """The batch and single-vector helpers agree, so bulk and single writes dedup together."""
    vectors = [[1.0, 2.0], [3.0, 4.0]]

    batch = hashing.hash_rows(hashing.to_storage_array(vectors, VectorType.vector, 2))
    singles = [hashing.hash_vector(vector, VectorType.vector, 2) for vector in vectors]

    assert batch == singles
