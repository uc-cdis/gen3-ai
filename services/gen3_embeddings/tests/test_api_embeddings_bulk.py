"""Tests for the bulk embedding read endpoints (routes/embeddings_bulk.py)."""

import base64
import struct

import pytest


def _decode_vector_base64(vector_base64: str, precision: str) -> list[float]:
    """Mirror gen3sdk-python's get_embeddings_from_bulk_content_guid decoding."""
    fmt_char = "e" if precision == "float16" else "f"
    padding_needed = -len(vector_base64) % 4
    padded_b64 = vector_base64 + ("=" * padding_needed)
    raw = base64.urlsafe_b64decode(padded_b64)
    count = len(raw) // struct.calcsize(fmt_char)
    return list(struct.unpack(f"<{count}{fmt_char}", raw))


def test_bulk_read_from_collection(client, allow_authz):
    """Bulk read from a known collection returns binary-encoded vectors with count and float32 precision."""
    allow_authz("docs")

    client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "documents", "dimensions": 3, "vector_type": "vector"},
    )

    create_resp = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding": [1.0, 0.0, 0.0], "metadata": {}},
                {"embedding": [0.0, 1.0, 0.0], "metadata": {}},
            ]
        },
    )
    assert create_resp.status_code == 200
    ids = [e["embedding_id"] for e in create_resp.json()["embeddings"]]

    bulk_resp = client.post("/vectorstore/collections/docs/embeddings/bulk", json=ids)
    assert bulk_resp.status_code == 200, bulk_resp.text

    data = bulk_resp.json()
    assert data["count"] == 2
    assert len(data["embeddings"]) == 2
    assert data["embeddings"][0]["precision"] == "float32"

    first, second = data["embeddings"]
    decoded_first = _decode_vector_base64(first["vector_base64"], first["precision"])
    decoded_second = _decode_vector_base64(second["vector_base64"], second["precision"])

    assert decoded_first == pytest.approx([1.0, 0.0, 0.0])
    assert decoded_second == pytest.approx([0.0, 1.0, 0.0])


def test_bulk_read_from_halfvec_collection(client, allow_authz):
    """Bulk read from a halfvec collection returns float16-precision binary-encoded vectors."""
    allow_authz("docs")

    client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "documents", "dimensions": 3, "vector_type": "halfvec"},
    )

    create_resp = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={
            "embeddings": [
                {"embedding": [0.1, 0.2, 0.3], "metadata": {}},
                {"embedding": [0.4, 0.5, 0.6], "metadata": {}},
            ]
        },
    )
    assert create_resp.status_code == 200
    ids = [e["embedding_id"] for e in create_resp.json()["embeddings"]]

    bulk_resp = client.post("/vectorstore/collections/docs/embeddings/bulk", json=ids)
    assert bulk_resp.status_code == 200, bulk_resp.text

    data = bulk_resp.json()
    assert data["count"] == 2
    assert len(data["embeddings"]) == 2
    assert data["embeddings"][0]["precision"] == "float16"

    first, second = data["embeddings"]
    decoded_first = _decode_vector_base64(first["vector_base64"], first["precision"])
    decoded_second = _decode_vector_base64(second["vector_base64"], second["precision"])

    # float16 has ~3 significant decimal digits of precision
    assert decoded_first == pytest.approx([0.1, 0.2, 0.3], rel=1e-3)
    assert decoded_second == pytest.approx([0.4, 0.5, 0.6], rel=1e-3)


def test_bulk_read_unknown_collections(client, allow_authz):
    """
    Bulk read spanning two collections returns each embedding's values, decoded at its own
    collection's precision, in the requested order, plus metadata for both collections.
    """
    allow_authz("docs", "images")

    # Two collections with different vector types, so each embedding must be decoded at
    # the precision of the collection it came from.
    for name, vector_type in (("docs", "vector"), ("images", "halfvec")):
        resp = client.post(
            "/vectorstore/collections",
            json={"collection_name": name, "description": name, "dimensions": 3, "vector_type": vector_type},
        )
        assert resp.status_code == 200, resp.text

    docs_vector = [1.0, 0.0, 0.0]
    # exactly representable in float16, so halfvec storage round-trips without loss
    images_vector = [0.25, 0.5, 0.75]

    docs_resp = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": docs_vector, "metadata": {"src": "a.txt"}}]},
    )
    assert docs_resp.status_code == 200, docs_resp.text
    docs_id = docs_resp.json()["embeddings"][0]["embedding_id"]

    images_resp = client.post(
        "/vectorstore/collections/images/embeddings",
        json={"embeddings": [{"embedding": images_vector, "metadata": {"src": "b.png"}}]},
    )
    assert images_resp.status_code == 200, images_resp.text
    images_id = images_resp.json()["embeddings"][0]["embedding_id"]

    # request images first to confirm the response follows input order, not storage order
    requested_ids = [images_id, docs_id]
    bulk_resp = client.post("/embeddings/bulk", json=requested_ids)
    assert bulk_resp.status_code == 200, bulk_resp.text

    data = bulk_resp.json()

    # both source collections are described, so the caller can interpret the vectors
    assert {c["collection_name"] for c in data["collections"]} == {"docs", "images"}
    collection_name_by_id = {c["id"]: c["collection_name"] for c in data["collections"]}

    embeddings = data["embeddings"]
    assert len(embeddings) == 2
    assert [e["embedding_id"] for e in embeddings] == requested_ids
    assert [e["input_index"] for e in embeddings] == [0, 1]

    by_id = {e["embedding_id"]: e for e in embeddings}

    images_emb = by_id[images_id]
    assert images_emb["precision"] == "float16"
    assert collection_name_by_id[images_emb["info"]["collection_id"]] == "images"
    assert _decode_vector_base64(images_emb["vector_base64"], images_emb["precision"]) == pytest.approx(images_vector)

    docs_emb = by_id[docs_id]
    assert docs_emb["precision"] == "float32"
    assert collection_name_by_id[docs_emb["info"]["collection_id"]] == "docs"
    assert _decode_vector_base64(docs_emb["vector_base64"], docs_emb["precision"]) == pytest.approx(docs_vector)
