"""
`cosine_similarity` and `cosine_distance` must rank identically and report inverse values.

Search used to emit `ORDER BY 1 - (embedding <=> $q) DESC` for `cosine_similarity`. No index
can serve an ordering wrapped in arithmetic, so it now orders by the bare distance ascending
and reports `1 - distance` as the value instead. The ranking is mathematically identical --
`1 - d` is strictly decreasing in `d` -- but "mathematically identical" is exactly the kind of
claim that should be checked rather than asserted, because getting the direction wrong would
silently return the FARTHEST rows while still looking well-formed.

The vectors are chosen so every expected value is exact and computable by hand.
"""

import pytest

QUERY = [1.0, 0.0, 0.0]

# (label, vector, cosine_similarity to QUERY, cosine_distance to QUERY)
CASES = [
    ("same", [1.0, 0.0, 0.0], 1.0, 0.0),
    ("diag", [1.0, 1.0, 0.0], 0.7071067811865475, 0.2928932188134525),
    ("orth", [0.0, 1.0, 0.0], 0.0, 1.0),
    ("oppo", [-1.0, 0.0, 0.0], -1.0, 2.0),
]


@pytest.fixture
def docs(client, allow_authz):
    """A collection holding the four vectors above, labelled in metadata."""
    allow_authz("docs")
    client.post(
        "/vectorstore/collections",
        json={"collection_name": "docs", "description": "d", "dimensions": 3, "vector_type": "vector"},
    )
    response = client.post(
        "/vectorstore/collections/docs/embeddings",
        json={"embeddings": [{"embedding": v, "metadata": {"label": label}} for label, v, _, _ in CASES]},
    )
    assert response.status_code == 200, response.text


def _search(client, **body):
    response = client.post("/vectorstore/collections/docs/search", json={"input": QUERY, "top_k": 10, **body})
    assert response.status_code == 200, response.text
    return response.json()["embeddings"]


def _labels(hits):
    return [hit["embedding"]["info"]["metadata"]["label"] for hit in hits]


def test_cosine_similarity_reports_exact_values_largest_first(client, docs):
    """Larger is closer, so the ordering runs 1.0 -> -1.0 and the values are 1 - distance."""
    hits = _search(client, distance_metric="cosine_similarity")

    assert _labels(hits) == ["same", "diag", "orth", "oppo"]
    for hit, (_label, _v, expected_similarity, _d) in zip(hits, CASES, strict=True):
        assert hit["value"] == pytest.approx(expected_similarity, abs=1e-6)


def test_cosine_distance_reports_exact_values_smallest_first(client, docs):
    """The same ranking, reported as distance: smaller is closer."""
    hits = _search(client, distance_metric="cosine_distance")

    assert _labels(hits) == ["same", "diag", "orth", "oppo"]
    for hit, (_label, _v, _s, expected_distance) in zip(hits, CASES, strict=True):
        assert hit["value"] == pytest.approx(expected_distance, abs=1e-6)


def test_the_two_metrics_agree_on_order_and_are_inverses(client, docs):
    """
    The property the rewrite depends on, checked end to end.

    If the ordering direction were ever flipped these two would disagree, and the similarity
    search would be returning the least similar rows.
    """
    similarity = _search(client, distance_metric="cosine_similarity")
    distance = _search(client, distance_metric="cosine_distance")

    assert [hit["id"] for hit in similarity] == [hit["id"] for hit in distance]
    for sim_hit, dist_hit in zip(similarity, distance, strict=True):
        assert sim_hit["value"] == pytest.approx(1.0 - dist_hit["value"], abs=1e-6)

    values = [hit["value"] for hit in similarity]
    assert values == sorted(values, reverse=True)


def test_min_value_on_cosine_similarity_keeps_the_most_similar(client, docs):
    """
    `min_value` means "at least this similar", not "at least this far".

    This is the threshold that now sits outside the materialized CTE and compares against the
    reported value rather than the raw distance. Comparing against the distance instead would
    invert it and keep exactly the rows this asserts are gone.
    """
    hits = _search(client, distance_metric="cosine_similarity", min_value=0.5)

    assert _labels(hits) == ["same", "diag"]


def test_max_value_on_cosine_distance_keeps_the_closest(client, docs):
    """The mirror case on a distance metric: `max_value` is the usual "minimum closeness" bound."""
    hits = _search(client, distance_metric="cosine_distance", max_value=0.5)

    assert _labels(hits) == ["same", "diag"]
