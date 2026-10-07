"""Tests for the database helper functions."""

import re

import pytest

from gen3_embeddings.database.helpers import (
    CollectionSearchTarget,
    affected_row_count,
    build_multi_collection_search_sql,
    build_search_sql,
)
from gen3_embeddings.models.schemas import DistanceMetric, VectorType


def search_sql(**overrides) -> str:
    """Build a search statement with the boilerplate arguments filled in."""
    kwargs = {
        "table": "embeddings_vector",
        "distance_metric": DistanceMetric.cosine_distance,
        "collection_id_param": "$1::bigint",
        "vector_placeholder": "$2",
        "top_k_param": "$3::int",
        "filters": None,
        "min_value": None,
        "max_value": None,
        "vector_type": VectorType.vector,
        "dimensions": 1536,
    }
    kwargs.update(overrides)
    return " ".join(build_search_sql(**kwargs)[0].split())


def order_by(sql: str) -> str:
    """Return the ORDER BY clause, which is the only part an index can serve."""
    return re.search(r"ORDER BY (.*?) LIMIT", sql).group(1)


@pytest.mark.parametrize(
    "command_tag, expected",
    [
        # the case that made `tag.startswith("DELETE")` wrong: a statement that matched
        # nothing still reports its verb
        ("DELETE 0", 0),
        ("DELETE 1", 1),
        ("DELETE 42", 42),
        # other command tags are shaped the same way
        ("UPDATE 0", 0),
        ("UPDATE 7", 7),
        # INSERT tags carry an oid before the count
        ("INSERT 0 3", 3),
        # nothing parsable means nothing was reported as affected
        ("DELETE", 0),
        ("", 0),
        ("SELECT", 0),
    ],
)
def test_affected_row_count(command_tag, expected):
    """The count comes from the tag's trailing number, not from the verb."""
    assert affected_row_count(command_tag) == expected


def test_affected_row_count_distinguishes_zero_from_nonzero():
    """The whole point: 'nothing matched' must be falsy where 'something matched' is truthy."""
    assert affected_row_count("DELETE 0") == 0
    assert affected_row_count("DELETE 1") > 0


class TestBuildSearchSql:
    """The single-collection search statement."""

    @pytest.mark.parametrize(
        "metric, operator",
        [
            (DistanceMetric.l2_distance, "<->"),
            (DistanceMetric.inner_product, "<#>"),
            (DistanceMetric.cosine_distance, "<=>"),
            (DistanceMetric.l1_distance, "<+>"),
            # shares the cosine operator and differs only in the value it reports
            (DistanceMetric.cosine_similarity, "<=>"),
        ],
    )
    def test_every_metric_orders_by_a_bare_operator_ascending(self, metric, operator):
        """
        The ORDER BY is always `indexed_expression <op> query ASC`, for every metric.

        That shape is the only one pgvector can answer from an index. `<#>` is the *negative*
        inner product, so ascending really is "closest first" for all of them, and
        `cosine_similarity` gets there by ordering on the distance rather than on `1 - distance`.
        """
        assert order_by(search_sql(distance_metric=metric)) == (
            f"embedding::vector(1536) {operator} $2::vector(1536) ASC"
        )

    def test_cosine_similarity_reports_one_minus_distance_without_ordering_by_it(self):
        """
        The reported value is the similarity while the ordering stays indexable.

        Ordering by `1 - (embedding <=> $2) DESC` would rank identically but wraps the operator
        in arithmetic, and an index cannot serve that. The
        projection still has to report the similarity, so value and order must differ here.
        """
        sql = search_sql(distance_metric=DistanceMetric.cosine_similarity)

        assert "1 - (embedding::vector(1536) <=> $2::vector(1536)) AS value" in sql
        assert "1 -" not in order_by(sql)
        assert "DESC" not in sql

    def test_a_threshold_moves_outside_a_materialized_cte(self):
        """
        A bound on the ordered expression has to be applied after the LIMIT, not beside it.

        Postgres cannot end an ordered index scan early while rows are still being filtered out,
        so a threshold sitting in the same WHERE as the ORDER BY makes the scan walk the entire
        index -- the exact cost this rewrite removes. MATERIALIZED is what stops the planner
        inlining the CTE and collapsing the two back together.
        """
        sql = search_sql(distance_metric=DistanceMetric.cosine_distance, max_value=0.2)

        assert "WITH nearest AS MATERIALIZED (" in sql
        assert "FROM nearest WHERE value <= $4" in sql
        # the bound must not also appear next to the ordering, or nothing was gained
        assert "LIMIT $3::int" in sql
        assert sql.index("LIMIT $3::int") < sql.index("value <= $4")

    def test_the_threshold_reads_the_reported_value_not_the_raw_distance(self):
        """
        `min_value`/`max_value` mean "similarity at least/at most" under `cosine_similarity`.

        The value and the ordering expression are inverses there, so comparing the bound against
        the raw distance would silently flip the filter and keep the farthest rows. Referring to
        the CTE's `value` column makes that impossible to get wrong.
        """
        sql = search_sql(distance_metric=DistanceMetric.cosine_similarity, min_value=0.8)

        assert "1 - (embedding::vector(1536) <=> $2::vector(1536)) AS value" in sql
        assert "WHERE value >= $4" in sql
        # ordering inside the CTE stays on the bare distance, which is what the index serves
        assert order_by(sql) == "embedding::vector(1536) <=> $2::vector(1536) ASC"
        # ...and the outer sort restores similarity order, where larger is closer
        assert sql.rstrip().endswith("ORDER BY value DESC")

    def test_no_threshold_means_no_cte(self):
        """The common search needs no CTE, and a plain ORDER BY ... LIMIT is easier to read in a log."""
        sql = search_sql(distance_metric=DistanceMetric.cosine_distance)

        assert "MATERIALIZED" not in sql
        assert sql.startswith("SELECT *")

    def test_binary_rescore_selects_by_hamming_then_reranks_exactly(self):
        """
        The two-stage shape: cheap candidates from the bit index, exact ranking over just those.

        Binary quantization keeps only the sign of each dimension, so the Hamming ordering decides
        which rows to look at and nothing else. The outer query re-ranks them by the real metric,
        which is what makes the reported value trustworthy despite the lossy index.
        """
        sql = search_sql(distance_metric=DistanceMetric.cosine_similarity, binary_rescore_limit=200)

        assert "WITH candidates AS MATERIALIZED (" in sql
        # candidates by Hamming, capped at the rescore pool
        assert "ORDER BY binary_quantize(embedding)::bit(1536) <~> binary_quantize($2::vector(1536)) ASC" in sql
        assert "LIMIT 200" in sql
        # then exact re-ranking, and only now the caller's own limit
        assert sql.rstrip().endswith("ORDER BY value DESC LIMIT $3::int")
        assert "1 - (embedding::vector(1536) <=> $2::vector(1536)) AS value" in sql

    def test_only_the_indexed_side_carries_the_bit_cast(self):
        """
        The left operand must match the indexed expression exactly; the right must not be coerced.

        The index is on `(binary_quantize(embedding))::bit(n)`. `binary_quantize()` of the query
        vector already yields a bit string of that length, so casting it again is unnecessary --
        and any mismatch on the left silently costs the index.
        """
        sql = search_sql(binary_rescore_limit=100, dimensions=768)

        assert "binary_quantize(embedding)::bit(768) <~> binary_quantize($2::vector(768))" in sql

    def test_binary_rescore_still_applies_thresholds_after_the_rerank(self):
        """A bound filters the re-ranked rows, so it compares against the exact value, not Hamming."""
        sql = search_sql(distance_metric=DistanceMetric.cosine_distance, binary_rescore_limit=100, max_value=0.2)

        assert sql.index("LIMIT 100") < sql.index("value <= $4")
        assert "FROM candidates WHERE value <= $4 ORDER BY value ASC LIMIT $3::int" in sql

    def test_no_rescore_limit_means_the_direct_shape(self):
        """A collection with a normal vector index must not be sent through the binary path."""
        sql = search_sql(distance_metric=DistanceMetric.cosine_distance)

        assert "binary_quantize" not in sql
        assert "candidates" not in sql

    def test_metadata_filters_stay_inside_the_cte(self):
        """
        Only the distance bound belongs outside; everything else must stay where the scan sees it.

        `hnsw.iterative_scan` keeps feeding an index scan until `top_k` rows survive the filters
        applied within it. A metadata filter hoisted outside the CTE would instead shrink an
        already-limited result, turning a full page of hits into a handful.
        """
        sql = search_sql(distance_metric=DistanceMetric.cosine_distance, filters={"kind": "doc"}, max_value=0.2)

        assert sql.index("metadata->>($4::text) = $5::text") < sql.index("LIMIT $3::int")
        assert "WHERE value <= $6" in sql

    def test_both_operands_carry_the_collections_dimension(self):
        """
        The cast is load-bearing, on the column *and* on the query vector.

        The partial index is built on `embedding::vector(n)`; Postgres uses an expression index
        only on an exact match, so emitting a bare `embedding` or a different `n` is a silent
        fall back to a sequential scan.
        """
        sql = search_sql(dimensions=256, vector_type=VectorType.halfvec, distance_metric=DistanceMetric.l2_distance)

        assert order_by(sql) == "embedding::halfvec(256) <-> $2::halfvec(256) ASC"

    def test_vector_type_may_be_the_plain_string(self):
        """`VectorType` is a StrEnum and callers pass either form; reaching for `.value` must not care."""
        assert "embedding::halfvec(8)" in search_sql(vector_type="halfvec", dimensions=8)

    def test_an_unknown_metric_is_rejected(self):
        """A metric with no pgvector operator must fail loudly rather than build broken SQL."""
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            search_sql(distance_metric="not_a_metric")


def multi_sql(**overrides) -> str:
    """Build a cross-collection search statement with the boilerplate arguments filled in."""
    kwargs = {
        "table": "embeddings_vector",
        "distance_metric": DistanceMetric.cosine_distance,
        "targets": [CollectionSearchTarget(1), CollectionSearchTarget(2)],
        "vector_placeholder": "$1",
        "top_k_param": "$2::int",
        "filters": None,
        "min_value": None,
        "max_value": None,
        "vector_type": VectorType.vector,
        "dimensions": 1536,
    }
    kwargs.update(overrides)
    return " ".join(build_multi_collection_search_sql(**kwargs)[0].split())


class TestBuildMultiCollectionSearchSql:
    """The cross-collection search statement, one UNION ALL arm per collection."""

    def test_each_collection_becomes_its_own_arm(self):
        """
        One arm per collection, because one arm is all a per-collection partial index can serve.

        This is the whole point of the rewrite: `collection_id = ANY($1::bigint[])` implies
        nothing about any single collection, so Postgres cannot prove the `WHERE collection_id =
        n` predicate of any partial index and scans the table instead.
        """
        sql = multi_sql(targets=[CollectionSearchTarget(1), CollectionSearchTarget(2), CollectionSearchTarget(3)])

        assert sql.count("UNION ALL") == 2
        assert "ANY(" not in sql
        for collection_id in (1, 2, 3):
            assert f"WHERE collection_id = {collection_id} " in sql

    def test_the_collection_id_is_a_literal_not_a_placeholder(self):
        """
        A parameter would leave the partial-index predicate unprovable in a generic plan.

        Postgres only sometimes re-plans a parameterized statement per value; a literal makes the
        implication hold unconditionally, which is what keeps the arm index-eligible.
        """
        sql = multi_sql(targets=[CollectionSearchTarget(101)])

        assert "WHERE collection_id = 101 " in sql
        # $1 is the vector and $2 the limit, so no placeholder was spent on the collection
        assert "collection_id = $" not in sql

    def test_a_non_integer_collection_id_is_rejected_rather_than_interpolated(self):
        """
        The ids are written into the SQL text, so anything that is not an integer must not reach it.

        They come from `collections.id` and cannot normally be anything else, but `int()` is what
        makes that a guarantee rather than an assumption.
        """
        with pytest.raises(ValueError):
            multi_sql(targets=[CollectionSearchTarget("1; DROP TABLE embeddings_vector")])

    def test_arms_carry_the_index_shape_their_own_collection_has(self):
        """
        A mixed set has to be split: collections are indexed independently.

        One collection may have a direct fp32 index while the next has only a binary-quantized
        one. A single query body has exactly one ordering expression, so serving both from one
        scan is impossible regardless of the predicate question.
        """
        sql = multi_sql(targets=[CollectionSearchTarget(1), CollectionSearchTarget(2, binary_rescore_limit=200)])

        direct, binary = sql.split("UNION ALL")
        # the direct arm orders by the indexed distance and never quantizes
        assert "binary_quantize" not in direct
        assert "ORDER BY embedding::vector(1536) <=> $1::vector(1536) ASC LIMIT $2::int" in direct
        # the binary arm picks a pool by Hamming, then re-ranks it exactly
        assert "ORDER BY binary_quantize(embedding)::bit(1536) <~> binary_quantize($1::vector(1536)) ASC" in binary
        assert "LIMIT 200) AS rescored_2 ORDER BY value ASC LIMIT $2::int" in binary

    def test_every_arm_takes_its_own_top_k(self):
        """
        Per-arm limits are what keep one dense collection from crowding out the rest.

        Limiting only the merged result would be correct but would force every arm to stream all
        of its rows; limiting each arm to `top_k` discards nothing, because no collection can
        contribute more than `top_k` rows to a global top-k anyway.
        """
        sql = multi_sql()

        assert sql.count("LIMIT $2::int") == 3  # two arms plus the merged result
        assert sql.rstrip().endswith("ORDER BY value ASC LIMIT $2::int")

    def test_a_single_arm_is_still_wrapped_in_the_cte(self):
        """
        A trailing ORDER BY is the union's only when there *is* a union.

        With one arm there is no set operation, so it attaches to a SELECT that already has an
        ORDER BY and Postgres rejects the statement: "multiple ORDER BY clauses not allowed".
        Always wrapping keeps one shape and no special case. String assertions cannot catch this
        -- only a real parser can -- so the integration suite is what guards it.
        """
        sql = multi_sql(targets=[CollectionSearchTarget(1)])

        assert "UNION ALL" not in sql
        assert sql.startswith("WITH candidates AS MATERIALIZED (")
        assert sql.rstrip().endswith("ORDER BY value ASC LIMIT $2::int")

    def test_a_threshold_filters_the_merged_result_outside_a_materialized_cte(self):
        """
        Same reason as the single-collection path: a bound beside the ORDER BY defeats early exit.

        Here it also has to be applied once to the merged rows rather than per arm, which the CTE
        makes structural.
        """
        sql = multi_sql(max_value=0.2)

        assert "FROM candidates WHERE value <= $3" in sql
        # every arm's own LIMIT is applied before the bound ever sees a row
        assert sql.index("UNION ALL") < sql.index("value <= $3")
        assert sql.rindex("LIMIT $2::int)") < sql.index("value <= $3")

    def test_filters_are_applied_in_every_arm_and_share_their_placeholders(self):
        """
        A metadata filter has to reach every scan, but must not cost a parameter per arm.

        Repeating the clause with the same placeholders keeps the parameter list independent of
        how many collections were searched.
        """
        sql, params = build_multi_collection_search_sql(
            table="embeddings_vector",
            distance_metric=DistanceMetric.cosine_distance,
            targets=[CollectionSearchTarget(1), CollectionSearchTarget(2)],
            vector_placeholder="$1",
            top_k_param="$2::int",
            filters={"organ": "lung"},
            min_value=None,
            max_value=None,
            vector_type=VectorType.vector,
            dimensions=1536,
        )
        sql = " ".join(sql.split())

        assert sql.count("metadata->>($3::text) = $4::text") == 2
        assert params == ["organ", "lung"]

    def test_cosine_similarity_reranks_descending_in_the_arms_and_the_merge(self):
        """
        The reported value inverts the ordering, so every re-sort on it has to flip direction.

        The arms still order by the bare distance ascending, which is what the index serves; only
        the sorts that read `value` change direction.
        """
        sql = multi_sql(
            distance_metric=DistanceMetric.cosine_similarity,
            targets=[CollectionSearchTarget(1, binary_rescore_limit=50)],
        )

        assert "1 - (embedding::vector(1536) <=> $1::vector(1536)) AS value" in sql
        # once to re-rank the arm's Hamming pool, once to merge the arms
        assert sql.count("ORDER BY value DESC") == 2
        # the arm still hands the index a bare ascending Hamming ordering to serve
        assert "ORDER BY binary_quantize(embedding)::bit(1536) <~> binary_quantize($1::vector(1536)) ASC" in sql
        assert "DESC LIMIT" in sql and "value DESC LIMIT $2::int" in sql

    def test_searching_no_collections_is_a_programming_error(self):
        """
        An empty union is not valid SQL, and the DAL already returns early for this case.

        Failing here makes a future caller that forgets to do so fail loudly rather than send
        Postgres a statement it cannot parse.
        """
        with pytest.raises(ValueError, match="at least one collection"):
            multi_sql(targets=[])
