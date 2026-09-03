"""Tests for the database helper functions."""

import pytest

from gen3_embeddings.database.helpers import affected_row_count


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
