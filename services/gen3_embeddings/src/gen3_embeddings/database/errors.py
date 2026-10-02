"""
Domain errors raised by the data access layer.

These deliberately carry no HTTP status codes. The DAL should not know how its failures are
presented over HTTP; `gen3_embeddings.error_handlers` owns the mapping from these types to
responses, so the DAL stays usable from anywhere (scripts, workers, tests) without FastAPI.
"""


class DataAccessError(Exception):
    """Base class for every error the data access layer raises."""


class InvalidCollectionNameError(DataAccessError):
    """A collection name failed normalization/validation."""


class CollectionNameNotAllowedError(DataAccessError):
    """The caller may not use this collection name."""


class RowLevelSecurityDeniedError(DataAccessError):
    """
    A policy's WITH CHECK rejected the row a write tried to produce.

    Only writes reach this, because only writes can fail loudly. A policy has two halves and
    they behave differently:

    - USING governs visibility for SELECT/UPDATE/DELETE. A row the caller may not see is
      simply absent -- zero rows, no error -- which is why a denied read surfaces as a 404
      or an empty list rather than here.
    - WITH CHECK governs the NEW row on INSERT and UPDATE. This is the only half that
      raises, and it means the caller asked to store an `authz` value (or a
      `collection_name`) outside their grants.

    The policy-engine check in the route layer normally rejects that first. This is the
    backstop for when it cannot: `DEBUG_SKIP_AUTH` skips the check entirely, and the policy
    engine's `auth_request` and `auth_mapping` answers can disagree.

    Without this translation the asyncpg `InsufficientPrivilegeError` escapes as a 500,
    reporting a caller's authorization failure as a server fault.
    """


class CollectionAlreadyExistsError(DataAccessError):
    """A collection with this name already exists."""


class CollectionCreateFailedError(DataAccessError):
    """The insert reported success but returned no row."""


class MetadataLengthMismatchError(DataAccessError):
    """A metadata list was supplied whose length does not match the embeddings list."""


class EmbeddingDimensionMismatchError(DataAccessError):
    """A vector's length does not match the dimensionality its collection declares."""


class EmbeddingNotRepresentableError(DataAccessError):
    """A vector holds a value the collection's storage type cannot represent."""


class EmbeddingsAlreadyExistError(DataAccessError):
    """One or more embeddings already exist in the collection."""


class DuplicateEmbeddingError(DataAccessError):
    """The write would leave two identical embeddings (same vector, metadata, and authz)."""


class EmbeddingWriteInconsistencyError(DataAccessError):
    """The number of rows written did not match the number of rows requested."""
