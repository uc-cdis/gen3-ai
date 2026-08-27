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
