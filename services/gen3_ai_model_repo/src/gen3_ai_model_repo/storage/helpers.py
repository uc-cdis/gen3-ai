"""Storage helper utilities for the Gen3 AI model repo service."""

from gen3_ai_model_repo.config import (
    LOCAL_STORAGE_PATH,
    S3_ACCESS_KEY_ID,
    S3_BUCKET,
    S3_ENDPOINT_URL,
    S3_REGION,
    S3_SECRET_ACCESS_KEY,
    S3_SESSION_TOKEN,
    STORAGE_CREATE_BUCKET_IF_MISSING,
    STORAGE_PROVIDER,
)
from gen3_ai_model_repo.storage.local import LocalStorageProvider

_provider_cache = None


def reset_storage_provider() -> None:
    """Clear the cached storage provider, primarily for test isolation."""
    global _provider_cache
    _provider_cache = None


def get_storage_provider():
    """
    Return the configured storage provider implementation.

    Returns:
        StorageProvider: The configured local or S3-compatible storage provider.

    Raises:
        ValueError: If an unsupported STORAGE_PROVIDER is configured.
    """
    global _provider_cache

    if _provider_cache is not None:
        return _provider_cache

    if STORAGE_PROVIDER == "s3":
        from gen3_ai_model_repo.storage.s3 import S3StorageProvider

        _provider_cache = S3StorageProvider(
            bucket_name=S3_BUCKET,
            region=S3_REGION,
            endpoint_url=S3_ENDPOINT_URL,
            access_key_id=S3_ACCESS_KEY_ID,
            secret_access_key=S3_SECRET_ACCESS_KEY,
            session_token=S3_SESSION_TOKEN,
            create_bucket_if_missing=STORAGE_CREATE_BUCKET_IF_MISSING,
        )
        return _provider_cache

    if STORAGE_PROVIDER != "local":
        raise ValueError(f"Unsupported STORAGE_PROVIDER: {STORAGE_PROVIDER}")

    _provider_cache = LocalStorageProvider(LOCAL_STORAGE_PATH)
    return _provider_cache


__all__ = ["get_storage_provider", "reset_storage_provider"]
