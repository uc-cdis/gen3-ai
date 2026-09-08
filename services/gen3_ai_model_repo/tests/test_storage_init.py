import importlib

import pytest


def test_get_storage_provider_returns_local_provider(monkeypatch):
    """
    Verify local provider is constructed when STORAGE_PROVIDER=local.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch storage config values.
    """
    storage_helpers = importlib.import_module("gen3_ai_model_repo.storage.helpers")
    monkeypatch.setattr(storage_helpers, "STORAGE_PROVIDER", "local", raising=False)
    monkeypatch.setattr(storage_helpers, "LOCAL_STORAGE_PATH", "/tmp/gen3-models", raising=False)

    provider = storage_helpers.get_storage_provider()

    assert provider.__class__.__name__ == "LocalStorageProvider"


def test_get_storage_provider_returns_s3_provider(monkeypatch):
    """
    Verify S3 provider is constructed when STORAGE_PROVIDER=s3.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch storage config values.
    """
    storage_helpers = importlib.import_module("gen3_ai_model_repo.storage.helpers")
    monkeypatch.setattr(storage_helpers, "STORAGE_PROVIDER", "s3", raising=False)
    monkeypatch.setattr(storage_helpers, "S3_BUCKET", "model-repo", raising=False)
    monkeypatch.setattr(storage_helpers, "S3_REGION", "us-east-1", raising=False)
    monkeypatch.setattr(storage_helpers, "S3_ENDPOINT_URL", "", raising=False)
    monkeypatch.setattr(storage_helpers, "S3_ACCESS_KEY_ID", "", raising=False)
    monkeypatch.setattr(storage_helpers, "S3_SECRET_ACCESS_KEY", "", raising=False)

    provider = storage_helpers.get_storage_provider()

    assert provider.__class__.__name__ == "S3StorageProvider"


def test_get_storage_provider_raises_on_unknown_provider(monkeypatch):
    """
    Verify unsupported provider names raise a clear ValueError.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to patch storage config values.
    """
    storage_helpers = importlib.import_module("gen3_ai_model_repo.storage.helpers")
    monkeypatch.setattr(storage_helpers, "STORAGE_PROVIDER", "unknown-provider", raising=False)

    with pytest.raises(ValueError, match="Unsupported STORAGE_PROVIDER"):
        storage_helpers.get_storage_provider()
