import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_get_storage_provider_returns_local_provider(monkeypatch):
    storage_init = importlib.import_module("gen3_ai_model_repo.storage.init")

    monkeypatch.setattr(storage_init, "STORAGE_PROVIDER", "local", raising=False)
    monkeypatch.setattr(storage_init, "LOCAL_STORAGE_PATH", "/tmp/gen3-models", raising=False)

    provider = storage_init.get_storage_provider()

    assert provider.__class__.__name__ == "LocalStorageProvider"
