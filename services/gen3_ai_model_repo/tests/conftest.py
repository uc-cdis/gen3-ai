"""Shared fixtures and test-process configuration."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# The service declares common as a local path dependency. Keep pytest aligned
# with that source tree even when a stale wheel remains in the service venv.
COMMON_SOURCE = Path(__file__).resolve().parents[3] / "libraries" / "common" / "src"
sys.path.insert(0, str(COMMON_SOURCE))

from gen3_ai_model_repo.storage.helpers import reset_storage_provider  # noqa: E402

# prometheus_client selects its storage backend when it is first imported.
os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(prefix="gen3-model-repo-metrics-")
os.environ["ENABLE_OPENTELEMETRY_TRACES"] = "false"


@pytest.fixture(autouse=True)
def reset_storage_provider_cache():
    """Ensure each test starts and ends without a cached storage provider."""
    reset_storage_provider()
    yield
    reset_storage_provider()
