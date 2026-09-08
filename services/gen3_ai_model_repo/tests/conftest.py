import os
import tempfile

import pytest

from gen3_ai_model_repo.storage.helpers import reset_storage_provider

# prometheus_client selects its storage backend when it is first imported.
os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(prefix="gen3-model-repo-metrics-")
os.environ["ENABLE_OPENTELEMETRY_TRACES"] = "false"


@pytest.fixture(autouse=True)
def reset_storage_provider_cache():
    """Ensure each test starts and ends without a cached storage provider."""
    reset_storage_provider()
    yield
    reset_storage_provider()
