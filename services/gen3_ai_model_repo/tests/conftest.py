import os
import sys
import tempfile
from pathlib import Path

# prometheus_client selects its storage backend when it is first imported.
os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(prefix="gen3-model-repo-metrics-")
os.environ["ENABLE_OPENTELEMETRY_TRACES"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

anyio_backend = "asyncio"
