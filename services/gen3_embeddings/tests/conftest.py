"""Shared test setup for the Gen3 Embeddings service."""

import os
import tempfile

# prometheus_client selects its multiprocess value class when it is first imported, so this has
# to happen before any test module imports the app. Without it the client falls back to
# in-memory values, writes no files, and every /metrics scrape comes back empty. Assigned
# rather than defaulted: a developer with this exported already would otherwise have the suite
# reading and writing whichever directory their shell points at.
os.environ["PROMETHEUS_MULTIPROC_DIR"] = tempfile.mkdtemp(prefix="gen3-embeddings-metrics-")

# Building the app installs a span exporter by default. Left on, every test pays for repeated retries
# against a collector that is not there.
os.environ["ENABLE_OPENTELEMETRY_TRACES"] = "false"
