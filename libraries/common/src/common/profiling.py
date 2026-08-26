"""
Continuous profiling with Pyroscope, shared by all services.

The agent handling lives in `cdispyutils.observability.continuous_profiling`, which is Gen3-wide
and configured through keyword arguments. This module is where those arguments come from
`common.config`.
"""

from cdispyutils.observability.continuous_profiling import (
    configure_profiling as _configure_profiling,
)

# Re-exported so service code imports these from here and never names the library directly.
# Repeating the name after `as` is what marks an import as a deliberate re-export: nothing in
# this module calls them, so ruff would otherwise prune them as unused.
from cdispyutils.observability.continuous_profiling import (
    profiling_active as profiling_active,
)
from cdispyutils.observability.continuous_profiling import (
    stop_profiling as stop_profiling,
)

from common import config


def configure_profiling(service_name: str) -> None:
    """
    Start the Pyroscope agent so this process pushes CPU and memory profiles.

    Does nothing when ENABLE_CONTINUOUS_PROFILING is off, or when the agent is already running in
    this process. Call this before `common.telemetry.configure_tracing`, which asks
    `profiling_active` whether to link spans to profiles.

    Args:
        service_name (str): Value for Pyroscope's application name, which is what the profiles
            are grouped and queried under.
    """
    _configure_profiling(
        service_name,
        enabled=config.ENABLE_CONTINUOUS_PROFILING,
        server_address=config.PYROSCOPE_SERVER_ADDRESS,
        sample_rate=config.PYROSCOPE_SAMPLE_RATE,
        upload_interval=config.PYROSCOPE_UPLOAD_INTERVAL,
        profile_cpu=config.PROFILE_CPU,
        profile_memory=config.PROFILE_MEMORY,
        on_cpu_only=config.PROFILE_ON_CPU_ONLY,
        basic_auth_username=config.PYROSCOPE_BASIC_AUTH_USERNAME,
        basic_auth_password=config.PYROSCOPE_BASIC_AUTH_PASSWORD,
        tenant_id=config.PYROSCOPE_TENANT_ID,
    )
