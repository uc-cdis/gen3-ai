"""
Common handling for metrics
"""

# isort: off
# prometheus_client freezes its choice of storage backend when it is first imported, based on
# PROMETHEUS_MULTIPROC_DIR being present. common.config is what puts that in the environment, so
# it has to be imported before anything that reaches prometheus_client - cdispyutils does.
# Get this backwards and /metrics serves an empty 200 forever, silently.
from common import config

from cdispyutils.metrics import BaseMetrics as PrometheusMetrics

# Re-exported for services to import from here. `cdispyutils.observability` sorts above
# `common`, so a service importing this directly gets prometheus_client before common.config.
# Repeating the name after `as` is what marks an import as a deliberate re-export: nothing in
# this module calls it, so ruff would otherwise prune it as unused.
from cdispyutils.observability.request_metrics import (
    add_request_metrics_middleware as add_request_metrics_middleware,
)
from fastapi import FastAPI
# isort: on


class ServiceMetrics:
    """
    Services should extend this class to support additional metrics they need to manage
    """

    def __init__(self, metrics_client) -> None:
        """
        Create instance of metrics class with client
        """
        self.metrics_client = metrics_client

    def add_to_api_interaction_counter(self, name, description, **kwargs) -> None:
        """
        Increment the counter for API requests related to this service. We use the provided
        keyword arguments as labels for the counter.

        Args:
            name: name of the counter (ideally this is configured globally for a specific service)
            description: description of the counter (ideally this is configured globally for a specific service)
            **kwargs: Arbitrary keyword arguments used as labels for the counter. Typically includes labels
            such as http method, path, response time, and user id.
        """
        if not self.metrics_client.enabled:
            return

        self.metrics_client.increment_counter(labels=kwargs, name=name, description=description)


def get_metrics_client(fastapi_app: FastAPI):
    """
    Get the client for handling metrics.

    Args:
        fastapi_app: The FastAPI application to which the metrics
            endpoint should be added (if any)

    Returns:
        The metrics client, or None when metrics are disabled.

    Raises:
        Exception: If metrics are enabled but METRICS_PROVIDER names an unsupported provider.
    """
    metrics_client = None
    metrics_client_kwargs = {}
    if config.METRICS_PROVIDER == "prometheus":
        metrics_client_kwargs = {"path": config.PROMETHEUS_MULTIPROC_DIR}
        metrics_client = PrometheusMetrics(
            enabled=config.ENABLE_METRICS, prometheus_dir=config.PROMETHEUS_MULTIPROC_DIR
        )
        metrics_app = metrics_client.get_metrics_app(**metrics_client_kwargs)
        fastapi_app.mount("/metrics", metrics_app)

    if not metrics_client and config.ENABLE_METRICS:
        raise Exception(
            f"ENABLE_METRICS is {config.ENABLE_METRICS}, but METRICS_PROVIDER=`{config.METRICS_PROVIDER}` is not supported"
        )

    return metrics_client
