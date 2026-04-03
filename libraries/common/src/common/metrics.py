from cdispyutils.metrics import BaseMetrics as PrometheusMetrics
from fastapi import FastAPI

from common import config


class ServiceMetrics:
    """
    Services should extend this class to support additional metrics they need to manage
    """

    def __init__(self, metrics_client) -> None:
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
