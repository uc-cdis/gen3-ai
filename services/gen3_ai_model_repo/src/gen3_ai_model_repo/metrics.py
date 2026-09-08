"""Metrics helpers for the Gen3 AI model repo service."""

from common.metrics import ServiceMetrics


class AiModelRepoServiceMetrics(ServiceMetrics):
    """Service metrics collector for the Gen3 AI model repo service."""

    def add_models_count_metric(self, model_count: int, **labels) -> None:
        """Set a gauge for the current number of model repositories."""
        if not self.metrics_client.enabled:
            return
        self.metrics_client.set_gauge(
            name="gen3_ai_model_repo_models_count",
            description="Observed number of model repositories.",
            labels=labels,
            value=model_count,
        )

    def add_stored_files_count_metric(self, file_count: int, **labels) -> None:
        """Set a gauge for the current number of tracked files."""
        if not self.metrics_client.enabled:
            return
        self.metrics_client.set_gauge(
            name="gen3_ai_model_repo_files_count",
            description="Observed number of files tracked in model repositories.",
            labels=labels,
            value=file_count,
        )

    def add_stored_models_size_metric(self, total_size_bytes: int, **labels) -> None:
        """Set a gauge for total stored model size in bytes."""
        if not self.metrics_client.enabled:
            return
        self.metrics_client.set_gauge(
            name="gen3_ai_model_repo_total_size_bytes",
            description="Observed total size in bytes of model files stored by the service.",
            labels=labels,
            value=total_size_bytes,
        )
