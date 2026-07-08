from common.metrics import ServiceMetrics


class AiModelRepoServiceMetrics(ServiceMetrics):
    """Service metrics collector for the Gen3 AI model repo service."""

    def __init__(self, metrics_client) -> None:
        self.metrics_client = metrics_client

    def add_to_api_interaction_counter(
        self, name="gen3_ai_model_repo_api_requests", description="API requests for Gen3 AI Model Repo.", **kwargs
    ) -> None:
        """
        Increment the counter for API requests related to this service. We use the provided
        keyword arguments as labels for the counter.

        Args:
            name (str): Name of the counter to increment.
            description (str): Description of the counter.
            **kwargs: Arbitrary keyword arguments used as labels for the counter. Typically includes labels
            such as http method, path, response time, and user id.
        """
        self.metrics_client.increment_counter(name=name, description=description, labels=kwargs)

    def add_models_count_metric(self, model_count: int, **labels) -> None:
        """Set a gauge for the current number of model repositories."""
        self.metrics_client.set_gauge(
            name="gen3_ai_model_repo_models_count",
            description="Observed number of model repositories.",
            labels=labels,
            value=model_count,
        )

    def add_stored_files_count_metric(self, file_count: int, **labels) -> None:
        """Set a gauge for the current number of tracked files."""
        self.metrics_client.set_gauge(
            name="gen3_ai_model_repo_files_count",
            description="Observed number of files tracked in model repositories.",
            labels=labels,
            value=file_count,
        )

    def add_stored_models_size_metric(self, total_size_bytes: int, **labels) -> None:
        """Set a gauge for total stored model size in bytes."""
        self.metrics_client.set_gauge(
            name="gen3_ai_model_repo_total_size_bytes",
            description="Observed total size in bytes of model files stored by the service.",
            labels=labels,
            value=total_size_bytes,
        )
