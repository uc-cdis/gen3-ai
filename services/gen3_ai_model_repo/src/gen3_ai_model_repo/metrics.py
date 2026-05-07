from common.metrics import ServiceMetrics


class AiModelRepoServiceMetrics(ServiceMetrics):
    def __init__(self, metrics_client) -> None:
        self.metrics_client = metrics_client

    def add_to_api_interaction_counter(
        self, name="gen3_ai_model_repo_api_requests", description="API requests for Gen3 AI Model Repo.", **kwargs
    ) -> None:
        """
        Increment the counter for API requests related to this service. We use the provided
        keyword arguments as labels for the counter.

        Args:
            **kwargs: Arbitrary keyword arguments used as labels for the counter. Typically includes labels
            such as http method, path, response time, and user id.
        """
        self.metrics_client.increment_counter(name=name, description=description, labels=kwargs)
