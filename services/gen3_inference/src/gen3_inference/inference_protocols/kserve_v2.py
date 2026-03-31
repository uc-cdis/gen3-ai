from openresponses_types.types import (
    CreateResponseBody,
    ResponseResource,
)

from gen3_inference.inference_protocols.base import InferenceProtocolClient


class KServev2Client(InferenceProtocolClient):
    NAME = "kserve_v2"

    def __init__(self, base_url: str, model_name: str):
        """
        TODO: actually implement
        base_url: e.g. "http://my-kserve-service.ml.svc.cluster.local"
        model_name: name used in /v2/models/{model_name}/infer
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    async def generate_non_streaming_response(self, body: CreateResponseBody) -> ResponseResource:
        raise NotImplementedError()

    async def generate_streaming_response(self, body: CreateResponseBody):
        raise NotImplementedError()
