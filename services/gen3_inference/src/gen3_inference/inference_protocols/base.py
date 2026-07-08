"""
Model Info from HF:

https://github.com/huggingface/huggingface_hub/blob/176bdfb9c1459f5c5f0b70a2ca6b2b9fa02dffc1/src/huggingface_hub/hf_api.py#L822
"""

from abc import ABC, abstractmethod

from fastapi.responses import JSONResponse, StreamingResponse
from openresponses_types import CreateResponseBody

"""Base class for inference protocol clients.

generate_non_streaming_response: Generate a non-streaming response from the request body.
generate_streaming_response: Generate a streaming response from the request body.
"""


class InferenceProtocolClient(ABC):
    """Base class for inference protocol clients.

    generate_non_streaming_response: Generate a non-streaming response from the request body.
    generate_streaming_response: Generate a streaming response from the request body.
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url

    @abstractmethod
    async def generate_non_streaming_response(self, body: CreateResponseBody, model_info: dict) -> JSONResponse:
        """Generate a non-streaming response.

        body: The request payload to send to the inference provider.
        model_info: Metadata describing the target model.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_streaming_response(self, body: CreateResponseBody, model_info: dict) -> StreamingResponse:
        """Generate a streaming response.

        body: The request payload to send to the inference provider.
        model_info: Metadata describing the target model.
        """
        raise NotImplementedError()
