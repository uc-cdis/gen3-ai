"""
Model Info from HF:

https://github.com/huggingface/huggingface_hub/blob/176bdfb9c1459f5c5f0b70a2ca6b2b9fa02dffc1/src/huggingface_hub/hf_api.py#L822
"""

from abc import ABC, abstractmethod

from fastapi.responses import JSONResponse, StreamingResponse
from openresponses_types import CreateResponseBody


class InferenceProtocolClient(ABC):
    """
    Interface every upstream inference protocol client must implement.

    Subclasses adapt one wire protocol (OpenAI Chat, Open Responses, ...) so the routes can
    stay protocol-agnostic.
    """

    def __init__(self, base_url: str | None = None):
        """
        Args:
            base_url (str | None): Base URL of the upstream inference server.
        """
        self.base_url = base_url

    @abstractmethod
    async def generate_non_streaming_response(self, body: CreateResponseBody, model_info: dict) -> JSONResponse:
        """
        Generate a complete response in a single reply.

        Args:
            body (CreateResponseBody): The Open Responses request.
            model_info (dict): Metadata about the target model.

        Returns:
            JSONResponse: The full response.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_streaming_response(self, body: CreateResponseBody, model_info: dict) -> StreamingResponse:
        """
        Generate a response as a stream of Open Responses events.

        Args:
            body (CreateResponseBody): The Open Responses request.
            model_info (dict): Metadata about the target model.

        Returns:
            StreamingResponse: The streamed response.
        """
        raise NotImplementedError()
