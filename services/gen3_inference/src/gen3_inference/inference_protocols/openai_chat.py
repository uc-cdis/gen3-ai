from fastapi.responses import JSONResponse, StreamingResponse
from openresponses_types.types import (
    CreateResponseBody,
)

from gen3_inference.inference_protocols.base import InferenceProtocolClient


class OpenaiChat(InferenceProtocolClient):
    NAME = "openai_chat"

    def __init__(self, base_url: str | None = None):
        super().__init__()

    async def generate_non_streaming_response(self, body: CreateResponseBody, model_info: dict) -> JSONResponse:
        pass

    def generate_streaming_response(self, body: CreateResponseBody, model_info: dict) -> StreamingResponse:
        pass
