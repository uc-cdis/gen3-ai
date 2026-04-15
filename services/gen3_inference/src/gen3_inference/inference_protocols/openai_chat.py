from urllib.parse import urlparse

import openai
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from openresponses_types import Content1, Role, UserMessageItemParam
from openresponses_types.types import (
    CreateResponseBody,
)

from gen3_inference.config import HOST_TO_CREDS, logging
from gen3_inference.errors import ERROR_TYPE_NOT_FOUND
from gen3_inference.inference_protocols.base import InferenceProtocolClient
from gen3_inference.inference_protocols.utils.openai_chat_to_openresponses import (
    chat_completion_to_openresponses_response,
    convert_chat_completion_stream_to_sse,
)
from gen3_inference.types import OpenResponsesError


class OpenaiChat(InferenceProtocolClient):
    NAME = "openai_chat"

    def __init__(self, base_url: str | None = None):
        super().__init__(base_url=base_url)

    async def generate_non_streaming_response(self, body: CreateResponseBody, model_info: dict) -> JSONResponse:
        """
        Generate a non-streaming chat completion response from OpenAI Chat API

        Args:
            body (CreateResponseBody): The request body containing model, input messages, and other parameters
            model_info (dict): Dictionary containing model information including url

        Returns:
            JSONResponse: The formatted chat completion response
        """
        host = str(urlparse(self.base_url).hostname) or ""
        if host not in HOST_TO_CREDS:
            logging.warning(
                f"{host} is not in HOST_TO_CREDS, unable to retrieve creds. Valid hosts: {[host for host in HOST_TO_CREDS]}. Continuing anyway..."
            )

        client = OpenAI(
            api_key=HOST_TO_CREDS.get(host),
            base_url=self.base_url,
            webhook_secret=HOST_TO_CREDS.get(host),
        )

        logging.debug(f"Generating chat completion for model `{body.model}`")

        messages = _get_messages(body)

        # TODO: backoff and retry
        try:
            # TODO: do we need to pass more metadata from the Open Responses request in here?
            response = client.chat.completions.create(
                model=str(body.model),
                messages=messages,
                stream=False,
                max_completion_tokens=int(body.max_output_tokens) if body.max_output_tokens else None,
            )
        except openai.NotFoundError as exc:
            error = OpenResponsesError(
                type=ERROR_TYPE_NOT_FOUND,
                code=ERROR_TYPE_NOT_FOUND,
                message="The requested model was not found.",
            )
            logging.error(f"{type(exc).__name__}: {exc}", exc_info=True)
            raise HTTPException(status_code=404, detail=error.to_json())
        except Exception as exc:
            logging.error(f"{type(exc).__name__}: {exc}", exc_info=True)
            raise HTTPException(status_code=503, detail="Unable to connect to inference server")

        standard_response = chat_completion_to_openresponses_response(response)
        logging.debug("Successfully generated chat completion response")
        return JSONResponse(content=standard_response.model_dump())

    def generate_streaming_response(self, body: CreateResponseBody, model_info: dict) -> StreamingResponse:
        """
        Generate a streaming chat completion response from OpenAI Chat API

        Args:
            body (CreateResponseBody): The request body containing model, input messages, and other parameters
            model_info (dict): Dictionary containing model information including url

        Returns:
            StreamingResponse: The formatted streaming chat completion response
        """
        host = str(urlparse(self.base_url).hostname) or ""
        if host not in HOST_TO_CREDS:
            logging.warning(
                f"{host} is not in HOST_TO_CREDS, unable to retrieve creds. Valid hosts: {[host for host in HOST_TO_CREDS]}. Continuing anyway..."
            )

        client = OpenAI(
            api_key=HOST_TO_CREDS.get(host),
            base_url=self.base_url,
            webhook_secret=HOST_TO_CREDS.get(host),
        )

        logging.debug(f"Generating streaming chat completion for model `{body.model}`")

        messages = _get_messages(body)

        # TODO: backoff and error handling
        # TODO: do we need to pass more metadata from the Open Responses request in here?
        stream = client.chat.completions.create(
            model=str(body.model),
            messages=messages,
            stream=True,
            stream_options={"include_usage": True} if body.stream_options else None,
            max_completion_tokens=int(body.max_output_tokens) if body.max_output_tokens else None,
        )

        logging.debug("Successfully initiated streaming chat completion")
        return convert_chat_completion_stream_to_sse(stream, include_usage=True)


def _get_messages(body: CreateResponseBody) -> list[dict]:
    """
    Get the list of messages from the body input parameter and convert
    to valid messages for the OpenAI Chat interface.

    TODO: Does not support functions, this is a minimal implementation optimized
          around text-based user messages.
    """
    body_input = body.input or []
    if isinstance(body_input, str):
        body_input = [UserMessageItemParam(type="message", content=Content1(body_input), role=Role.user)]

    messages = []
    for message in body_input:
        if isinstance(message, tuple):
            message_content = message[1]
            message_role = Role.user
        else:
            message_content = getattr(message, "content", "")
            message_role = getattr(message, "role", Role.user)

        messages.append({"role": message_role, "content": message_content})
    return messages
