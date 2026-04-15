from urllib.parse import urlparse

import openai
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from openresponses_types import (
    CreateResponseBody,
)

from gen3_inference.config import HOST_TO_CREDS, logging
from gen3_inference.errors import ERROR_TYPE_NOT_FOUND
from gen3_inference.inference_protocols.base import InferenceProtocolClient
from gen3_inference.inference_protocols.utils.openai_responses_to_openresponses import (
    openai_response_to_openresponses,
    openai_streaming_response_to_openresponses,
)
from gen3_inference.types import OpenResponsesError


class OpenResponsesClient(InferenceProtocolClient):
    NAME = "openresponses"

    def __init__(self, base_url: str | None = None):
        super().__init__(base_url=base_url)

    async def generate_non_streaming_response(self, body: CreateResponseBody, model_info: dict) -> JSONResponse:
        return self._create_non_streaming_response(body, model_info)

    def generate_streaming_response(self, body: CreateResponseBody, model_info: dict) -> StreamingResponse:
        return self._create_streaming_response(body, model_info)

    def _create_non_streaming_response(self, body: CreateResponseBody, model_info: dict) -> JSONResponse:
        host = str(urlparse(self.base_url).hostname) or ""
        if host not in HOST_TO_CREDS:
            logging.warning(
                f"{host} is not in HOST_TO_CREDS, unable to retrieve creds. Valid hosts: {[host for host in HOST_TO_CREDS]}. Continuing anyway..."
            )

        client = OpenAI(
            # TODO: FIXME: actually add thes from the model info
            # For local testing: ollama supports Responses
            # https://docs.ollama.com/api/openai-compatibility#simple-/v1/responses-example
            api_key=HOST_TO_CREDS.get(host),
            base_url=self.base_url,
            webhook_secret=HOST_TO_CREDS.get(host),
        )

        logging.debug("successfully setup client. sending responses request...")

        # TODO: backoff and retry
        try:
            response = client.responses.create(
                # TODO: probably need to pass in a ton more stuff from the request
                model=str(body.model),
                instructions=body.instructions,
                # TODO: convert body to Openai format
                input=body.input,
                stream=False,
            )
        except openai.NotFoundError:
            error = OpenResponsesError(
                type=ERROR_TYPE_NOT_FOUND,
                code=ERROR_TYPE_NOT_FOUND,
                message="The requested model was not found.",
            )
            raise HTTPException(status_code=404, detail=error.to_json())
        except Exception as exc:
            logging.error(f"{type(exc).__name__}: {exc}", exc_info=True)
            raise HTTPException(status_code=503, detail="Unable to connect to inference server")

        logging.debug("successfully got response. parsing response...")

        standard_response = openai_response_to_openresponses(
            response, metadata={"inference_protocol_client": self.NAME}
        )

        return JSONResponse(content=standard_response.model_dump())

    # TODO: FIXME: Actually implement streaming response
    def _create_streaming_response(self, body: CreateResponseBody, model_info: dict) -> StreamingResponse:
        host = str(urlparse(self.base_url).hostname) or ""
        if host not in HOST_TO_CREDS:
            logging.warning(
                f"{host} is not in HOST_TO_CREDS, unable to retrieve creds. Valid hosts: {[host for host in HOST_TO_CREDS]}. Continuing anyway..."
            )

        client = OpenAI(
            # TODO: FIXME: actually add thes from the model info
            api_key=HOST_TO_CREDS.get(host),
            base_url=self.base_url,
            webhook_secret=HOST_TO_CREDS.get(host),
        )

        response = client.responses.create(
            # TODO: probably need to pass in a ton more stuff from the request
            model=str(body.model),
            instructions=body.instructions,
            # TODO: convert body to Openai format
            input=body.input,
            stream=True,
        )
        standard_streaming_response = openai_streaming_response_to_openresponses(
            response, metadata={"inference_protocol_client": self.NAME}
        )

        return standard_streaming_response
