from typing import Union
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException
from openresponses_types.types import (
    # Request body
    CreateResponseBody,
    ErrorStreamingEvent,
    ResponseCompletedStreamingEvent,
    ResponseContentPartAddedStreamingEvent,
    ResponseContentPartDoneStreamingEvent,
    # Streaming events
    ResponseCreatedStreamingEvent,
    ResponseFailedStreamingEvent,
    ResponseFunctionCallArgumentsDeltaStreamingEvent,
    ResponseFunctionCallArgumentsDoneStreamingEvent,
    ResponseIncompleteStreamingEvent,
    ResponseInProgressStreamingEvent,
    ResponseOutputItemAddedStreamingEvent,
    ResponseOutputItemDoneStreamingEvent,
    ResponseOutputTextAnnotationAddedStreamingEvent,
    ResponseOutputTextDeltaStreamingEvent,
    ResponseOutputTextDoneStreamingEvent,
    ResponseQueuedStreamingEvent,
    ResponseReasoningDeltaStreamingEvent,
    ResponseReasoningDoneStreamingEvent,
    ResponseReasoningSummaryDeltaStreamingEvent,
    ResponseReasoningSummaryDoneStreamingEvent,
    ResponseReasoningSummaryPartAddedStreamingEvent,
    ResponseReasoningSummaryPartDoneStreamingEvent,
    ResponseRefusalDeltaStreamingEvent,
    ResponseRefusalDoneStreamingEvent,
    # Core response resource
    ResponseResource,
)
from starlette import status

from gen3_inference.config import (
    ALLOWED_GEN3_INFERENCE_HOSTS,
    GEN3_AI_MODEL_REPO_URL,
    MOCK_AI_MODEL_REPO_REPONSE,
    logging,
)
from gen3_inference.errors import (
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_NOT_FOUND,
)
from gen3_inference.inference_protocols.base import InferenceProtocolClient
from gen3_inference.inference_protocols.openai_chat import OpenaiChat
from gen3_inference.inference_protocols.openresponses import OpenResponsesClient
from gen3_inference.types import OpenResponsesError

responses_router = APIRouter()

AllResponseTypes = Union[
    # non-streaming
    ResponseResource,
    # streaming events
    ResponseCreatedStreamingEvent,
    ResponseQueuedStreamingEvent,
    ResponseInProgressStreamingEvent,
    ResponseCompletedStreamingEvent,
    ResponseFailedStreamingEvent,
    ResponseIncompleteStreamingEvent,
    ResponseOutputItemAddedStreamingEvent,
    ResponseOutputItemDoneStreamingEvent,
    ResponseReasoningSummaryPartAddedStreamingEvent,
    ResponseReasoningSummaryPartDoneStreamingEvent,
    ResponseContentPartAddedStreamingEvent,
    ResponseContentPartDoneStreamingEvent,
    ResponseOutputTextDeltaStreamingEvent,
    ResponseOutputTextDoneStreamingEvent,
    ResponseRefusalDeltaStreamingEvent,
    ResponseRefusalDoneStreamingEvent,
    ResponseReasoningDeltaStreamingEvent,
    ResponseReasoningDoneStreamingEvent,
    ResponseReasoningSummaryDeltaStreamingEvent,
    ResponseReasoningSummaryDoneStreamingEvent,
    ResponseOutputTextAnnotationAddedStreamingEvent,
    ResponseFunctionCallArgumentsDeltaStreamingEvent,
    ResponseFunctionCallArgumentsDoneStreamingEvent,
    ErrorStreamingEvent,
]


@responses_router.post(
    "/v1/responses",
    response_model=AllResponseTypes,
    summary="Standard Open Responses API",
    description=(
        "See official spec for details ([https://openresponses.org](https://openresponses.org)). "
        "This OpenAPI spec here is auto-generated."
    ),
    responses={
        status.HTTP_200_OK: {
            "content": {
                "application/json": {},
                "text/event-stream": {},
            }
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Bad request, unable to get response"},
        status.HTTP_401_UNAUTHORIZED: {"description": "User unauthenticated"},
        status.HTTP_403_FORBIDDEN: {"description": "User does not have access"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Something went wrong internally when processing the request"
        },
    },
    tags=["Inference"],
)
async def create_response(
    body: CreateResponseBody,
):
    """
    Implements the /responses endpoint defined in the Open Responses OpenAPI spec,
    using the models from openresponses_types.types
    """
    # this will search "locally" first, then try other configured hosts
    ai_model_info = await get_ai_model_info(body)
    ai_model_url = ai_model_info.get("url")
    logging.debug(f"Found model at `{ai_model_url}`, info: `{ai_model_info}`")

    # this will prefer using Open Responses inference protocol if available
    inference_protocol_client = await get_inference_protocol_client(
        ai_model_info.get("metadata", {}).get("inference_protocol_clients", []), ai_model_url
    )
    logging.debug(f"Using inference protocol: `{inference_protocol_client.NAME}`")

    if not body.stream:
        # ResponseResource as JSON
        response = await inference_protocol_client.generate_non_streaming_response(body=body, model_info=ai_model_info)
    else:
        # text/event-stream of streaming events
        response = inference_protocol_client.generate_streaming_response(body=body, model_info=ai_model_info)

    return response


async def get_ai_model_info(body: CreateResponseBody) -> dict:
    """
    Get AI Model Info by talking with local and connected Gen3 AI Model Repos

    Args:
        body CreateResponseBody: the request body containing model info
    """
    ai_model = body.model
    if not ai_model:
        error = OpenResponsesError(
            type=ERROR_TYPE_INVALID_REQUEST, code=ERROR_TYPE_INVALID_REQUEST, message="Must request a model."
        )
        raise HTTPException(status_code=400, detail=error.to_json())

    # 1. is model available at this domain?
    # hit GET {GEN3_AI_MODEL_REPO_URL}/ai-models/{model_name} to check if model is available at this domain
    # hit with oauth2 client credentials

    # 2. is model available at configured trusted domains?
    # check config for other domains, for each one, hit GET {that domain}/ai-models/{model_name}
    # with oauth2 client credentials
    #
    # 3. auth user

    # Check if model is available at the primary domain
    is_model_available = False
    ai_model_url = None

    primary_url = f"{GEN3_AI_MODEL_REPO_URL}/ai-models/{ai_model}"
    async with httpx.AsyncClient() as client:
        response = None
        # TODO: use auth: HOST_TO_CREDS

        # TODO: FOR TESTING, REMOVE THIS
        if MOCK_AI_MODEL_REPO_REPONSE:
            response = httpx.Response(
                # json={
                #     "name": ai_model,
                #     "url": "https://api.openai.com/v1/",
                #     "version": "foobar",
                #     "type": "foobar",
                #     "description": "foobar",
                #     "tags": ["foobar"],
                #     "inference_protocol_clients": ["openai_chat"],  # "kserve_v2", "openresponses",
                # },
                json={
                    "name": ai_model,
                    "url": "http://localhost:11434/v1/",
                    "version": "1",
                    "type": "model",
                    "description": "description",
                    "tags": ["foobar"],
                    "inference_protocol_clients": ["openresponses"],  # "openai_chat", "kserve_v2", "openresponses",
                },
                status_code=200,
            )

        if not response:
            # TODO: remove conditional above and add retry and backoff
            logging.debug(f"Checking if model is available at primary domain: {primary_url}")
            response = await client.get(primary_url)

        if response.status_code == 404:
            # not found in primary, so check if model is available at any of the configured trusted domains
            trusted_domains = list(ALLOWED_GEN3_INFERENCE_HOSTS)
            for domain in trusted_domains:
                # TODO: use auth: HOST_TO_CREDS
                url = f"{domain.rstrip('/')}/ai-models/{ai_model}"
                response = await client.get(url)
                if response.status_code == 200:
                    is_model_available = True
                    ai_model_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                    break
        else:
            is_model_available = True
            ai_model_url = f"{urlparse(primary_url).scheme}://{urlparse(primary_url).netloc}"

    if not is_model_available:
        error = OpenResponsesError(
            type=ERROR_TYPE_NOT_FOUND,
            code=ERROR_TYPE_NOT_FOUND,
            message=f"The requested model '{ai_model}' was not found on any trusted domain.",
        )
        raise HTTPException(status_code=404, detail=error.to_json())

    if not response.json():
        error = OpenResponsesError(
            type=ERROR_TYPE_NOT_FOUND,
            code=ERROR_TYPE_NOT_FOUND,
            message="The requested model information was not found. Choose a different model.",
        )
        raise HTTPException(status_code=400, detail=error.to_json())

    if "url" in response.json() and response.json().get("url"):
        new_url = response.json().get("url")
        new_domain = f"{urlparse(new_url).scheme}://{urlparse(new_url).netloc}"
        trusted_domains = {f"{urlparse(url).scheme}://{urlparse(url).netloc}" for url in ALLOWED_GEN3_INFERENCE_HOSTS}
        if new_domain not in trusted_domains:
            raise Exception(
                f"AI Model provided a url `{new_url}` whose domain `{new_domain}` is NOT in the ALLOWED_GEN3_INFERENCE_HOSTS: `{ALLOWED_GEN3_INFERENCE_HOSTS}`"
            )

        logging.info(f"using AI model provided url: {new_url}")
        ai_model_url = new_url

    ai_model_info = {"url": ai_model_url, "metadata": response.json()}

    return ai_model_info


async def get_inference_protocol_client(all_model_inference_protocol_client_names: list[str], ai_model_url: str | None):
    """
    Get the client class given a list of all available inference protocol client names for a
    given model. This will select the appropriate one or raise an error.

    WARNING: Validate ai_model_url is allowed before passing it in here, this function
             blindly adds that to the client.

    Args:
        all_model_inference_protocol_client_names list[str]: A list of all available inference
            protocol client names for a given model
    """
    inference_protocol_client: InferenceProtocolClient | None = None

    # TODO: is it worth caching instances of these for different model URLs?
    if OpenResponsesClient.NAME in all_model_inference_protocol_client_names:
        inference_protocol_client = OpenResponsesClient(base_url=ai_model_url)
    elif OpenaiChat.NAME in all_model_inference_protocol_client_names:
        inference_protocol_client = OpenaiChat(base_url=ai_model_url)
    else:
        error = OpenResponsesError(
            type=ERROR_TYPE_INVALID_REQUEST,
            code=ERROR_TYPE_INVALID_REQUEST,
            message="None of the inference protocols for the AI model are supported. Choose a different model.",
        )
        raise HTTPException(status_code=400, detail=error.to_json())

    return inference_protocol_client
