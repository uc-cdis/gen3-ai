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

from gen3_inference import config
from gen3_inference.config import logging
from gen3_inference.errors import (
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_NOT_FOUND,
)
from gen3_inference.inference_protocols.base import InferenceProtocolClient
from gen3_inference.inference_protocols.kserve_v2 import KServev2Client
from gen3_inference.inference_protocols.openresponses import OpenResponsesClient
from gen3_inference.types import OpenResponsesError

responses_router = APIRouter()


@responses_router.post(
    "/responses",
    response_model=ResponseResource,
    responses={
        200: {
            "content": {
                "application/json": {"schema": ResponseResource.model_json_schema()},
                "text/event-stream": {
                    "schema": {
                        "oneOf": [
                            ResponseCreatedStreamingEvent.model_json_schema(),
                            ResponseQueuedStreamingEvent.model_json_schema(),
                            ResponseInProgressStreamingEvent.model_json_schema(),
                            ResponseCompletedStreamingEvent.model_json_schema(),
                            ResponseFailedStreamingEvent.model_json_schema(),
                            ResponseIncompleteStreamingEvent.model_json_schema(),
                            ResponseOutputItemAddedStreamingEvent.model_json_schema(),
                            ResponseOutputItemDoneStreamingEvent.model_json_schema(),
                            ResponseReasoningSummaryPartAddedStreamingEvent.model_json_schema(),
                            ResponseReasoningSummaryPartDoneStreamingEvent.model_json_schema(),
                            ResponseContentPartAddedStreamingEvent.model_json_schema(),
                            ResponseContentPartDoneStreamingEvent.model_json_schema(),
                            ResponseOutputTextDeltaStreamingEvent.model_json_schema(),
                            ResponseOutputTextDoneStreamingEvent.model_json_schema(),
                            ResponseRefusalDeltaStreamingEvent.model_json_schema(),
                            ResponseRefusalDoneStreamingEvent.model_json_schema(),
                            ResponseReasoningDeltaStreamingEvent.model_json_schema(),
                            ResponseReasoningDoneStreamingEvent.model_json_schema(),
                            ResponseReasoningSummaryDeltaStreamingEvent.model_json_schema(),
                            ResponseReasoningSummaryDoneStreamingEvent.model_json_schema(),
                            ResponseOutputTextAnnotationAddedStreamingEvent.model_json_schema(),
                            ResponseFunctionCallArgumentsDeltaStreamingEvent.model_json_schema(),
                            ResponseFunctionCallArgumentsDoneStreamingEvent.model_json_schema(),
                            ErrorStreamingEvent.model_json_schema(),
                        ]
                    }
                },
            }
        }
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
    logging.debug(f"Found model, info: {ai_model_info}")

    # this will prefer using Open Responses inference protocol if available
    inference_protocol_client = await get_inference_protocol_client(ai_model_info.get("inference_protocol_clients", []))
    logging.debug(f"Using inference protocol: `{inference_protocol_client.NAME}`")

    if not body.stream:
        # ResponseResource as JSON
        response = await inference_protocol_client.generate_non_streaming_response(body=body)
    else:
        # text/event-stream of streaming events
        response = inference_protocol_client.generate_streaming_response(body=body)

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
    # hit GET {config.GEN3_AI_MODEL_REPO_URL}/ai-models/{model_name} to check if model is available at this domain
    # hit with oauth2 client credentials

    # 2. is model available at configured trusted domains?
    # check config for other domains, for each one, hit GET {that domain}/ai-models/{model_name}
    # with oauth2 client credentials
    #
    # 3. auth user

    # Check if model is available at the primary domain
    is_model_available = False

    primary_url = f"{config.GEN3_AI_MODEL_REPO_URL}/ai-models/{ai_model}"
    async with httpx.AsyncClient() as client:
        # TODO: use auth: OAUTH2_CLIENT_CREDENTIALS

        # TODO: try and backoff
        primary_resp = await client.get(primary_url)
        # primary_resp = httpx.Response(
        #     json={
        #         "primary_url": primary_url,
        #         "ai_model_info": {
        #             "name": ai_model,
        #             "version": "llama3.2:latest",
        #             "type": "llama3.2:latest",
        #             "description": "llama3.2:latest",
        #             "url": "llama3.2:latest",
        #             "tags": ["llama3.2:latest"],
        #             "inference_protocol_clients": ["kserve_v2", "openresponses", "kserve_v1"],
        #         },
        #     },
        #     status_code=200,
        # )

        if primary_resp.status_code == 404:
            # not found in primary, so check if model is available at any of the configured trusted domains
            trusted_domains = list(config.ALLOWED_GEN3_INFERENCE_HOSTS)
            for domain in trusted_domains:
                # TODO: use auth: OAUTH2_CLIENT_CREDENTIALS
                url = f"{domain.rstrip('/')}/ai-models/{ai_model}"
                primary_resp = await client.get(url)
                if primary_resp.status_code == 200:
                    is_model_available = True
                    break
        else:
            is_model_available = True

    if not is_model_available:
        error = OpenResponsesError(
            type=ERROR_TYPE_NOT_FOUND,
            code=ERROR_TYPE_NOT_FOUND,
            message=f"The requested model '{ai_model}' was not found on any trusted domain.",
        )
        raise HTTPException(status_code=404, detail=error.to_json())

    ai_model_info = primary_resp.json().get("ai_model_info")

    if not ai_model_info:
        error = OpenResponsesError(
            type=ERROR_TYPE_NOT_FOUND,
            code=ERROR_TYPE_NOT_FOUND,
            message="The requested model information was not found. Choose a different model.",
        )
        raise HTTPException(status_code=400, detail=error.to_json())

    return ai_model_info


async def get_inference_protocol_client(all_model_inference_protocol_client_names: list[str]):
    """
    Get the client class given a list of all available inference protocol client names for a
    given model. This will select the appropriate one or raise an error.

    Args:
        all_model_inference_protocol_client_names list[str]: A list of all available inference
            protocol client names for a given model
    """
    inference_protocol_client: InferenceProtocolClient | None = None

    if OpenResponsesClient.NAME in all_model_inference_protocol_client_names:
        inference_protocol_client = OpenResponsesClient()
    elif KServev2Client.NAME in all_model_inference_protocol_client_names:
        inference_protocol_client = KServev2Client(base_url="", model_name="")
    else:
        error = OpenResponsesError(
            type=ERROR_TYPE_INVALID_REQUEST,
            code=ERROR_TYPE_INVALID_REQUEST,
            message="None of the inference protocols for the AI model are supported. Choose a different model.",
        )
        raise HTTPException(status_code=400, detail=error.to_json())

    return inference_protocol_client
