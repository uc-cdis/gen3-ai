import time
from importlib.metadata import version

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from common.auth import get_user_id
from common.config import logging
from common.fastapi.routes.common import common_router
from common.metrics import get_metrics_client
from gen3_inference import config
from gen3_inference.errors import ERROR_TYPE_INVALID_REQUEST, ERROR_TYPE_SERVER_ERROR
from gen3_inference.metrics import InferenceServiceMetrics
from gen3_inference.routes.basic import basic_router
from gen3_inference.routes.responses import responses_router
from gen3_inference.types import OpenResponsesError


def get_app() -> FastAPI:
    """
    Return the web framework app object after adding routes

    Returns:
        FastAPI: The FastAPI app object
    """
    fastapi_app = FastAPI(
        title="Gen3 Inference Service",
        version=version("gen3_inference"),
        debug=config.DEBUG,
        root_path=config.URL_PREFIX,
    )
    fastapi_app.include_router(common_router)
    fastapi_app.include_router(basic_router)
    fastapi_app.include_router(responses_router)

    fastapi_app.state.metrics = InferenceServiceMetrics(metrics_client=get_metrics_client(fastapi_app))

    @fastapi_app.middleware("http")
    async def middleware_log_response_and_api_metric(request: Request, call_next):
        """
        This FastAPI middleware effectively allows pre and post logic to a request.

        We are using this to log the response consistently across defined endpoints (including execution time).
        It also ensures an Open Responses compliant error on 500s

        Args:
            request (Request): the incoming HTTP request
            call_next (Callable): function to call (this is handled by FastAPI's middleware support)
        """
        start_time = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            error_output = OpenResponsesError(
                type=ERROR_TYPE_SERVER_ERROR, code=ERROR_TYPE_SERVER_ERROR, message="Internal server error"
            ).to_json()
            logging.error(f"{type(exc).__name__}: {exc}", exc_info=True)
            return JSONResponse(error_output, status_code=500)

        response_time_ms = (time.perf_counter() - start_time) * 1000

        path = request.url.path
        method = request.method

        if path in config.ENDPOINTS_WITHOUT_METRICS:
            return response

        # don't add logs or metrics for the actual metrics gathering endpoint
        try:
            user_id = await get_user_id(request=request)
        except HTTPException as exc:
            logging.debug(
                f"Could not retrieve user_id. Error: '{exc}'. For logging and metrics, setting user_id to 'Unknown'"
            )
            user_id = "Unknown"

        if not getattr(fastapi_app.state, "metrics", None):
            return

        metrics = fastapi_app.state.metrics
        metrics.add_to_api_interaction_counter(
            method=method,
            path=path,
            user_id=user_id,
            response_time_ms=response_time_ms,
            status_code=response.status_code,
        )

        return response

    # override default exception handling to match expected error format
    # described by the Open Responses specification
    @fastapi_app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(exc.detail, status_code=exc.status_code)

    @fastapi_app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        message = "Invalid request, check user input. "
        for error in exc.errors():
            loc = error["loc"]
            filtered_loc = [str(item) for item in loc[1:]] if loc[0] in ("body", "query", "path") else loc
            param = ".".join(filtered_loc)

            param = loc
            message += f"\nField: {param}, Error: {error['msg']}"

        error_output = OpenResponsesError(
            type=ERROR_TYPE_INVALID_REQUEST, code=ERROR_TYPE_INVALID_REQUEST, message=message
        ).to_json()

        # if there's only 1 parameter with an issue, include it
        # the spec is somewhat unclear about how to enumerate multiple parameter issues
        # and this is optional
        if len(exc.errors()) == 1:
            loc = exc.errors()[0]["loc"]
            param = ".".join([str(item) for item in loc])
            error_output["error"].update({"param": param})

        return JSONResponse(error_output, status_code=400)

    return fastapi_app


app_instance = get_app()
