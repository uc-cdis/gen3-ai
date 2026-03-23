import time
from importlib.metadata import version

from fastapi import FastAPI, HTTPException, Request

from common.auth import get_user_id
from common.config import logging
from common.fastapi.routes.common import common_router
from common.metrics import get_metrics_client
from gen3_inference import config
from gen3_inference.metrics import InferenceServiceMetrics
from gen3_inference.routes.basic import basic_router
from gen3_inference.routes.responses import responses_router


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
    async def middleware_log_response_and_api_metric(request: Request, call_next) -> None:
        """
        This FastAPI middleware effectively allows pre and post logic to a request.

        We are using this to log the response consistently across defined endpoints (including execution time).

        Args:
            request (Request): the incoming HTTP request
            call_next (Callable): function to call (this is handled by FastAPI's middleware support)
        """
        start_time = time.perf_counter()
        response = await call_next(request)
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
                f"Could not retrieve user_id. Error: {exc}. For logging and metrics, setting user_id to 'Unknown'"
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

    return fastapi_app


app_instance = get_app()
