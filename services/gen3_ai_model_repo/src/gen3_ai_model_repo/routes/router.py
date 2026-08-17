"""Router configuration for the Gen3 AI model repo service."""

from fastapi import APIRouter

from common.fastapi.routes.common import common_router
from gen3_ai_model_repo.routes.ai_models_files import ai_models_files_router
from gen3_ai_model_repo.routes.ai_models_repositories import ai_models_repositories_router
from gen3_ai_model_repo.routes.ai_models_uploads import ai_models_uploads_router
from gen3_ai_model_repo.routes.storage import storage_router

route_aggregator = APIRouter()
route_aggregator.include_router(common_router)
route_aggregator.include_router(ai_models_files_router)
route_aggregator.include_router(ai_models_repositories_router)
route_aggregator.include_router(ai_models_uploads_router)
route_aggregator.include_router(storage_router)
