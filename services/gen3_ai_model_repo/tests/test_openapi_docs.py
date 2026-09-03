"""Tests that the generated OpenAPI spec stays fit for a public, user-facing audience."""

from fastapi import FastAPI

from common.fastapi.testing import OpenApiDocsContract
from gen3_ai_model_repo.main import get_app as get_service_app


class TestOpenApiDocs(OpenApiDocsContract):
    """Applies the shared OpenAPI docs contract to this service."""

    expects_public_operations = True

    @staticmethod
    def get_app() -> FastAPI:
        """
        Return this service's app.

        Returns:
            FastAPI: The gen3_ai_model_repo app.
        """
        return get_service_app()
