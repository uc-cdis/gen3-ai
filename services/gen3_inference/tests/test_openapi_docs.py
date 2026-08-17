"""Tests that the generated OpenAPI spec stays fit for a public, user-facing audience."""

from common.fastapi.testing import OpenApiDocsContract
from fastapi import FastAPI

from gen3_inference.main import get_app as get_service_app


class TestOpenApiDocs(OpenApiDocsContract):
    """Applies the shared OpenAPI docs contract to this service."""

    authenticated_path_prefixes = ("/v1",)

    @staticmethod
    def get_app() -> FastAPI:
        """
        Return this service's app.

        Returns:
            FastAPI: The gen3_inference app.
        """
        return get_service_app()
