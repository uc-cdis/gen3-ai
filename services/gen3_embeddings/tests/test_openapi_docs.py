"""Tests that the generated OpenAPI spec stays fit for a public, user-facing audience."""

from fastapi import FastAPI

from common.fastapi.testing import OpenApiDocsContract
from gen3_embeddings.main import get_app as get_service_app


class TestOpenApiDocs(OpenApiDocsContract):
    """Applies the shared OpenAPI docs contract to this service."""

    authenticated_path_prefixes = ("/vectorstore", "/embeddings")

    @staticmethod
    def get_app() -> FastAPI:
        """
        Return this service's app.

        Returns:
            FastAPI: The gen3_embeddings app.
        """
        return get_service_app()
