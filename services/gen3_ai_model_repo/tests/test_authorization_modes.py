"""Authorization-mode regression tests using the real route dependency."""

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from gen3_ai_model_repo.routes.ai_models_repositories import ai_models_repositories_router


def _app() -> FastAPI:
    """Build an app without overriding the authorization dependency."""
    app = FastAPI()
    app.include_router(ai_models_repositories_router)
    return app


def test_debug_skip_auth_allows_model_listing_without_token(monkeypatch):
    """Missing-token debug bypass reaches the model listing route."""
    import gen3_ai_model_repo.config as service_config
    import gen3_ai_model_repo.routes.ai_models_repositories as repo_routes
    from common import config as common_config

    async def empty_models(**kwargs):
        del kwargs
        return []

    monkeypatch.setattr(service_config, "DEBUG_SKIP_AUTH", True)
    monkeypatch.setattr(common_config, "DEBUG_SKIP_AUTH", True)
    monkeypatch.setattr(repo_routes, "list_models", empty_models)
    response = TestClient(_app()).get("/api/models")
    assert response.status_code == 200
    assert response.json()["models"] == []


def test_debug_skip_auth_does_not_accept_invalid_token(monkeypatch):
    """A supplied invalid token is still rejected in debug mode."""
    import gen3_ai_model_repo.config as service_config
    from common import auth as common_auth
    from common import config as common_config

    monkeypatch.setattr(service_config, "DEBUG_SKIP_AUTH", True)
    monkeypatch.setattr(common_config, "DEBUG_SKIP_AUTH", True)

    async def arborist_must_not_be_contacted(*args, **kwargs):
        raise AssertionError("Arborist must not receive a malformed token")

    monkeypatch.setattr(common_auth.arborist, "auth_mapping", arborist_must_not_be_contacted)
    response = TestClient(_app()).get("/api/models", headers={"Authorization": "Bearer not-a-jwt"})
    assert response.status_code in {401, 403}


def test_repository_denial_remains_when_debug_bypass_is_disabled(monkeypatch):
    """The real dependency still performs repository authorization normally."""
    import gen3_ai_model_repo.auth as auth
    import gen3_ai_model_repo.config as service_config
    from common import config as common_config

    calls = []

    async def authorize_request(*, authz_resources, authz_service_name, authz_access_method, request):
        del authz_service_name, request
        calls.append((authz_resources, authz_access_method))
        if authz_resources == ["/ai_model_repo/ns/repo"]:
            raise HTTPException(status_code=403, detail="repository denied")

    monkeypatch.setattr(service_config, "DEBUG_SKIP_AUTH", False)
    monkeypatch.setattr(common_config, "DEBUG_SKIP_AUTH", False)
    monkeypatch.setattr(auth, "authorize_request", authorize_request)
    response = TestClient(_app()).get("/api/models/ns/repo")
    assert response.status_code == 403
    assert calls[-1] == (["/ai_model_repo/ns/repo"], "read")


def test_anonymous_bypass_logs_explicit_warning(monkeypatch):
    """Unconditional anonymous bypass is visibly distinct from debug bypass."""
    import asyncio

    from common import auth as common_auth
    from common import config as common_config

    warnings = []
    monkeypatch.setattr(common_config, "ALLOW_ANONYMOUS_ACCESS", True)
    monkeypatch.setattr(common_config, "DEBUG_SKIP_AUTH", False)
    monkeypatch.setattr(common_auth.logging, "warning", warnings.append)
    asyncio.run(
        common_auth.authorize_request(
            authz_resources=["/services/example"],
            authz_service_name="example",
        )
    )
    assert any("ALLOW_ANONYMOUS_ACCESS" in message and "NON-PRODUCTION" in message for message in warnings)
