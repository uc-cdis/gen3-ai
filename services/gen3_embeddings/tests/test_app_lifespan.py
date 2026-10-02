"""
Tests for the app's ownership of the database connection pool.

The pool lives on `app.state.db_pool`, created and closed by the lifespan handler, rather
than in a module-level global. These tests pin that contract down: without them, the pool
could go back to being created per-request, or be leaked when startup fails, and every other
test in this suite would still pass.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gen3_embeddings import main as main_module
from gen3_embeddings.dependencies import db_pool_from_request


def test_lifespan_puts_a_usable_pool_on_app_state(app):
    """Startup creates the pool, and it is live for the duration of the lifespan."""
    assert getattr(app.state, "db_pool", None) is None, "pool exists before startup ran"

    with TestClient(app):
        pool = app.state.db_pool
        assert pool is not None
        assert not pool.is_closing()

    assert pool.is_closing(), "shutdown left the pool open"
    assert app.state.db_pool is None


def test_every_request_shares_the_one_pool(app):
    """
    Requests read the pool off the app rather than building their own.

    This is the property the module-level global used to provide by accident, and the reason
    `create_pool` is not cached: a per-request pool would exhaust Postgres' connection slots.
    """
    with TestClient(app) as client:
        pool = app.state.db_pool

        for _ in range(3):
            assert client.get("/_status/").status_code == 200

        assert app.state.db_pool is pool


def test_a_failed_startup_does_not_leak_the_pool(app, monkeypatch):
    """
    When a startup check rejects the database, the pool it was checking is still closed.

    The checks run after the pool is created, so an abort between the two would otherwise
    strand PGPOOL_MIN_SIZE connections for as long as the process lives -- and a crash-looping
    pod would strand a set per restart.
    """

    async def _reject(conn):
        raise Exception("simulated RLS check failure")

    monkeypatch.setattr(main_module, "check_rls_is_enabled", _reject)

    created = []
    real_create_pool = main_module.create_pool

    async def _capturing_create_pool():
        pool = await real_create_pool()
        created.append(pool)
        return pool

    monkeypatch.setattr(main_module, "create_pool", _capturing_create_pool)

    with pytest.raises(Exception, match="simulated RLS check failure"), TestClient(app):
        pass  # pragma: no cover - startup raises before the body runs

    assert len(created) == 1, "startup did not create exactly one pool"
    assert created[0].is_closing(), "failed startup leaked the pool"


def test_a_request_without_a_lifespan_is_an_error_not_a_new_pool():
    """
    Reading the pool fails loudly when the lifespan never ran.

    Falling back to creating one here is what we removed: it would race to build duplicate
    pools, and would serve traffic against a database whose row-level security was never
    verified, because that verification only happens on the startup path.
    """
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    with pytest.raises(Exception, match="lifespan did not run"):
        db_pool_from_request(request)
