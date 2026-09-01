import asyncio
import os
import secrets
import string
import subprocess
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio
from dotenv import load_dotenv
from fastapi.testclient import TestClient

load_dotenv(Path(__file__).parent / ".env")


def _quote_ident(value: str) -> str:
    """Double-quote a Postgres identifier, escaping any embedded quotes."""
    return '"' + value.replace('"', '""') + '"'


def _random_string(n: int = 10) -> str:
    """Generate a random lowercase alphanumeric string of length n."""
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


def _unique_db_name() -> str:
    """Generate a unique name for this test session's scratch database."""
    return f"gen3_embeddings_test_db_{_random_string()}"


def _unique_user_name() -> str:
    """Generate a unique name for this test session's Postgres app user."""
    return f"gen3_embeddings_app_user_{_random_string(8)}"


def _unique_password() -> str:
    """Generate a random password for this test session's Postgres app user."""
    return secrets.token_urlsafe(32)


def _pg_host() -> str:
    """
    Resolve the Postgres host for tests.

    Prefers TEST_PGHOST, falls back to the standard PGHOST, then to localhost, so tests
    run with no configuration at all. See tests/.env.example.
    """
    return os.environ.get("TEST_PGHOST") or os.environ.get("PGHOST") or "localhost"


def _pg_port() -> str:
    """Resolve the Postgres port for tests, preferring TEST_PGPORT, then PGPORT, then 5432."""
    return os.environ.get("TEST_PGPORT") or os.environ.get("PGPORT") or "5432"


def _admin_dsn(db_name: str = "postgres") -> str:
    """
    Build a DSN for connecting to `db_name` as the Postgres admin user.

    The admin user must be a superuser: the test setup creates databases and roles, and
    the migrations create the `vector` extension.
    """
    user = os.environ.get("TEST_PGADMIN_USER") or os.environ.get("PGUSER") or "postgres"
    password = os.environ.get("TEST_PGADMIN_PASSWORD") or os.environ.get("PGPASSWORD") or "postgres"
    return f"postgresql://{user}:{password}@{_pg_host()}:{_pg_port()}/{db_name}"


def _app_dsn(user: str, password: str, db_name: str) -> str:
    """Build a DSN for connecting to `db_name` as the given app user."""
    return f"postgresql://{user}:{password}@{_pg_host()}:{_pg_port()}/{db_name}"


@pytest.fixture(scope="session")
def event_loop():
    """Provide a session-scoped asyncio event loop shared by all async fixtures and tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_database():
    """
    Create an isolated Postgres database and app user for the test session, migrated to the
    latest schema, and drop them afterward (unless KEEP_TEST_DB is set).

    Yields:
        dict: Connection info for the test database (db_name, app_user, app_password,
            admin_dsn, app_dsn).
    """
    keep_db = os.environ.get("KEEP_TEST_DB", "").lower() in {"1", "true", "yes"}

    db_name = _unique_db_name()
    app_user = _unique_user_name()
    app_password = _unique_password()

    admin_conn = await asyncpg.connect(_admin_dsn("postgres"))
    try:
        # create database if not exist
        exists = await admin_conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            db_name,
        )
        if not exists:
            await admin_conn.execute(f"CREATE DATABASE {_quote_ident(db_name)}")

        user_exists = await admin_conn.fetchval(
            "SELECT 1 FROM pg_user WHERE usename = $1",
            app_user,
        )
        if not user_exists:
            await admin_conn.execute(f"CREATE USER {_quote_ident(app_user)} WITH PASSWORD '{app_password}'")

        await admin_conn.execute(
            f"GRANT ALL PRIVILEGES ON DATABASE {_quote_ident(db_name)} TO {_quote_ident(app_user)}"
        )
        await admin_conn.execute(f"ALTER ROLE {_quote_ident(app_user)} WITH LOGIN")
    finally:
        await admin_conn.close()

    admin_db_conn = await asyncpg.connect(_admin_dsn(db_name))
    try:
        # mimic Helm create service user behavior
        await admin_db_conn.execute(f"ALTER SCHEMA public OWNER TO {_quote_ident(app_user)}")
        await admin_db_conn.execute(f"GRANT ALL ON SCHEMA public TO {_quote_ident(app_user)}")
        await admin_db_conn.execute(f"GRANT ALL ON ALL TABLES IN SCHEMA public TO {_quote_ident(app_user)}")
        await admin_db_conn.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {_quote_ident(app_user)}"
        )
    finally:
        await admin_db_conn.close()

    root_dir = Path(__file__).resolve().parents[1]
    migrations_dir = root_dir / "db/migrations"

    # run migrations as admin because migration creates extension(s)
    dbmate_env = os.environ.copy()
    dbmate_env["DATABASE_URL"] = _admin_dsn(db_name) + "?sslmode=disable"

    subprocess.run(
        ["dbmate", "--migrations-dir", str(migrations_dir), "migrate"],
        check=True,
        cwd=root_dir,
        env=dbmate_env,
    )

    yield {
        "db_name": db_name,
        "app_user": app_user,
        "app_password": app_password,
        "admin_dsn": _admin_dsn(db_name),
        "app_dsn": _app_dsn(app_user, app_password, db_name),
    }

    if keep_db:
        print(f"KEEP_TEST_DB=1 set; leaving database={db_name}, user={app_user}")
        return

    admin_conn = await asyncpg.connect(_admin_dsn("postgres"))
    try:
        await admin_conn.execute(
            """
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = $1
              AND pid <> pg_backend_pid()
            """,
            db_name,
        )
        await admin_conn.execute(f"DROP DATABASE IF EXISTS {_quote_ident(db_name)}")
        await admin_conn.execute(f"DROP ROLE IF EXISTS {_quote_ident(app_user)}")
    finally:
        await admin_conn.close()


@pytest_asyncio.fixture
async def reset_db(test_database):
    """Truncate all tables between tests so each test starts from an empty database."""
    conn = await asyncpg.connect(test_database["app_dsn"])
    try:
        await conn.execute("TRUNCATE TABLE embeddings_vector, embeddings_halfvec, collections CASCADE")
    finally:
        await conn.close()


@pytest.fixture
def app(test_database, monkeypatch):
    """
    Build a fresh FastAPI app wired to the test database, with DEBUG_SKIP_AUTH enabled.

    Yields:
        FastAPI: The app instance; its connection pool is closed on teardown.
    """
    from gen3_embeddings import config
    from gen3_embeddings.database import db as db_module
    from gen3_embeddings.main import get_app

    monkeypatch.setattr(config, "DB_CONNECTION_STRING", test_database["app_dsn"])
    monkeypatch.setattr(config, "DEBUG_SKIP_AUTH", True)

    db_module._pool = None

    app = get_app()
    yield app

    async def _close_pool():
        if db_module._pool is not None:
            await db_module._pool.close()
            db_module._pool = None

    asyncio.run(_close_pool())


@pytest.fixture
def client(app, reset_db):
    """Provide a TestClient for the app, with the database reset before each test."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def allow_authz(monkeypatch):
    """
    Patch authz resolution to simulate the caller having access to specific collections.

    Grants every action on the named collections, which is what most tests want. Tests that
    care about the distinction between actions should use `allow_authz_per_action` instead.

    Example:
        allow_authz("alpha", "beta")
    """
    from gen3_embeddings import dependencies as dependencies_module

    def _apply(*collection_names: str):
        allowed = [f"/vectorstore/collections/{name}" for name in collection_names]

        async def fake_get_allowed_authz_for_request(request, method, authz_config):
            return allowed

        monkeypatch.setattr(dependencies_module, "get_allowed_authz_for_request", fake_get_allowed_authz_for_request)
        return allowed

    return _apply


@pytest.fixture
def allow_authz_paths(monkeypatch):
    """
    Patch authz resolution with literal resource paths, of any shape.

    Unlike `allow_authz`, this does not prepend the collection base path. An embedding's
    `authz` is an arbitrary stored string, so testing that requires granting paths that are
    not collection-shaped.

    Example:
        allow_authz_paths("/vectorstore/collections/docs", "/programs/foo/projects/bar")
    """
    from gen3_embeddings import dependencies as dependencies_module

    def _apply(*paths: str):
        allowed = list(paths)

        async def fake_get_allowed_authz_for_request(request, method, authz_config):
            return allowed

        monkeypatch.setattr(dependencies_module, "get_allowed_authz_for_request", fake_get_allowed_authz_for_request)
        return allowed

    return _apply


@pytest.fixture
def allow_authz_per_action(monkeypatch):
    """
    Patch authz resolution with a different grant per action.

    This is what makes it testable that a route authorizes the action it actually performs
    rather than the one its HTTP verb implies: a caller granted only `read` must still be
    able to POST to a search or bulk-read endpoint.

    Example:
        allow_authz_per_action(read=("alpha",), create=())
    """
    from gen3_embeddings import dependencies as dependencies_module

    def _apply(**by_action: tuple[str, ...]):
        paths_by_action = {
            action: [f"/vectorstore/collections/{name}" for name in names] for action, names in by_action.items()
        }

        async def fake_get_allowed_authz_for_request(request, method, authz_config):
            return paths_by_action.get(method, [])

        monkeypatch.setattr(dependencies_module, "get_allowed_authz_for_request", fake_get_allowed_authz_for_request)
        return paths_by_action

    return _apply
