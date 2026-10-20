"""
PostgreSQL integration tests for direct-upload transaction guarantees.

Run with ``GEN3_AI_MODEL_REPO_TEST_DATABASE_URL`` pointing at a disposable
database after applying the model-repository migrations.
"""

import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from uuid import uuid4

import pytest
import pytest_asyncio

asyncpg = pytest.importorskip("asyncpg")


@pytest_asyncio.fixture
async def pg_connection(monkeypatch):
    """Connect to the explicitly configured disposable PostgreSQL database."""
    dsn = os.environ.get("GEN3_AI_MODEL_REPO_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("set GEN3_AI_MODEL_REPO_TEST_DATABASE_URL for PostgreSQL integration tests")
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)
    try:
        async with pool.acquire() as connection:
            tables = await connection.fetch(
                """SELECT table_name FROM information_schema.tables
                   WHERE table_schema='public' AND table_name = ANY($1::text[])""",
                ["models", "model_upload_intents", "model_revisions", "model_files"],
            )
            found = {row["table_name"] for row in tables}
            missing = {"models", "model_upload_intents", "model_revisions", "model_files"} - found
            if missing:
                pytest.fail(f"database is missing required tables: {sorted(missing)}")
        from gen3_ai_model_repo.database import upload_intents

        async def get_test_pool():
            return pool

        monkeypatch.setattr(upload_intents, "get_db_pool", get_test_pool)
        from gen3_ai_model_repo.routes import ai_models_uploads

        monkeypatch.setattr(ai_models_uploads, "get_db_pool", get_test_pool)
        yield pool
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_second_file_failure_rolls_back_revision_and_intents(pg_connection, monkeypatch):
    """A real completion transaction rolls back when its second file insert fails."""
    import gen3_ai_model_repo.routes.ai_models_uploads as upload_routes
    from gen3_ai_model_repo.models.schemas import DirectUploadCompleteRequest

    namespace = f"tx-{uuid4().hex}"
    repo = "rollback"
    intent_ids = [uuid4(), uuid4()]
    checksum = "a" * 64
    try:
        async with pg_connection.acquire() as connection:
            model_id = await connection.fetchval(
                "INSERT INTO models(namespace,model_name) VALUES($1,$2) RETURNING id", namespace, repo
            )
            for intent_id, name in zip(intent_ids, ("first.bin", "second.bin")):
                await connection.execute(
                    """INSERT INTO model_upload_intents
                       (id,namespace,repo,revision_name,file_name,object_key,checksum_sha256,reserved_bytes,expires_at)
                       VALUES($1,$2,$3,$4,$5,$6,$7,$8,NOW()+INTERVAL '1 hour')""",
                    intent_id,
                    namespace,
                    repo,
                    "main",
                    name,
                    f"{namespace}/{intent_id}/{name}",
                    checksum,
                    1,
                )
            await connection.execute(
                """CREATE OR REPLACE FUNCTION test_fail_second_file() RETURNS trigger LANGUAGE plpgsql AS $$
                   BEGIN IF NEW.file_path='second.bin' THEN RAISE EXCEPTION 'test second file failure'; END IF; RETURN NEW; END $$"""
            )
            await connection.execute(
                "CREATE TRIGGER test_fail_second_file BEFORE INSERT ON model_files FOR EACH ROW EXECUTE FUNCTION test_fail_second_file()"
            )

        class Provider:
            async def get_file_metadata(self, key):
                return {"size": 1, "etag": "etag", "checksum_sha256": checksum}

        monkeypatch.setattr(upload_routes, "get_storage_provider", lambda: Provider())
        with pytest.raises(Exception, match="test second file failure"):
            await upload_routes.complete_upload(
                namespace, repo, DirectUploadCompleteRequest(upload_ids=[str(value) for value in intent_ids])
            )
        async with pg_connection.acquire() as connection:
            assert await connection.fetchval("SELECT COUNT(*) FROM model_revisions WHERE model_id=$1", model_id) == 0
            assert (
                await connection.fetchval(
                    "SELECT COUNT(*) FROM model_files WHERE revision_id NOT IN (SELECT id FROM model_revisions)"
                )
                == 0
            )
            assert (
                await connection.fetchval(
                    "SELECT COUNT(*) FROM model_upload_intents WHERE id=ANY($1::uuid[]) AND completed_at IS NOT NULL",
                    intent_ids,
                )
                == 0
            )
    finally:
        async with pg_connection.acquire() as connection:
            await connection.execute("DROP TRIGGER IF EXISTS test_fail_second_file ON model_files")
            await connection.execute("DROP FUNCTION IF EXISTS test_fail_second_file()")
            await connection.execute("DELETE FROM models WHERE namespace=$1", namespace)


@pytest.mark.asyncio
async def test_intent_completion_failure_rolls_back_all_preceding_writes(pg_connection, monkeypatch):
    """A real completion transaction rolls back when intent completion fails."""
    import gen3_ai_model_repo.routes.ai_models_uploads as upload_routes
    from gen3_ai_model_repo.models.schemas import DirectUploadCompleteRequest

    namespace = f"tx-{uuid4().hex}"
    repo = "completion"
    intent_ids = [uuid4(), uuid4()]
    checksum = "b" * 64
    try:
        async with pg_connection.acquire() as connection:
            model_id = await connection.fetchval(
                "INSERT INTO models(namespace,model_name) VALUES($1,$2) RETURNING id", namespace, repo
            )
            for intent_id, name in zip(intent_ids, ("first.bin", "second.bin")):
                await connection.execute(
                    """INSERT INTO model_upload_intents
                       (id,namespace,repo,revision_name,file_name,object_key,checksum_sha256,reserved_bytes,expires_at)
                       VALUES($1,$2,$3,$4,$5,$6,$7,$8,NOW()+INTERVAL '1 hour')""",
                    intent_id,
                    namespace,
                    repo,
                    "main",
                    name,
                    f"{namespace}/{intent_id}/{name}",
                    checksum,
                    1,
                )
            await connection.execute(
                """CREATE OR REPLACE FUNCTION test_fail_intent_completion() RETURNS trigger LANGUAGE plpgsql AS $$
                   BEGIN IF NEW.completed_at IS NOT NULL THEN RAISE EXCEPTION 'test intent completion failure'; END IF; RETURN NEW; END $$"""
            )
            await connection.execute(
                "CREATE TRIGGER test_fail_intent_completion BEFORE UPDATE ON model_upload_intents FOR EACH ROW EXECUTE FUNCTION test_fail_intent_completion()"
            )

        class Provider:
            async def get_file_metadata(self, key):
                return {"size": 1, "etag": "etag", "checksum_sha256": checksum}

        monkeypatch.setattr(upload_routes, "get_storage_provider", lambda: Provider())
        with pytest.raises(Exception, match="test intent completion failure"):
            await upload_routes.complete_upload(
                namespace, repo, DirectUploadCompleteRequest(upload_ids=[str(value) for value in intent_ids])
            )
        async with pg_connection.acquire() as connection:
            assert await connection.fetchval("SELECT COUNT(*) FROM model_revisions WHERE model_id=$1", model_id) == 0
            assert (
                await connection.fetchval(
                    "SELECT COUNT(*) FROM model_files WHERE revision_id NOT IN (SELECT id FROM model_revisions)"
                )
                == 0
            )
            assert (
                await connection.fetchval(
                    "SELECT COUNT(*) FROM model_upload_intents WHERE id=ANY($1::uuid[]) AND completed_at IS NOT NULL",
                    intent_ids,
                )
                == 0
            )
    finally:
        async with pg_connection.acquire() as connection:
            await connection.execute("DROP TRIGGER IF EXISTS test_fail_intent_completion ON model_upload_intents")
            await connection.execute("DROP FUNCTION IF EXISTS test_fail_intent_completion()")
            await connection.execute("DELETE FROM models WHERE namespace=$1", namespace)


@pytest.mark.asyncio
async def test_locked_recheck_rejects_expired_intent_without_writes(pg_connection):
    """An intent expiring after prevalidation is rejected by the locked query."""
    intent_id = uuid4()
    async with pg_connection.acquire() as connection:
        await connection.execute(
            """INSERT INTO model_upload_intents
           (id,namespace,repo,revision_name,file_name,object_key,checksum_sha256,reserved_bytes,expires_at)
           VALUES($1,$2,$3,$4,$5,$6,$7,$8,NOW()-INTERVAL '1 second')""",
            intent_id,
            f"tx-{uuid4().hex}",
            "expired",
            "main",
            "x",
            f"expired/{intent_id}",
            "a" * 64,
            1,
        )
        rows = await connection.fetch(
            "SELECT id FROM model_upload_intents WHERE id=$1 AND completed_at IS NULL AND expires_at > NOW() FOR UPDATE",
            intent_id,
        )
    assert rows == []


@pytest.mark.asyncio
async def test_completion_rejects_object_larger_than_reserved_bytes(pg_connection, monkeypatch):
    """Completion rechecks the stored size against the locked intent reservation."""
    import gen3_ai_model_repo.routes.ai_models_uploads as upload_routes
    from gen3_ai_model_repo.models.schemas import DirectUploadCompleteRequest

    namespace = f"tx-{uuid4().hex}"
    repo = "size-check"
    intent_id = uuid4()
    checksum = hashlib.sha256(b"too-large").hexdigest()
    try:
        async with pg_connection.acquire() as connection:
            await connection.execute("INSERT INTO models(namespace,model_name) VALUES($1,$2)", namespace, repo)
            await connection.execute(
                """INSERT INTO model_upload_intents
                   (id,namespace,repo,revision_name,file_name,object_key,checksum_sha256,reserved_bytes,expires_at)
                   VALUES($1,$2,$3,'main','file.bin',$4,$5,1,NOW()+INTERVAL '1 hour')""",
                intent_id,
                namespace,
                repo,
                f"{namespace}/{intent_id}/file.bin",
                checksum,
            )

        class Provider:
            async def get_file_metadata(self, object_key):
                return {"size": 2, "etag": "etag", "checksum_sha256": checksum}

        monkeypatch.setattr(upload_routes, "get_storage_provider", lambda: Provider())
        with pytest.raises(Exception, match="exceed their reservations"):
            await upload_routes.complete_upload(
                namespace,
                repo,
                DirectUploadCompleteRequest(upload_ids=[str(intent_id)]),
            )
        async with pg_connection.acquire() as connection:
            assert (
                await connection.fetchval(
                    "SELECT COUNT(*) FROM model_revisions mr JOIN models m ON m.id=mr.model_id WHERE m.namespace=$1",
                    namespace,
                )
                == 0
            )
            assert (
                await connection.fetchval(
                    "SELECT completed_at IS NULL FROM model_upload_intents WHERE id=$1", intent_id
                )
                is True
            )
    finally:
        async with pg_connection.acquire() as connection:
            await connection.execute("DELETE FROM models WHERE namespace=$1", namespace)


def test_metrics_follow_create_upload_file_delete_and_repository_delete():
    """Run the metrics transition test in a fresh Prometheus subprocess."""
    database_url = os.environ.get("GEN3_AI_MODEL_REPO_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("set GEN3_AI_MODEL_REPO_TEST_DATABASE_URL for PostgreSQL integration tests")
    metrics_dir = tempfile.mkdtemp(prefix="gen3-model-repo-test-metrics-")
    environment = os.environ.copy()
    environment["GEN3_AI_MODEL_REPO_TEST_DATABASE_URL"] = database_url
    environment["PROMETHEUS_MULTIPROC_DIR"] = metrics_dir
    environment["MODEL_REPO_METRICS_CHILD"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "python_functions=metrics_transition_child",
            f"{__file__}::metrics_transition_child",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"metrics child failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "1 passed" in result.stdout, (
        f"metrics child did not report a passing test:\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.asyncio
async def metrics_transition_child(pg_connection, monkeypatch):
    """Exercise the real transition flow inside the isolated child process."""
    """Repository gauges track real route mutations against PostgreSQL."""
    from httpx import ASGITransport, AsyncClient

    import gen3_ai_model_repo.database.db as database
    import gen3_ai_model_repo.database.file_tracking as file_tracking
    import gen3_ai_model_repo.database.repo_metadata as repo_metadata
    import gen3_ai_model_repo.database.revisions as revisions
    import gen3_ai_model_repo.main as main
    import gen3_ai_model_repo.routes.ai_models_files as file_routes
    import gen3_ai_model_repo.routes.ai_models_repositories as repo_routes
    import gen3_ai_model_repo.routes.ai_models_uploads as upload_routes
    from gen3_ai_model_repo.auth import verify_authorization

    namespace = f"metrics-{uuid4().hex}"
    repo = "transitions"
    payload = b"metrics-payload"
    checksum_b64 = base64.b64encode(hashlib.sha256(payload).digest()).decode()

    class Provider:
        def __init__(self):
            self.objects = {}

        async def generate_upload_url(self, object_key, **kwargs):
            return f"stub://{object_key}"

        async def get_file_metadata(self, object_key):
            value = self.objects[object_key]
            return {"size": len(value), "etag": "etag", "checksum_sha256": hashlib.sha256(value).hexdigest()}

        async def delete_file(self, object_key):
            self.objects.pop(object_key, None)

        async def delete_prefix(self, prefix):
            for key in list(self.objects):
                if key.startswith(prefix):
                    self.objects.pop(key, None)

    provider = Provider()

    async def get_test_pool():
        return pg_connection

    monkeypatch.setattr(database, "db_pool", pg_connection)
    for module in (file_tracking, repo_metadata, revisions, upload_routes, file_routes, repo_routes):
        if hasattr(module, "get_db_pool"):
            monkeypatch.setattr(module, "get_db_pool", get_test_pool)
        if hasattr(module, "get_storage_provider"):
            monkeypatch.setattr(module, "get_storage_provider", lambda provider=provider: provider)
    monkeypatch.setattr(main, "get_db_pool", get_test_pool)
    monkeypatch.setattr(main, "get_repository_metrics", repo_metadata.get_repository_metrics)
    app = main.get_app()
    app.dependency_overrides[verify_authorization] = lambda: None

    def gauge_values(body):
        values = {}
        for name in ("models_count", "files_count", "total_size_bytes"):
            prefix = f"gen3_ai_model_repo_{name}{{"
            line = next(line for line in body.splitlines() if line.startswith(prefix))
            values[name] = float(line.rsplit(" ", 1)[1])
        return values

    async def assert_gauges(client, expected):
        response = await client.get("/metrics/")
        assert response.status_code == 200
        assert response.text
        assert gauge_values(response.text) == expected

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            await main.update_metrics(app)
            baseline = gauge_values((await client.get("/metrics/")).text)
            created = await client.post(f"/api/models/{namespace}/{repo}", json={"description": "metrics"})
            assert created.status_code == 200
            await main.update_metrics(app)
            await assert_gauges(
                client,
                {
                    "models_count": baseline["models_count"] + 1,
                    "files_count": baseline["files_count"],
                    "total_size_bytes": baseline["total_size_bytes"],
                },
            )

            intent_response = await client.post(
                f"/api/models/{namespace}/{repo}/upload-url",
                json={
                    "revision_name": "main",
                    "file_name": "config.json",
                    "checksum_sha256": checksum_b64,
                    "file_size": len(payload),
                },
            )
            assert intent_response.status_code == 200, intent_response.text
            intent = intent_response.json()
            provider.objects[intent["object_key"]] = payload
            completed = await client.post(
                f"/api/models/{namespace}/{repo}/complete-upload",
                json={"revision_name": "main", "upload_ids": [intent["upload_id"]]},
            )
            assert completed.status_code == 200, completed.text
            await assert_gauges(
                client,
                {
                    "models_count": baseline["models_count"] + 1,
                    "files_count": baseline["files_count"] + 1,
                    "total_size_bytes": baseline["total_size_bytes"] + len(payload),
                },
            )

            deleted_file = await client.delete(
                f"/api/models/{namespace}/{repo}/files/{namespace}:{repo}:main:config.json"
            )
            assert deleted_file.status_code == 200, deleted_file.text
            await assert_gauges(
                client,
                {
                    "models_count": baseline["models_count"] + 1,
                    "files_count": baseline["files_count"],
                    "total_size_bytes": baseline["total_size_bytes"],
                },
            )

            deleted_repo = await client.delete(f"/api/models/{namespace}/{repo}")
            assert deleted_repo.status_code == 200, deleted_repo.text
            await assert_gauges(client, baseline)
    finally:
        async with pg_connection.acquire() as connection:
            await connection.execute("DELETE FROM models WHERE namespace=$1", namespace)


@pytest.mark.asyncio
async def test_two_reservations_compete_for_last_capacity(pg_connection, monkeypatch):
    """Advisory locking allows exactly one reservation at the quota boundary."""
    namespace = f"tx-{uuid4().hex}"
    repo = "quota"
    from gen3_ai_model_repo.database import upload_intents

    monkeypatch.setattr(
        upload_intents.config,
        "DIRECT_UPLOAD_QUOTA_BYTES",
        2,
    )
    monkeypatch.setattr(
        upload_intents.config,
        "DIRECT_UPLOAD_MAX_FILE_BYTES",
        max(2, upload_intents.config.DIRECT_UPLOAD_MAX_FILE_BYTES),
    )
    monkeypatch.setattr(upload_intents.config, "DIRECT_UPLOAD_MAX_INTENTS", 10)
    reserve_upload = upload_intents.reserve_upload

    async def reserve(index):
        try:
            await reserve_upload(namespace, repo, "main", str(index), f"{namespace}/{index}", "b" * 64, 2)
            return True
        except ValueError as error:
            assert str(error) == "direct upload quota exhausted"
            return False

    results = await asyncio.gather(reserve(1), reserve(2))
    assert sum(results) == 1
    async with pg_connection.acquire() as connection:
        active = await connection.fetchval(
            """SELECT COUNT(*) FROM model_upload_intents
               WHERE namespace=$1 AND repo=$2 AND completed_at IS NULL AND expires_at > NOW()""",
            namespace,
            repo,
        )
        assert active == 1
        await connection.execute("DELETE FROM model_upload_intents WHERE namespace=$1 AND repo=$2", namespace, repo)


@pytest.mark.asyncio
async def test_lfs_batch_reuses_active_checksum_reservation(pg_connection, monkeypatch):
    """The LFS batch route reuses one active reservation on the migrated schema."""
    from starlette.requests import Request

    import gen3_ai_model_repo.database.upload_intents as upload_intents
    import gen3_ai_model_repo.routes.ai_models_uploads as upload_routes

    namespace = f"tx-{uuid4().hex}"
    repo = "lfs-batch"
    oid = hashlib.sha256(b"lfs-content").hexdigest()

    async with pg_connection.acquire() as connection:
        await connection.execute("INSERT INTO models(namespace, model_name) VALUES($1, $2)", namespace, repo)

    payload = json.dumps(
        {
            "operation": "upload",
            "hash_algo": "sha256",
            "ref": {"name": "main"},
            "objects": [{"oid": oid, "size": len(b"lfs-content")}],
        }
    ).encode()

    def make_request() -> Request:
        consumed = False

        async def receive():
            nonlocal consumed
            if consumed:
                return {"type": "http.disconnect"}
            consumed = True
            return {"type": "http.request", "body": payload, "more_body": False}

        return Request(
            {
                "type": "http",
                "method": "POST",
                "path": f"/{namespace}/{repo}.git/info/lfs/objects/batch",
                "raw_path": f"/{namespace}/{repo}.git/info/lfs/objects/batch".encode(),
                "query_string": b"",
                "headers": [],
                "scheme": "http",
                "server": ("testserver", 80),
                "client": ("testclient", 123),
                "root_path": "",
            },
            receive,
        )

    class Provider:
        async def generate_upload_url(self, object_key, **kwargs):
            raise NotImplementedError

    monkeypatch.setattr(upload_routes, "get_storage_provider", lambda: Provider())
    monkeypatch.setattr(upload_intents, "get_db_pool", lambda: pg_connection)
    monkeypatch.setattr(upload_routes, "get_db_pool", lambda: pg_connection)
    try:
        first = await upload_routes.lfs_batch(namespace, repo, make_request())
        second = await upload_routes.lfs_batch(namespace, repo, make_request())
        expected_checksum = base64.b64encode(bytes.fromhex(oid)).decode()
        assert first["objects"][0]["actions"]["upload"]["header"] == {
            "x-amz-sdk-checksum-algorithm": "SHA256",
            "x-amz-checksum-sha256": expected_checksum,
        }
        assert first["objects"][0]["actions"]["upload"]["href"] == second["objects"][0]["actions"]["upload"]["href"]
        async with pg_connection.acquire() as connection:
            count = await connection.fetchval(
                """SELECT COUNT(*) FROM model_upload_intents
                   WHERE namespace=$1 AND repo=$2 AND revision_name='main'
                     AND checksum_sha256=$3 AND completed_at IS NULL""",
                namespace,
                repo,
                oid,
            )
        assert count == 1
    finally:
        async with pg_connection.acquire() as connection:
            await connection.execute("DELETE FROM models WHERE namespace=$1 AND model_name=$2", namespace, repo)


@pytest.mark.asyncio
async def test_lfs_proxy_rejects_expired_capability(pg_connection, monkeypatch):
    """An expired LFS capability cannot reach storage through the proxy."""
    from fastapi import HTTPException
    from starlette.requests import Request

    import gen3_ai_model_repo.routes.ai_models_uploads as upload_routes

    namespace = f"tx-{uuid4().hex}"
    repo = "expired-lfs"
    intent_id = uuid4()
    async with pg_connection.acquire() as connection:
        await connection.execute(
            """INSERT INTO model_upload_intents
               (id,namespace,repo,revision_name,file_name,object_key,checksum_sha256,reserved_bytes,expires_at)
               VALUES($1,$2,$3,'main','model.bin',$4,$5,4,NOW()-INTERVAL '1 minute')""",
            intent_id,
            namespace,
            repo,
            f"{namespace}/{intent_id}/model.bin",
            hashlib.sha256(b"dead").hexdigest(),
        )

    async def receive():
        return {"type": "http.request", "body": b"dead", "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "PUT",
            "path": f"/api/models/{namespace}/{repo}/lfs/upload/{intent_id}",
            "raw_path": b"",
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 123),
            "root_path": "",
        },
        receive,
    )
    monkeypatch.setattr(upload_routes, "get_db_pool", lambda: pg_connection)
    try:
        with pytest.raises(HTTPException, match="not found") as error:
            await upload_routes.lfs_upload_object(namespace, repo, intent_id, request)
        assert error.value.status_code == 404
    finally:
        async with pg_connection.acquire() as connection:
            await connection.execute("DELETE FROM model_upload_intents WHERE id=$1", intent_id)


@pytest.mark.asyncio
async def test_existing_repository_later_revision_direct_upload_flow(pg_connection, monkeypatch, tmp_path):
    """Exercise repository creation, direct revision completion, and 409 safety."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    import gen3_ai_model_repo.database.db as database
    import gen3_ai_model_repo.routes.ai_models_files as file_routes
    import gen3_ai_model_repo.routes.ai_models_repositories as repo_routes
    import gen3_ai_model_repo.routes.ai_models_uploads as upload_routes
    from gen3_ai_model_repo.auth import verify_authorization
    from gen3_ai_model_repo.routes.ai_models_files import ai_models_files_router
    from gen3_ai_model_repo.routes.ai_models_repositories import ai_models_repositories_router
    from gen3_ai_model_repo.routes.ai_models_uploads import ai_models_uploads_router

    namespace = f"tx-{uuid4().hex}"
    repo = "later-revision"
    payload = b"revision-two"
    checksum_b64 = base64.b64encode(hashlib.sha256(payload).digest()).decode()

    class Provider:
        def __init__(self):
            self.objects = {}

        async def generate_upload_url(self, object_key, **kwargs):
            return f"stub://{object_key}"

        async def get_file_metadata(self, object_key):
            value = self.objects[object_key]
            return {"size": len(value), "etag": "etag", "checksum_sha256": hashlib.sha256(value).hexdigest()}

        async def get_file_path(self, object_key):
            path = tmp_path / object_key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.objects[object_key])
            return path

        async def delete_file(self, object_key):
            self.objects.pop(object_key, None)

        async def delete_prefix(self, prefix):
            for key in list(self.objects):
                if key.startswith(prefix):
                    self.objects.pop(key)

    provider = Provider()
    monkeypatch.setattr(database, "db_pool", pg_connection)
    for module in (upload_routes, file_routes, repo_routes):
        monkeypatch.setattr(module, "get_storage_provider", lambda provider=provider: provider)

    app = FastAPI()
    app.include_router(ai_models_repositories_router)
    app.include_router(ai_models_uploads_router)
    app.include_router(ai_models_files_router)
    app.dependency_overrides[verify_authorization] = lambda: None

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            assert (
                await client.post(f"/api/models/{namespace}/{repo}", json={"description": "test"})
            ).status_code == 200
            upload_url = await client.post(
                f"/api/models/{namespace}/{repo}/upload-url",
                json={
                    "revision_name": "v2",
                    "file_name": "config.json",
                    "checksum_sha256": checksum_b64,
                    "file_size": len(payload),
                },
            )
            if upload_url.status_code == 422:
                raise AssertionError(upload_url.text)
            assert upload_url.status_code == 200
            intent = upload_url.json()
            provider.objects[intent["object_key"]] = payload
            completed = await client.post(
                f"/api/models/{namespace}/{repo}/complete-upload",
                json={"revision_name": "v2", "upload_ids": [intent["upload_id"]]},
            )
            assert completed.status_code == 200
            resolved_before = await client.get(f"/api/models/{namespace}/{repo}/resolve/v2/config.json")
            assert resolved_before.content == payload
            assert (await client.head(f"/api/models/{namespace}/{repo}/resolve/v2/config.json")).headers[
                "x-linked-size"
            ] == str(len(payload))
            replay = await client.post(
                f"/api/models/{namespace}/{repo}/complete-upload",
                json={"revision_name": "v2", "upload_ids": [intent["upload_id"]]},
            )
            assert replay.status_code == 404
            existing = dict(provider.objects)
            multipart = await client.post(
                f"/api/models/{namespace}/{repo}/upload",
                data={"revision_name": "main"},
                files={"files": ("config.json", b"must-not-replace", "application/json")},
            )
            assert multipart.status_code == 409
            assert provider.objects == existing
            resolved_after = await client.get(f"/api/models/{namespace}/{repo}/resolve/v2/config.json")
            assert resolved_after.status_code == 200
            assert resolved_after.content == resolved_before.content == payload
    finally:
        async with pg_connection.acquire() as connection:
            await connection.execute("DELETE FROM models WHERE namespace=$1", namespace)
