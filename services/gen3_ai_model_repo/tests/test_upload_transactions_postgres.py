"""
PostgreSQL integration tests for direct-upload transaction guarantees.

Run with ``GEN3_AI_MODEL_REPO_TEST_DATABASE_URL`` pointing at a disposable
database after applying the model-repository migrations.
"""

import asyncio
import base64
import hashlib
import os
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
async def test_two_reservations_compete_for_last_capacity(pg_connection, monkeypatch):
    """Advisory locking allows exactly one reservation at the quota boundary."""
    namespace = f"tx-{uuid4().hex}"
    repo = "quota"
    from gen3_ai_model_repo.database import upload_intents

    monkeypatch.setattr(
        upload_intents.config,
        "DIRECT_UPLOAD_QUOTA_BYTES",
        upload_intents.config.DIRECT_UPLOAD_MAX_FILE_BYTES,
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
                json={"revision_name": "v2", "file_name": "config.json", "checksum_sha256": checksum_b64},
            )
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
