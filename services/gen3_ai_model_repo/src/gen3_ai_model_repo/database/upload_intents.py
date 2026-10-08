"""Durable reservations for direct-to-object-store uploads."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from gen3_ai_model_repo import config
from gen3_ai_model_repo.database.db import get_db_pool


async def reserve_upload(namespace: str, repo: str, revision: str, file_name: str, object_key: str) -> UUID:
    """
    Reserve quota and persist an upload intent atomically.

    Returns:
        UUID: The durable upload intent identifier.
    Raises:
        ValueError: If the active quota is exhausted.
    """
    intent_id = uuid4()
    expires = datetime.now(UTC) + timedelta(seconds=config.DIRECT_UPLOAD_EXPIRY_SECONDS)
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            used = await conn.fetchval(
                """SELECT COALESCE(SUM(reserved_bytes), 0) FROM model_upload_intents
                   WHERE namespace=$1 AND repo=$2 AND completed_at IS NULL AND expires_at > NOW()""",
                namespace,
                repo,
            )
            if int(used or 0) >= config.DIRECT_UPLOAD_QUOTA_BYTES:
                raise ValueError("direct upload quota exhausted")
            await conn.execute(
                """INSERT INTO model_upload_intents
                   (id, namespace, repo, revision_name, file_name, object_key, reserved_bytes, expires_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""",
                intent_id,
                namespace,
                repo,
                revision,
                file_name,
                object_key,
                config.DIRECT_UPLOAD_MAX_FILE_BYTES,
                expires,
            )
    return intent_id


async def get_upload_intents(intent_ids: list[UUID], namespace: str, repo: str, revision: str):
    """
    Fetch only active requested intents, with a hard completion bound.

    Returns:
        The active intent rows, or an empty list when the bound is exceeded.
    """
    if not intent_ids or len(intent_ids) > config.DIRECT_UPLOAD_MAX_INTENTS:
        return []
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(
            """SELECT id, file_name, object_key FROM model_upload_intents
               WHERE id = ANY($1::uuid[]) AND namespace=$2 AND repo=$3 AND revision_name=$4
                 AND completed_at IS NULL AND expires_at > NOW()
               LIMIT $5""",
            intent_ids,
            namespace,
            repo,
            revision,
            config.DIRECT_UPLOAD_MAX_INTENTS,
        )


async def complete_upload_intents(intent_ids: list[UUID]) -> None:
    """Mark validated intents complete."""
    if not intent_ids:
        return
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("UPDATE model_upload_intents SET completed_at=NOW() WHERE id=ANY($1::uuid[])", intent_ids)
