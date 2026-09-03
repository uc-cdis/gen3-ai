-- ===========================================================================
-- IMPORTANT! READ! DO NOT SKIP THIS!!
-- ===========================================================================
-- See 20260826120000_sha256_content_hashes.sql for what this change is and how it rolls out.
-- This file exists only to hold the index build, because it has to run without a transaction:
-- dbmate hands a migration body to Postgres as a single query, a multi-statement query runs
-- in an implicit transaction block, and CREATE INDEX CONCURRENTLY refuses to run in one. So
-- this migration is exactly ONE statement -- no SET, no second index. Adding anything else
-- here brings back "CREATE INDEX CONCURRENTLY cannot run inside a transaction block".
--
-- This is the constraint that makes (collection_id, embedding_hash_v2, metadata_hash_v2,
-- authz) unique, i.e. the thing that makes a repeated POST a 409 and gives PUT something to
-- conflict on. A unique INDEX rather than a unique CONSTRAINT, because only an index can be
-- built CONCURRENTLY; `ON CONFLICT (...)` in the app infers an index just as happily.
--
-- Left at the default NULLS DISTINCT deliberately. NULLS NOT DISTINCT is the tighter rule,
-- but every not-yet-backfilled row has NULL in both hash columns, so under NULLS NOT DISTINCT
-- any two of them sharing a collection and authz would collide and the index could not be
-- built on existing data at all.
--
-- CONCURRENTLY does not take a write lock, so this is safe against live traffic, but it can
-- take a while on a large table and there is no statement_timeout guard here (a SET would
-- make this a multi-statement migration). If it fails it leaves an INVALID index behind,
-- which has to go before retrying:
--   DROP INDEX CONCURRENTLY IF EXISTS embeddings_vector_uniq_collection_embhash_v2;
--
-- ===========================================================================
-- UPGRADE MIGRATION SQL
-- ===========================================================================
-- migrate:up transaction:false
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
embeddings_vector_uniq_collection_embhash_v2 ON embeddings_vector (
  collection_id, embedding_hash_v2, metadata_hash_v2, authz
);

-- ===========================================================================
-- DOWNGRADE MIGRATION SQL
-- ===========================================================================
-- migrate:down transaction:false
DROP INDEX CONCURRENTLY IF EXISTS embeddings_vector_uniq_collection_embhash_v2;
