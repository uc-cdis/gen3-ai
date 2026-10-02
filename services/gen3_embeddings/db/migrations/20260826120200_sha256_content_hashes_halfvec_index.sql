-- ===========================================================================
-- IMPORTANT! READ! DO NOT SKIP THIS!!
-- ===========================================================================
-- The embeddings_halfvec half of 20260826120100_sha256_content_hashes_vector_index.sql; see
-- that file for why each index build gets a migration of its own (one statement, no
-- transaction) and 20260826120000_sha256_content_hashes.sql for the change as a whole.
--
-- If this build fails it leaves an INVALID index behind, which has to go before retrying:
--   DROP INDEX CONCURRENTLY IF EXISTS embeddings_halfvec_uniq_collection_embhash_v2;
--
-- ===========================================================================
-- UPGRADE MIGRATION SQL
-- ===========================================================================
-- migrate:up transaction:false
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
embeddings_halfvec_uniq_collection_embhash_v2 ON embeddings_halfvec (
  collection_id, embedding_hash_v2, metadata_hash_v2, authz
);

-- ===========================================================================
-- DOWNGRADE MIGRATION SQL
-- ===========================================================================
-- migrate:down transaction:false
DROP INDEX CONCURRENTLY IF EXISTS embeddings_halfvec_uniq_collection_embhash_v2;
