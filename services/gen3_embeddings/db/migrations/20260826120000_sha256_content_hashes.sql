-- ===========================================================================
-- IMPORTANT! READ! DO NOT SKIP THIS!!
-- ===========================================================================
-- We need to follow an expand/contract pattern for db migrations for safety.
-- e.g. if you are replacing a column or doing anything that would BREAK running
--      services, stop. Do not do that. Instead, _expand_ the current schema to ONLY ADD
--      new things which are not breaking. Then, once everyone has completed that migration,
--      _contract_ the schema, actually doing the removal of anything you wanted to.
--
-- IMPORTANT QUESTIONS:
-- 1. Is this change backwards compatible with running application pods?
--      - Yes? Great, you're following the "expand" correctly. Now consider when and how you should "contract".
--      - No? Stop. You could break production. You need the OLD application code to work with the NEW database
--        and the NEW application code to work with the NEW database. Eventually, when you apply the "contract" migration, the
--        OLD logic can be safely removed (assuming everyone has migrated to the expansion).
-- 2. DO NOT drop or rename columns in an initial migration. Use a multi-stage rollout
--      - Add new column in one migration, remove old in a FUTURE migration after everyone is on the first one
-- 3. If you are only adding a heavy index, consider adding "transaction:false" to the migrate line
--    and use CREATE INDEX CONCURRENTLY.
--    Changing "-- migrate:up" to "-- migrate:up transaction:false" tells dbmate to turn OFF transactional
--    migrations.
--
--    If you don't do this and your migration takes a long time, it will block live operations (read/write).
--    You can use CONCURRENTLY to not block and allow read/write while
--    the action happens in the background in the db, but only if you tell dbmate to not put the whole migration
--    in a transaction.
--
--    NOTE: Heavy indexes may take longer than the statement_timeout default below. If building a large index,
--          manually add "SET statement_timeout = '0';" (disabled) or some other reasonable maximum. Just note
--          that postgres will kill a statement after that maximum.
--
-- ===========================================================================
-- WHAT THIS MIGRATION IS FOR
-- ===========================================================================
-- `embedding_hash` and `metadata_hash` are the service's definition of "the same embedding":
-- the unique constraint on (collection_id, embedding_hash, metadata_hash, authz) is what
-- makes a repeated POST a 409 and what PUT conflicts on to become an update.
--
-- Those hashes were `md5(...)`, computed inside the INSERT over the JSON text the app sent.
-- Two problems:
--
--   1. md5 collisions are cheap to construct, and the hashed bytes are entirely
--      caller-controlled. Two crafted payloads that hash alike collapse into one row, so a
--      POST of new content can be refused as a duplicate and a PUT can overwrite an
--      unrelated row.
--   2. The hash was taken over the caller's full-precision JSON text, not over what
--      Postgres stores. `vector` stores float32 and `halfvec` stores float16, so inputs that
--      differ only below the storage precision produce byte-identical stored vectors with
--      DIFFERENT hashes, which slips past the constraint entirely. float16 has ~3 decimal
--      digits, so on halfvec collections this is easy to hit by accident.
--
-- The app now computes sha256 in Python (see database/hashing.py) over the storage-precision
-- bytes, truncated to the 128 bits a uuid column holds.
--
-- Rehashing the EXISTING columns in place is not possible in SQL: the old hash input was
-- Python's `json.dumps` of float64s (`[0.1, 0.2]`), which Postgres cannot reproduce from a
-- stored float4 vector (`[0.1,0.2]`). So this is an expand: new columns, populated by a
-- separate backfill, with the old columns left exactly as they are.
--
-- ===========================================================================
-- ROLLOUT ORDER -- the backfill is NOT optional
-- ===========================================================================
--   1. Apply this migration and    (safe with old pods running: the added columns are
--      the two index migrations      nullable and nothing reads them yet)
--      that follow it.
--   2. Deploy the new app code.    It writes the sha256 hash to BOTH the v2 columns and the
--                                  legacy columns. Writing it to the legacy columns keeps
--                                  their NOT NULL/unique constraint satisfied, so a rollback
--                                  of this migration stays possible.
--   3. Run the backfill:           uv run python db/backfill_sha256_content_hashes.py
--                                  Until it finishes, rows written before step 2 have a NULL
--                                  v2 hash and therefore do not participate in the new unique
--                                  index -- a re-POST of pre-existing content inserts a
--                                  duplicate row instead of returning 409. Run it promptly.
--   4. LATER, once every deployment is on the new code and backfilled: a CONTRACT migration
--      drops the legacy `embedding_hash`/`metadata_hash` columns and their constraint and
--      renames the v2 columns into their place.
--
-- During the step 1-3 window, old pods (md5) and new pods (sha256) do not recognize each
-- other's rows as duplicates, so a concurrent POST of identical content through both can
-- produce two rows. That is the deliberate cost of not blocking writes; there is no
-- correctness loss beyond dedup misses.
--
-- ===========================================================================
-- UPGRADE MIGRATION SQL
-- ===========================================================================
-- migrate:up
--
-- Columns only. The unique index that gives them meaning is built CONCURRENTLY in the two
-- migrations that follow, which cannot share a file with anything else: dbmate sends a
-- migration body to Postgres as a single query, and a multi-statement query runs in an
-- implicit transaction block, which CREATE INDEX CONCURRENTLY refuses.
--
-- ==== guardrails: prod safety ====
-- in postgres, while a migration waits for a lock, it blocks ALL live app traffic
-- behind it in the queue.
--
-- lock_timeout:      If the migration can't get a lock within n seconds (e.g. stuck
--                    behind a slow app query), it aborts safely to protect the live app traffic.
-- statement_timeout: If the migration gets the lock but takes longer than n seconds
--                    to run, it aborts so it doesn't hold the DB hostage.
SET lock_timeout = '10s';
SET statement_timeout = '60s';

-- ==== 1. the new hash columns ====
-- Nullable, no default: adding them is a metadata-only change that does not rewrite the
-- table, and NULL is what marks a row as "not yet backfilled". Old pods never mention these
-- columns, so they keep working untouched.
ALTER TABLE embeddings_vector
ADD COLUMN IF NOT EXISTS embedding_hash_v2 uuid,
ADD COLUMN IF NOT EXISTS metadata_hash_v2 uuid;

ALTER TABLE embeddings_halfvec
ADD COLUMN IF NOT EXISTS embedding_hash_v2 uuid,
ADD COLUMN IF NOT EXISTS metadata_hash_v2 uuid;

-- Kept short enough to read in \d+ output; database/hashing.py is the real explanation.
COMMENT ON COLUMN embeddings_vector.embedding_hash_v2 IS
'sha256/128 of the stored float32 bytes, NULL until backfilled';
COMMENT ON COLUMN embeddings_vector.metadata_hash_v2 IS
'sha256/128 of canonical metadata JSON, NULL until backfilled';
COMMENT ON COLUMN embeddings_halfvec.embedding_hash_v2 IS
'sha256/128 of the stored float16 bytes, NULL until backfilled';
COMMENT ON COLUMN embeddings_halfvec.metadata_hash_v2 IS
'sha256/128 of canonical metadata JSON, NULL until backfilled';

-- ===========================================================================
-- DOWNGRADE MIGRATION SQL
-- ===========================================================================
-- migrate:down
--
-- Take the app back to the previous release first; new-code pods fail on the missing columns.
--
-- Dropping the v2 columns loses the backfill, so a re-upgrade has to re-run it. The legacy
-- md5 columns and their unique constraint are untouched by the upgrade, so they are still
-- correct here -- except that rows written by new code hold a sha256 value in them, which
-- old code will not match against, i.e. those rows can be duplicated after a rollback.
--
-- Dropping the columns drops the v2 index with them, so rolling this back does not require
-- the index migrations to have been rolled back first.
SET lock_timeout = '10s';
SET statement_timeout = '60s';

ALTER TABLE embeddings_vector
DROP COLUMN IF EXISTS embedding_hash_v2,
DROP COLUMN IF EXISTS metadata_hash_v2;

ALTER TABLE embeddings_halfvec
DROP COLUMN IF EXISTS embedding_hash_v2,
DROP COLUMN IF EXISTS metadata_hash_v2;
