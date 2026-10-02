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
-- UPGRADE MIGRATION SQL
-- ===========================================================================
-- migrate:up
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

-- ==== 1. make the authz policies fail CLOSED ====
-- The original policies read the setting as:
--     authz = ANY(current_setting('app.allowed_authz', true)::text[])
--
-- `current_setting(..., true)` returns NULL only while the setting has NEVER been set in
-- the session. Once the app has run one `set_config('app.allowed_authz', ..., true)`
-- (transaction-local, which is what the DAL does), the value reverts to the EMPTY STRING
-- at end of transaction, not to NULL. `''::text[]` is not an empty array, it is an error:
--     ERROR: malformed array literal: ""
--
-- So any query against these tables on a pooled connection that had previously served an
-- RLS-scoped transaction, but is not itself wrapped in one, fails LOUDLY with a 500-shaped
-- error instead of failing CLOSED with zero visible rows. Normalizing '' to an empty array
-- makes the "no authz supplied" case deny access, which is the safe default.
DROP POLICY IF EXISTS authz_policy_vector ON embeddings_vector;
CREATE POLICY authz_policy_vector ON embeddings_vector
USING (
  authz
  = ANY(
    COALESCE(
      NULLIF(CURRENT_SETTING('app.allowed_authz', true), '')::text[],
      '{}'::text[]
    )
  )
)
WITH CHECK (
  authz
  = ANY(
    COALESCE(
      NULLIF(CURRENT_SETTING('app.allowed_authz', true), '')::text[],
      '{}'::text[]
    )
  )
);

DROP POLICY IF EXISTS authz_policy_halfvec ON embeddings_halfvec;
CREATE POLICY authz_policy_halfvec ON embeddings_halfvec
USING (
  authz
  = ANY(
    COALESCE(
      NULLIF(CURRENT_SETTING('app.allowed_authz', true), '')::text[],
      '{}'::text[]
    )
  )
)
WITH CHECK (
  authz
  = ANY(
    COALESCE(
      NULLIF(CURRENT_SETTING('app.allowed_authz', true), '')::text[],
      '{}'::text[]
    )
  )
);

-- ==== 2. FORCE row level security ====
-- `ENABLE ROW LEVEL SECURITY` alone does NOT apply to the table OWNER; owners bypass their
-- own policies silently. Today the app user and the migration user are separate (migrations
-- run as the admin/superuser, the service connects as DB_APP_USER), so RLS does apply. But
-- if a deployment ever has the service connect as the role that owns these tables, every
-- policy above would be silently skipped and all rows would be visible.
--
-- The service's startup check (main.py) already refuses to run as SUPERUSER or with
-- BYPASSRLS, but it does not check table ownership, so this closes that remaining gap.
--
-- NOTE: superusers still bypass RLS even when it is FORCED, so superuser-run migrations and
-- maintenance are unaffected. A future DATA migration run as a non-superuser owner would be
-- subject to these policies and must set app.allowed_authz (or temporarily disable RLS).
ALTER TABLE embeddings_vector FORCE ROW LEVEL SECURITY;
ALTER TABLE embeddings_halfvec FORCE ROW LEVEL SECURITY;

-- ===========================================================================
-- DOWNGRADE MIGRATION SQL
-- ===========================================================================
-- migrate:down
SET lock_timeout = '10s';
SET statement_timeout = '60s';

ALTER TABLE embeddings_vector NO FORCE ROW LEVEL SECURITY;
ALTER TABLE embeddings_halfvec NO FORCE ROW LEVEL SECURITY;

-- restore the original (fail-loud) policy expressions
DROP POLICY IF EXISTS authz_policy_vector ON embeddings_vector;
CREATE POLICY authz_policy_vector ON embeddings_vector
USING (
  authz = ANY(CURRENT_SETTING('app.allowed_authz', true)::text[])
)
WITH CHECK (
  authz = ANY(CURRENT_SETTING('app.allowed_authz', true)::text[])
);

DROP POLICY IF EXISTS authz_policy_halfvec ON embeddings_halfvec;
CREATE POLICY authz_policy_halfvec ON embeddings_halfvec
USING (
  authz = ANY(CURRENT_SETTING('app.allowed_authz', true)::text[])
)
WITH CHECK (
  authz = ANY(CURRENT_SETTING('app.allowed_authz', true)::text[])
);
