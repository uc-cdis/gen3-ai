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
-- WHAT THIS MIGRATION DOES
-- ===========================================================================
-- Brings `collections` under row-level security, so authorization for collections is
-- enforced by Postgres like it already is for `embeddings_vector`/`embeddings_halfvec`,
-- rather than by a Python set-membership filter in the data access layer.
--
-- `collections` has no `authz` column, and deliberately does not get one. A collection's
-- authz identity IS its name: the service derives the resource path
-- `/vectorstore/collections/{collection_name}` from it by convention. So the policy keys on
-- `collection_name` and reads the set of names the caller may act on from
-- `app.allowed_collection_names`, which the data access layer sets per transaction next to
-- the `app.allowed_authz` it already sets.
--
-- NOT BACKWARDS COMPATIBLE, deliberately. Pods running code from before this migration
-- never set `app.allowed_collection_names`, so they will see zero collections. This is
-- accepted: the rollout takes downtime rather than carrying a transitional policy that
-- would have to hardcode the resource-path convention into the schema.
--
-- The '' normalization is there for the same reason as in the embeddings policies (see
-- 20260817150236_rls_fail_closed_and_force.sql): a transaction-local setting reverts to the
-- EMPTY STRING, not NULL, and `''::text[]` is a `malformed array literal` ERROR rather than
-- an empty array. Normalizing makes "no authz supplied" deny rather than blow up.
--
-- ===========================================================================
-- UPGRADE MIGRATION SQL
-- ===========================================================================
-- migrate:up
--
-- ==== guardrails: prod safety ====
-- lock_timeout:      If the migration can't get a lock within n seconds (e.g. stuck
--                    behind a slow app query), it aborts safely to protect the live app traffic.
-- statement_timeout: If the migration gets the lock but takes longer than n seconds
--                    to run, it aborts so it doesn't hold the DB hostage.
SET lock_timeout = '10s';
SET statement_timeout = '60s';

-- ==== 1. the policy ====
DROP POLICY IF EXISTS authz_policy_collections ON collections;
CREATE POLICY authz_policy_collections ON collections
USING (
  collection_name
  = ANY(
    COALESCE(
      NULLIF(CURRENT_SETTING('app.allowed_collection_names', true), '')::text[],
      '{}'::text[]
    )
  )
)
WITH CHECK (
  collection_name
  = ANY(
    COALESCE(
      NULLIF(CURRENT_SETTING('app.allowed_collection_names', true), '')::text[],
      '{}'::text[]
    )
  )
);

-- ==== 2. enable and FORCE row level security ====
-- ENABLE alone does not apply to the table OWNER; owners bypass their own policies
-- silently. FORCE closes that gap, matching what the embeddings tables already do.
-- Superusers still bypass RLS even when FORCEd, so superuser-run migrations and
-- administrative maintenance (including renaming a collection, which the API does not
-- expose) are unaffected.
ALTER TABLE collections ENABLE ROW LEVEL SECURITY;
ALTER TABLE collections FORCE ROW LEVEL SECURITY;

-- ===========================================================================
-- DOWNGRADE MIGRATION SQL
-- ===========================================================================
-- migrate:down
SET lock_timeout = '10s';
SET statement_timeout = '60s';

ALTER TABLE collections NO FORCE ROW LEVEL SECURITY;
ALTER TABLE collections DISABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS authz_policy_collections ON collections;
