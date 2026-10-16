-- The restricted role the service connects as (PGUSER), and the grants it runs with.
--
-- Run as the migration/admin role (DB_MIGRATION_USER), after the schema exists:
--
--     psql -v ON_ERROR_STOP=1 -v app_user="$PGUSER" -v app_password="$PGPASSWORD" -f db/app_user.sql
--
-- `just db_load` and `just db_migrate` run this for you, through `just db_app_user`.
--
-- Idempotent: an existing role is left as it is, including its password, and repeating a GRANT
-- is a no-op. Roles are cluster-wide and carry a password, so they are not part of the dbmate
-- migrations or db/schema.sql.
--
-- psql variables are not expanded inside dollar-quoted DO blocks, so each statement is built
-- with format() and run with \gexec. %I and %L quote the role name and password.

-- The migrations create the `vector` extension, which takes a superuser, and the service
-- refuses to start as a superuser. So PGUSER has to be a different role from the one running
-- this.
SELECT format(
    'DO $$ BEGIN RAISE EXCEPTION ''%%'', %L; END $$',
    format(
        'PGUSER (%s) is the role running this script. The service cannot run as the migration '
        'role: set DB_MIGRATION_USER/DB_MIGRATION_PASSWORD to an admin role and PGUSER to a '
        'separate role for the service.',
        :'app_user'
    )
)
WHERE :'app_user' = current_user \gexec

-- An existing role is not altered, so one that bypasses row-level security would still fail
-- the service's startup check after this script succeeded.
SELECT format(
    'DO $$ BEGIN RAISE EXCEPTION ''%%'', %L; END $$',
    format(
        'Role %s already exists and is SUPERUSER or BYPASSRLS, which the service refuses to run '
        'as. Set PGUSER to a different role, or remove those attributes from this one.',
        :'app_user'
    )
)
FROM pg_roles
WHERE rolname = :'app_user' AND (rolsuper OR rolbypassrls) \gexec

SELECT format(
    'CREATE ROLE %I LOGIN PASSWORD %L NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS',
    :'app_user',
    :'app_password'
)
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_user') \gexec

SELECT format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), :'app_user') \gexec

SELECT format('GRANT USAGE ON SCHEMA public TO %I', :'app_user') \gexec

-- Row-level security decides which rows these reach. No TRUNCATE, which RLS does not apply to.
SELECT format(
    'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE collections, embeddings_vector, embeddings_halfvec TO %I',
    :'app_user'
) \gexec
