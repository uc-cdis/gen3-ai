#!/usr/bin/env bash
# b/c dbmate doesn't allow templating, we need some shenanigans here.
# this script will rewrite the newest migration with the template provided later in this file.
set -euo pipefail

# MUST be provided an argument of the migrations directory
if [ -z "${1:-}" ]; then
    echo "Error: Migrations directory argument is required."
    echo "Usage: $0 <path/to/migrations>"
    exit 1
fi

MIGRATIONS_DIR="$1"

# grab just the filename of the newest file
FILENAME=$(ls -t "$MIGRATIONS_DIR" 2>/dev/null | head -n 1)

TARGET_FILE="$MIGRATIONS_DIR/$FILENAME"

# safety checks: ensure a file was actually found
if [ -z "$TARGET_FILE" ]; then
    echo "Generation of migration file failed for some reason. Could not find ${TARGET_FILE}"
    exit 1
fi

# safety checks: ensure it's a .sql file
if [[ "$TARGET_FILE" != *.sql ]]; then
    echo "Error: '$TARGET_FILE' is not a .sql file."
    exit 1
fi

# safety check: ensure it's a completely fresh, unedited dbmate migration file
# strip all spaces/newlines to do a strict but spacing-agnostic match
FILE_CONTENT_CLEAN=$(tr -d '[:space:]' < "$TARGET_FILE")
EXPECTED_SIGNATURE="--migrate:up--migrate:down"

if [ "$FILE_CONTENT_CLEAN" != "$EXPECTED_SIGNATURE" ]; then
    echo "Error: '$TARGET_FILE' is not a fresh dbmate file, could not add templated comments to help."
    echo "It looks like it has already been modified or contains existing code. Aborting to protect your work."
    echo "Ensure you check other migrations to understand the requirements for new ones!!"
    exit 1
fi

# overwrite the default dbmate file with template
cat <<EOF > "$TARGET_FILE"
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

-- > YOUR MIGRATION SQL GOES HERE < --

-- ===========================================================================
-- DOWNGRADE MIGRATION SQL
-- ===========================================================================
-- migrate:down
SET lock_timeout = '10s';
SET statement_timeout = '60s';

-- > YOUR MIGRATION SQL GOES HERE < --

EOF

echo "Done."
