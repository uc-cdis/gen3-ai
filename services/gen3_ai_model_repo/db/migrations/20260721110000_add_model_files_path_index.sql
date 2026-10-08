-- migrate:up
SET lock_timeout = '10s';
SET statement_timeout = '60s';

CREATE INDEX IF NOT EXISTS idx_model_files_path -- noqa: PG01
ON model_files (file_path);

-- migrate:down
SET lock_timeout = '10s';
SET statement_timeout = '60s';

DROP INDEX IF EXISTS idx_model_files_path; -- noqa: PG01
