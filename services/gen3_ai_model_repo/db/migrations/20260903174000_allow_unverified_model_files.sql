-- migrate:up
SET lock_timeout = '10s';
SET statement_timeout = '60s';

ALTER TABLE model_files
ALTER COLUMN content_sha DROP NOT NULL;

-- migrate:down
UPDATE model_files
SET content_sha = ''
WHERE content_sha IS NULL;

ALTER TABLE model_files
ALTER COLUMN content_sha SET NOT NULL;
