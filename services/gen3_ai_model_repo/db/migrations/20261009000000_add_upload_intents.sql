-- migrate:up
CREATE TABLE IF NOT EXISTS model_upload_intents (
  id uuid PRIMARY KEY,
  namespace text NOT NULL,
  repo text NOT NULL,
  revision_name text NOT NULL,
  file_name text NOT NULL,
  object_key text NOT NULL UNIQUE,
  reserved_bytes bigint NOT NULL,
  expires_at timestamptz NOT NULL,
  completed_at timestamptz
);
CREATE INDEX IF NOT EXISTS idx_model_upload_intents_expiry
ON model_upload_intents (expires_at);

-- migrate:down
DROP TABLE IF EXISTS model_upload_intents;
