-- migrate:up
ALTER TABLE model_upload_intents
ADD COLUMN IF NOT EXISTS checksum_sha256 text;

UPDATE model_upload_intents
SET checksum_sha256 = repeat('0', 64)
WHERE checksum_sha256 IS NULL;

ALTER TABLE model_upload_intents
ALTER COLUMN checksum_sha256 SET NOT NULL;

-- migrate:down
ALTER TABLE model_upload_intents
DROP COLUMN IF EXISTS checksum_sha256;
