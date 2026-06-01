-- Migration: Add revision and file tracking tables

-- Create table to track model revisions
CREATE TABLE IF NOT EXISTS model_revisions (
    id SERIAL PRIMARY KEY,
    repository_id INTEGER NOT NULL REFERENCES model_repositories(id) ON DELETE CASCADE,
    revision_name VARCHAR(255) NOT NULL DEFAULT 'main',
    commit_sha VARCHAR(64) NOT NULL,
    etag VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(repository_id, revision_name)
);

-- Create table to track file content hashes for optimized storage
CREATE TABLE IF NOT EXISTS model_files (
    id SERIAL PRIMARY KEY,
    revision_id INTEGER NOT NULL REFERENCES model_revisions(id) ON DELETE CASCADE,
    file_path VARCHAR(1024) NOT NULL,
    file_size BIGINT NOT NULL,
    content_sha VARCHAR(64) NOT NULL,
    content_etag VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(revision_id, file_path)
);

-- Create index for faster revision lookups
CREATE INDEX IF NOT EXISTS idx_model_revisions_repo
ON model_revisions(repository_id);

-- Create index for faster revision lookups by name
CREATE INDEX IF NOT EXISTS idx_model_revisions_revision
ON model_revisions(repository_id, revision_name);

-- Create index for faster file lookups
CREATE INDEX IF NOT EXISTS idx_model_files_revision
ON model_files(revision_id);

-- Create index for faster file lookups by path
CREATE INDEX IF NOT EXISTS idx_model_files_path
ON model_files(revision_id, file_path);

-- Create index for content SHA lookups
CREATE INDEX IF NOT EXISTS idx_model_files_sha
ON model_files(content_sha);
