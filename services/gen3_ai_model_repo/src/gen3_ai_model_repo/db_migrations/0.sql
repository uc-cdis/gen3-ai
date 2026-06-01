-- Migration: Create model_repositories table for storing AI model metadata

CREATE TABLE IF NOT EXISTS model_repositories (
    id SERIAL PRIMARY KEY,
    namespace VARCHAR(255) NOT NULL,
    repo_name VARCHAR(255) NOT NULL,
    description TEXT,
    tags TEXT[] DEFAULT '{}',
    current_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(namespace, repo_name)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_model_repositories_namespace_repo
ON model_repositories(namespace, repo_name);

-- Create index for listing repositories by namespace
CREATE INDEX IF NOT EXISTS idx_model_repositories_namespace
ON model_repositories(namespace);
