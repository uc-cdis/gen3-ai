-- Migration: Add additional indexes and table structures for model repo

-- Ensure updated_at column has a trigger to auto-update
CREATE OR REPLACE FUNCTION update_model_repositories_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop trigger if it exists to avoid conflicts
DROP TRIGGER IF EXISTS update_model_repositories_updated_at ON model_repositories;

-- Create trigger for auto-updating updated_at
CREATE TRIGGER update_model_repositories_updated_at
BEFORE UPDATE ON model_repositories
FOR EACH ROW
EXECUTE FUNCTION update_model_repositories_updated_at();
