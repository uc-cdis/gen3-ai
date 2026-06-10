"""
    Run all SQL migration files against a Postgres database.

    Behavior:
    - Determines a base directory for migration files:
        * If a path is passed as the first CLI argument, that path is used.
        * Otherwise, the directory containing this script is used.
    - Recursively searches the base directory for all files ending in `.sql`.
    - Connects to a Postgres database using environment variables:
        * PGUSER        (required)
        * PGPASSWORD    (required)
        * PGHOST        (optional, default: "localhost")
        * PGPORT        (optional, default: "5432")
        * PGDATABASE    (required)
    - Executes each SQL file in sorted order via asyncpg, allowing multiple
      statements per file.

    Usage examples:
    - Run migrations using the directory where this script resides:
        python migrate.py

    - Run migrations using a specific directory:
        python run_sql_files.py /app/db_migrations

    Notes:
    - Execution order is determined by sorting the full file paths, so it is
      recommended to prefix filenames with an ordered index (e.g., 001_*, 002_*).
    - This script assumes each SQL file is idempotent or safe to run as-is.
    """

import asyncio
import os
import sys
import asyncpg
from pathlib import Path

async def main():

    # ---- Determine base path for SQL files ----
    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1]).resolve()
    else:
        # Directory where this script is located
        base_path = Path(__file__).resolve().parent

    print(f"Using SQL directory (recursive): {base_path}")

    if not base_path.is_dir():
        raise ValueError(f"Provided path is not a directory: {base_path}")

    # Collect all .sql files recursively
    sql_files = sorted(base_path.rglob("*.sql"))

    if not sql_files:
        print("No .sql files found.")
        return

    # ---- DB connection details ----
    pguser = os.environ["PGUSER"]
    pgpassword = os.environ["PGPASSWORD"]
    pghost = os.environ.get("PGHOST", "localhost")
    pgport = int(os.environ.get("PGPORT", "5432"))
    pgdatabase = os.environ["PGDATABASE"]

    print(f"Connecting to Postgres at {pghost}:{pgport}, db={pgdatabase}, user={pguser}")

    conn = await asyncpg.connect(
        user=pguser,
        password=pgpassword,
        host=pghost,
        port=pgport,
        database=pgdatabase,
    )

    try:
        print("Running migrations...")
        for migration_file in sql_files:
            print(f"Loading migration file: {migration_file}")
            sql = migration_file.read_text()

            await conn.execute(sql)
            print(f"Executed: {migration_file}")

        print("All migrations completed successfully.")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
