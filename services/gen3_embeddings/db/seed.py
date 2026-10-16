"""
Load test collections and embeddings into a local database for development.

Writes through the service's own DataAccessLayer (`create_collection` and
`create_embeddings_bulk`), connected as the role the service runs as, so the rows get the same
content hashes, authz paths and normalized names the API would give them. The tables FORCE row
level security, so a successful run also shows that the role's grants and the RLS policies let
the service write what it is authorized to.

    just db_seed gen3_embeddings              # uses DB_CONNECTION_STRING, built from PGUSER etc.
    just db_seed gen3_embeddings --reset      # delete and reload the test collections

    cd services/gen3_embeddings
    uv run python db/seed.py --dsn postgresql://...

A collection that already exists is skipped, so re-running loads only what is missing. Refuses
to run as a role that bypasses row level security, since then nothing would be checked.

The collections and their authz paths:

    noaccess        /vectorstore/collections/noaccess        vector,  3 dims,    10,000 rows
    public          /vectorstore/collections/public          vector,  3 dims,     5,000 rows
    internal        /vectorstore/collections/internal        vector,  3 dims,     3,000 rows
    d3vector        /vectorstore/collections/d3vector        vector,  3 dims,     2,000 rows
    d200vector      /vectorstore/collections/d200vector      vector,  200 dims,     500 rows
    d3000halfvec    /vectorstore/collections/d3000halfvec    halfvec, 3000 dims,     50 rows
    d3halfvec       /vectorstore/collections/d3halfvec       halfvec, 3 dims,     1,000 rows
"""

import argparse
import asyncio
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

# allow running as a plain script from the service directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gen3_embeddings import config  # noqa: E402
from gen3_embeddings.auth import (  # noqa: E402
    get_allowed_collection_names_from_authz,
    get_authz_resource_path_from_collection_name,
)
from gen3_embeddings.database.db import DataAccessLayer  # noqa: E402
from gen3_embeddings.models.helpers import normalize_collection_name  # noqa: E402
from gen3_embeddings.models.schemas import VectorType  # noqa: E402

AI_MODEL_NAME = "test-model"


@dataclass(frozen=True)
class SeedCollection:
    """One test collection and how to generate its embeddings."""

    name: str
    description: str
    dimensions: int
    vector_type: VectorType
    rows: int
    embedding: Callable[[int], list[float]]

    def embeddings(self) -> list[list[float]]:
        """
        Generate this collection's vectors.

        Returns:
            list[list[float]]: One vector per row, for rows 1..rows.
        """
        return [self.embedding(i) for i in range(1, self.rows + 1)]

    def metadata(self) -> list[dict]:
        """
        Generate this collection's metadata.

        Returns:
            list[dict]: One metadata dict per row, matching `embeddings` by position.
        """
        return [{"name": f"{self.name}_{i}", "i": i} for i in range(1, self.rows + 1)]


SEED_COLLECTIONS = (
    SeedCollection("noaccess", "noaccess collection", 3, VectorType.vector, 10_000, lambda i: [i, i + 1, i + 2]),
    SeedCollection("public", "public collection", 3, VectorType.vector, 5_000, lambda i: [i, i + 10, i + 20]),
    SeedCollection(
        "internal", "internal collection", 3, VectorType.vector, 3_000, lambda i: [2 * i, 2 * i + 1, 2 * i + 2]
    ),
    SeedCollection("d3vector", "d3 vector collection", 3, VectorType.vector, 2_000, lambda i: [i, 3 * i, 5 * i]),
    SeedCollection(
        "d200vector", "d200 vector collection", 200, VectorType.vector, 500, lambda i: [i + o for o in range(200)]
    ),
    SeedCollection(
        "d3000halfvec",
        "d3000 halfvec collection",
        3000,
        VectorType.halfvec,
        50,
        lambda i: [i / (o + 1) for o in range(3000)],
    ),
    SeedCollection(
        "d3halfvec", "d3 halfvec collection", 3, VectorType.halfvec, 1_000, lambda i: [i + 0.1, i + 1.2, i + 2.3]
    ),
)


async def seed_collection(dal: DataAccessLayer, seed: SeedCollection, reset: bool) -> int:
    """
    Create one collection and its embeddings, unless it already exists.

    Embeddings go in batches of MAX_EMBEDDINGS_PER_REQUEST, the most one API request may carry,
    under the collection's own authz path, which is what the API assigns when a request names
    none.

    Args:
        dal (DataAccessLayer): Bound to grants covering every seed collection.
        seed (SeedCollection): The collection to load.
        reset (bool): Delete the collection first if it exists.

    Returns:
        int: Embeddings written; 0 if the collection was skipped.
    """
    name = normalize_collection_name(seed.name)
    if await dal.get_collection_by_name(name):
        if not reset:
            print(f"  {name}: already exists, skipping (pass --reset to reload it)")
            return 0
        await dal.delete_collection(name)

    collection = await dal.create_collection(
        collection_name=name,
        description=seed.description,
        dimensions=seed.dimensions,
        ai_model_name=AI_MODEL_NAME,
        vector_type=seed.vector_type,
    )
    authz = get_authz_resource_path_from_collection_name(name)
    embeddings = seed.embeddings()
    metadata = seed.metadata()
    batch_size = config.MAX_EMBEDDINGS_PER_REQUEST
    for start in range(0, len(embeddings), batch_size):
        await dal.create_embeddings_bulk(
            collection,
            embeddings[start : start + batch_size],
            authz,
            metadata[start : start + batch_size],
        )

    print(f"  {name}: {len(embeddings)} embeddings ({seed.vector_type.value}, {seed.dimensions} dims) under {authz}")
    return len(embeddings)


async def main(dsn: str, reset: bool) -> int:
    """
    Seed every test collection.

    Args:
        dsn (str): Postgres connection string for the role the service runs as.
        reset (bool): Delete and reload collections that already exist.

    Returns:
        int: Process exit code; non-zero if the role bypasses row level security.
    """
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, init=register_vector)
    try:
        role = await pool.fetchrow(
            "SELECT current_user AS name, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        )
        if role["rolsuper"] or role["rolbypassrls"]:
            print(
                f"Role '{role['name']}' bypasses row level security, so seeding as it would check nothing.\n"
                "Set PGUSER to the service's restricted role (see `just db_app_user`)."
            )
            return 1

        allowed_authz = [get_authz_resource_path_from_collection_name(seed.name) for seed in SEED_COLLECTIONS]
        dal = DataAccessLayer(
            pool,
            allowed_authz=allowed_authz,
            allowed_collection_names=get_allowed_collection_names_from_authz(allowed_authz),
        )
        print(f"Seeding test collections as '{role['name']}':")
        written = sum([await seed_collection(dal, seed, reset) for seed in SEED_COLLECTIONS])
    finally:
        await pool.close()

    print(f"Done, {written} embeddings written.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", default=str(config.DB_CONNECTION_STRING), help="Postgres connection string")
    parser.add_argument("--reset", action="store_true", help="delete and reload collections that already exist")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args.dsn, args.reset)))
