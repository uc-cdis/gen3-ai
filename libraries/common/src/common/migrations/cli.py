import pathlib
import sys

import click

BASE_DIR = pathlib.Path(__file__).parent


def generate_migration_script(rev, name):
    filename = f"{rev}_{name}.py"
    path = BASE_DIR / filename

    if path.exists():
        print("File already exists:", path)
        sys.exit(1)

    content = f'''# migrations/{filename}
revision = "{rev}"
down_revision = None  # FIXME: change to the last applied revision

async def upgrade(connection):
    """Write your upgrade logic here."""
    pass

async def downgrade(connection):
    """Write your downgrade logic here."""
    pass
'''
    path.write_text(content)
    print("Created", path)


@click.command()
@click.option("--revision", type=int, help="migration revision number to create. Should be previous+1")
@click.option("--name", prompt="short name for the migration", help="used in file name")
def main(rev, name):
    """Generate a new migration script."""
    generate_migration_script(rev, name)


if __name__ == "__main__":
    main()
