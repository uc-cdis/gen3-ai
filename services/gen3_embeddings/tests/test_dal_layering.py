"""
Tests that the data access layer stays free of HTTP and authorization concerns.

- HTTP status codes live in `error_handlers`, not in the DAL.
- Authorization decisions live in `auth`/`dependencies`; the DAL only filters by a set it is
  handed.
- The DAL talks to the database and nothing else.
"""

import ast
import pathlib

import pytest

from gen3_embeddings.auth import get_allowed_collection_names_from_authz
from gen3_embeddings.database import errors as dal_errors
from gen3_embeddings.error_handlers import DATA_ACCESS_ERROR_STATUS, get_status_code_for_error

DB_MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / "src/gen3_embeddings/database/db.py"

# Modules the data access layer must not depend on: web framework and authorization.
FORBIDDEN_DAL_IMPORTS = ("fastapi", "starlette", "gen3_embeddings.auth", "gen3authz", "httpx", "requests")


def _dal_imported_modules() -> set[str]:
    """Collect every module name imported by db.py, without importing it."""
    tree = ast.parse(DB_MODULE_PATH.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_dal_does_not_import_web_or_authz_modules():
    """The DAL should be usable without FastAPI and must not reach the policy engine."""
    imported = _dal_imported_modules()

    violations = [
        f"{module} (forbidden: {forbidden})"
        for module in imported
        for forbidden in FORBIDDEN_DAL_IMPORTS
        if module == forbidden or module.startswith(f"{forbidden}.")
    ]
    assert not violations, "db.py must not import web/authz modules: " + ", ".join(violations)


def test_dal_never_raises_http_errors():
    """Every failure leaves the DAL as a DataAccessError, so status codes are chosen elsewhere."""
    source = DB_MODULE_PATH.read_text()
    assert "HTTPException" not in source
    assert "status_code" not in source


def test_dal_does_not_interpret_authz_paths():
    """Turning authz paths into collection names is an authz concern, not a database one."""
    source = DB_MODULE_PATH.read_text()
    assert "/vectorstore/collections" not in source, (
        "db.py appears to derive authz paths itself; that belongs in auth.get_allowed_collection_names_from_authz"
    )


@pytest.mark.parametrize(
    "authz_paths, expected",
    [
        ([], set()),
        (["/vectorstore/collections/docs"], {"docs"}),
        (["/vectorstore/collections/docs", "/vectorstore/collections/images"], {"docs", "images"}),
        # the bare base resource does not currently grant every collection
        (["/vectorstore/collections"], set()),
        # deeper paths are not collection grants
        (["/vectorstore/collections/docs/embeddings"], set()),
        # unrelated resources are ignored
        (["/programs/foo"], set()),
    ],
)
def test_authz_path_to_collection_names(authz_paths, expected):
    """The extracted authz helper keeps the original path conventions."""
    assert get_allowed_collection_names_from_authz(authz_paths) == expected


def test_every_dal_error_has_an_explicit_status_mapping():
    """A new DataAccessError must be mapped deliberately, not silently become a 500."""
    defined = {
        obj
        for obj in vars(dal_errors).values()
        if isinstance(obj, type)
        and issubclass(obj, dal_errors.DataAccessError)
        and obj is not dal_errors.DataAccessError
    }
    unmapped = sorted(cls.__name__ for cls in defined - set(DATA_ACCESS_ERROR_STATUS))
    assert not unmapped, f"add these to DATA_ACCESS_ERROR_STATUS in error_handlers.py: {unmapped}"


def test_unmapped_errors_fall_back_to_500():
    """An unmapped error must never be reported as a client error."""

    class BrandNewError(dal_errors.DataAccessError):
        pass

    assert get_status_code_for_error(BrandNewError("boom")) == 500


def test_status_lookup_follows_subclasses():
    """A subclass inherits its parent's status rather than falling back to 500."""

    class MoreSpecificError(dal_errors.CollectionAlreadyExistsError):
        pass

    assert get_status_code_for_error(MoreSpecificError("boom")) == 409
