"""
Tests that each layer stays inside its own concern.

The intended split, which `gen3_embeddings.dependencies` documents in full:

- ROUTES do web work and declare which action they perform. They do not reach the policy
  engine themselves.
- `dependencies` is the only place that turns (action, resource) into a policy-engine check
  and an authz-scoped DAL.
- The DAL runs SQL and sets the row-level security context. It resolves nothing, raises no
  HTTP errors (status codes live in `error_handlers`), and does not know the resource-path
  convention.
- POSTGRES enforces row visibility.

These are import-graph and source assertions, so they catch a layering violation at the
point someone writes it rather than when its consequence shows up.
"""

import ast
import pathlib

from gen3_embeddings.auth import get_allowed_collection_names_from_authz
from gen3_embeddings.database import errors as dal_errors
from gen3_embeddings.error_handlers import DATA_ACCESS_ERROR_STATUS, get_status_code_for_error

SRC = pathlib.Path(__file__).resolve().parents[1] / "src/gen3_embeddings"
DB_MODULE_PATH = SRC / "database/db.py"
ROUTES_DIR = SRC / "routes"

# Modules the data access layer must not depend on: web framework and authorization.
FORBIDDEN_DAL_IMPORTS = (
    "fastapi",
    "starlette",
    "gen3_embeddings.auth",
    "gen3_embeddings.dependencies",
    "gen3authz",
    "common.auth",
    "httpx",
    "requests",
)

# Routes declare an action and let `dependencies` act on it. Importing the policy-engine
# client or `common.auth` directly is how authorization logic gets scattered back across
# handlers, which is what this refactor removed.
FORBIDDEN_ROUTE_IMPORTS = ("common.auth", "gen3authz", "authutils")


def _imported_modules(path: pathlib.Path) -> set[str]:
    """Collect every module name imported by a source file, without importing it."""
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _violations(imported: set[str], forbidden: tuple[str, ...]) -> list[str]:
    """Return the imports that match a forbidden module or one of its submodules."""
    return [
        f"{module} (forbidden: {item})"
        for module in sorted(imported)
        for item in forbidden
        if module == item or module.startswith(f"{item}.")
    ]


def test_dal_does_not_import_web_or_authz_modules():
    """The DAL should be usable without FastAPI and must not reach the policy engine."""
    violations = _violations(_imported_modules(DB_MODULE_PATH), FORBIDDEN_DAL_IMPORTS)
    assert not violations, "db.py must not import web/authz modules: " + ", ".join(violations)


def test_routes_do_not_call_the_policy_engine_directly():
    """
    A handler that calls the policy engine itself puts authorization back in the routes.

    Handlers that need a check the path cannot express use `ctx.require(...)`, which routes
    it through the same place as the declared one.
    """
    route_files = sorted(ROUTES_DIR.glob("*.py"))
    # so this cannot pass by globbing nothing
    assert len(route_files) >= 5

    offenders = {
        path.name: _violations(_imported_modules(path), FORBIDDEN_ROUTE_IMPORTS)
        for path in route_files
        if _violations(_imported_modules(path), FORBIDDEN_ROUTE_IMPORTS)
    }
    assert not offenders, f"routes must go through dependencies.authz, not the policy engine: {offenders}"


def test_dal_sets_both_rls_settings_together():
    """
    Every RLS-scoped transaction carries both settings.

    `collections` keys on `app.allowed_collection_names` and the embeddings tables key on
    `app.allowed_authz`. Setting only one would leave whichever table the operation happens
    to touch second running under a setting that was never set, which denies everything and
    looks like data loss rather than a bug.
    """
    source = DB_MODULE_PATH.read_text()
    tree = ast.parse(source)
    with_rls = next(
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "_with_rls"
    )
    body = ast.get_source_segment(source, with_rls) or ""

    assert "set_config('app.allowed_authz'" in body
    assert "set_config('app.allowed_collection_names'" in body

    # _with_rls is the only place that takes a connection, so no query can run without them
    assert source.count("self.pool.acquire()") == 1, (
        "a query outside _with_rls runs with no RLS context, so it sees nothing (or, worse, "
        "everything if a policy is ever removed)"
    )


def test_dal_never_raises_http_errors():
    """Every failure leaves the DAL as a DataAccessError, so status codes are chosen elsewhere."""
    source = DB_MODULE_PATH.read_text()
    assert "HTTPException" not in source
    assert "status_code" not in source


def test_dal_methods_do_not_take_authz_arguments():
    """
    The caller's authz arrives once, in the constructor, not per call.

    When it was a parameter on each collection method, a call site could pass a set resolved
    for a different action than the one the DAL was constructed for, and nothing would
    notice. One field per instance means the RLS context and the Python short-circuits
    cannot disagree.
    """
    tree = ast.parse(DB_MODULE_PATH.read_text())
    dal = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "DataAccessLayer")

    offenders = {}
    for method in dal.body:
        if not isinstance(method, ast.AsyncFunctionDef) or method.name == "__init__":
            continue
        args = {arg.arg for arg in method.args.args} | {arg.arg for arg in method.args.kwonlyargs}
        leaked = args & {"allowed_collection_names", "allowed_authz"}
        if leaked:
            offenders[method.name] = sorted(leaked)

    assert not offenders, f"authz belongs on the DAL instance, not these method signatures: {offenders}"


def test_dal_does_not_interpret_authz_paths():
    """Turning authz paths into collection names is an authz concern, not a database one."""
    source = DB_MODULE_PATH.read_text()
    assert "/vectorstore/collections" not in source, (
        "db.py appears to derive authz paths itself; that belongs in auth.get_allowed_collection_names_from_authz"
    )


def test_authz_path_interpretation_lives_in_the_auth_layer():
    """
    The helper behavior is covered in detail by test_collection_name_normalization.py; this only
    pins that the extraction happened and did not leave the DAL depending on it.
    """
    assert get_allowed_collection_names_from_authz(["/vectorstore/collections/docs"]) == {"docs"}
    assert get_allowed_collection_names_from_authz([]) == set()


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
