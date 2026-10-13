"""
Reusable pytest contract asserting a service's published API docs stay user-facing.

Services subclass `OpenApiDocsContract` in their own `tests/` directory rather than importing
test functions, because `just test` only collects from `services/*` -- test modules placed in
this library would never run.
"""

from typing import Any

import pytest
from fastapi import FastAPI

# Markers of maintainer-facing docstring content that must not reach the published docs.
# FastAPI falls back to the function docstring for `description` when a route omits the
# explicit `description=` kwarg, which is how Google-style `Args:` sections end up there.
INTERNAL_DOCSTRING_MARKERS = ("Args:", "Returns:", "Raises:")

# Endpoints the aggregated public spec drops as internal-only.
INTERNAL_PATH_MARKERS = ("_version", "_status")

Operation = tuple[str, str, dict[str, Any]]


def public_operations(spec: dict[str, Any]) -> list[Operation]:
    """
    Pull the operations that reach the published docs out of an OpenAPI spec.

    Args:
        spec (dict[str, Any]): An OpenAPI document, e.g. from `app.openapi()`.

    Returns:
        list[Operation]: (method, path, operation) triples, excluding internal endpoints.
    """
    return [
        (method.upper(), path, operation)
        for path, path_item in spec["paths"].items()
        if not any(marker in path for marker in INTERNAL_PATH_MARKERS)
        for method, operation in path_item.items()
    ]


class OpenApiDocsContract:
    """
    Checks that a service's published API docs read as user-facing rather than as internals.

    NOTE: Subclass as `TestOpenApiDocs` in a service's tests and override `get_app`. This base
    name deliberately lacks a `Test` prefix so pytest does not collect it on its own.
    """

    expects_public_operations: bool = True
    """Set False for a service with no routes yet, so the non-empty guard does not fail it."""

    authenticated_path_prefixes: tuple[str, ...] = ()
    """Path prefixes whose operations must document 401 and 403."""

    @staticmethod
    def get_app() -> FastAPI:
        """
        Return the service's app.

        Returns:
            FastAPI: The app whose generated spec is under test.

        Raises:
            NotImplementedError: If a subclass does not override this.
        """
        raise NotImplementedError("Subclasses must return their service's FastAPI app")

    @pytest.fixture(scope="class")
    @classmethod
    def operations(cls) -> list[Operation]:
        """Return every operation the service publishes to the aggregated public docs."""
        return public_operations(cls.get_app().openapi())

    def test_spec_documents_the_expected_operations(self, operations: list[Operation]) -> None:
        """A service that should publish routes does, so the other checks cannot pass against nothing."""
        if not self.expects_public_operations:
            assert operations == [], "Service published operations but claims to have none"
            return
        assert len(operations) > 0

    def test_every_operation_has_a_summary(self, operations: list[Operation]) -> None:
        """Each operation sets a summary, which Redocly uses as the section heading."""
        missing = [f"{method} {path}" for method, path, op in operations if not (op.get("summary") or "").strip()]
        assert not missing, "Operations missing `summary=`:\n  " + "\n  ".join(missing)

    def test_every_operation_has_a_description(self, operations: list[Operation]) -> None:
        """Each operation sets a description, so the rendered docs explain what it does."""
        missing = [f"{method} {path}" for method, path, op in operations if not (op.get("description") or "").strip()]
        assert not missing, "Operations missing `description=`:\n  " + "\n  ".join(missing)

    def test_no_description_leaks_maintainer_docstring_content(self, operations: list[Operation]) -> None:
        """No description is a raw docstring, which would expose internals and confuse clients."""
        leaking = [
            f"{method} {path} (contains {marker!r})"
            for method, path, op in operations
            for marker in INTERNAL_DOCSTRING_MARKERS
            if marker in (op.get("description") or "")
        ]
        assert not leaking, (
            "Operations whose description looks like a maintainer docstring. Pass an explicit, "
            "user-facing `description=` on the route decorator:\n  " + "\n  ".join(leaking)
        )

    def test_authenticated_operations_document_their_error_codes(self, operations: list[Operation]) -> None:
        """Operations behind auth document both 401 and 403."""
        # A service declaring no prefixes matches nothing here, which is the correct
        # assertion for it rather than a skip.
        missing = [
            f"{method} {path} (missing {code})"
            for method, path, op in operations
            if path.startswith(self.authenticated_path_prefixes)
            for code in ("401", "403")
            if code not in op.get("responses", {})
        ]
        assert not missing, "Operations with undocumented error responses:\n  " + "\n  ".join(missing)

    def test_no_operation_documents_a_server_error(self, operations: list[Operation]) -> None:
        """No operation advertises a 5xx, which is a bug rather than part of the contract."""
        documented = [
            f"{method} {path} (documents {code})"
            for method, path, op in operations
            for code in op.get("responses", {})
            if code.startswith("5")
        ]
        assert not documented, (
            "Callers should never be told to expect a server error, so 5xx responses must not be "
            "documented:\n  " + "\n  ".join(documented)
        )
