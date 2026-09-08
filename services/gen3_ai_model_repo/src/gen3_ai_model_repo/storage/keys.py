"""Validation and construction of model repository storage keys."""

import re

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_key_component(value: str, name: str) -> str:
    """
    Validate a namespace, repository, or revision key component.

    Returns:
        The validated component.

    Raises:
        ValueError: If the component contains disallowed characters.
    """
    if not value or not _SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Invalid {name}")
    return value


def build_object_key(namespace: str, repo: str, revision_name: str, filename: str) -> str:
    """
    Build a validated storage key for a repository file.

    Returns:
        The validated storage object key.

    Raises:
        ValueError: If any key component or the filename is invalid.
    """
    validate_key_component(namespace, "namespace")
    validate_key_component(repo, "repo")
    validate_key_component(revision_name, "revision")

    if not filename or "\\" in filename or "\x00" in filename or filename.startswith("/"):
        raise ValueError("Invalid filename")
    filename_parts = filename.split("/")
    if any(not part or part in {".", ".."} for part in filename_parts):
        raise ValueError("Invalid filename")

    return "/".join((namespace, repo, revision_name, *filename_parts))
