"""
Fail when a public function takes parameters but its docstring has no `Args:` section.

Ruff's D417 validates an `Args:` section only when one exists; this covers the missing case.
Ruff has a number of open GH issues where they may add this as a new rule, but none have been
merged in a years time, hence the custom support here. FIXME: when ruff adds support for this,
remove this custom script and simply add the new rule.

A function is checked when ruff's D1 rules treat it as public: defined at module level or in a
public class, not nested inside another function, with no leading `_` in its name. Functions
without a docstring are skipped, since D1 already reports those. `test_*` functions are skipped
too: their parameters are pytest fixtures, which tests do not document.

Usage: python check_docstring_args.py PATH...

Each PATH is a Python file or a directory searched recursively for `*.py` files.
"""

import ast
import re
import sys
from collections.abc import Iterator
from pathlib import Path

ARGS_SECTION = re.compile(r"^\s*Args:\s*$", re.MULTILINE)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def main(paths: list[str]) -> int:
    """
    Check every Python file under the given paths and print each offending function.

    Args:
        paths (list[str]): Python files, or directories to search recursively for `*.py` files.

    Returns:
        int: 1 if any function is missing an `Args:` section, otherwise 0.
    """
    problems = []
    for file_path in _python_files(paths):
        problems.extend(find_missing_args_sections(file_path))

    if problems:
        print("\n".join(problems))
        return 1
    return 0


def find_missing_args_sections(path: Path) -> list[str]:
    """
    Find public functions in a file whose docstring lacks an `Args:` section.

    Args:
        path (Path): Python file to check.

    Returns:
        list[str]: One `path:line: name ...` entry per offending function.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))

    problems = []
    for function, is_method in _public_functions(tree, in_class=False):
        if function.name.startswith("test_"):
            continue

        docstring = ast.get_docstring(function)
        if not docstring or not _has_parameters(function, is_method):
            continue

        if not ARGS_SECTION.search(docstring):
            problems.append(f"{path}:{function.lineno}: {function.name} has parameters but no Args: section")

    return problems


def _python_files(paths: list[str]) -> Iterator[Path]:
    """
    Expand file and directory arguments into the Python files they name.

    Args:
        paths (list[str]): Python files, or directories to search recursively for `*.py` files.

    Yields:
        Path: Each Python file, with a directory's files in sorted order.
    """
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        else:
            yield path


def _public_functions(node: ast.AST, in_class: bool) -> Iterator[tuple[FunctionNode, bool]]:
    """
    Find the public functions reachable from a node without entering a function body.

    Compound statements such as `if` and `try` are searched through, so a function defined under
    `if TYPE_CHECKING:` still counts as module level.

    Args:
        node (ast.AST): The module or class to search.
        in_class (bool): Whether `node`'s direct function children are methods.

    Yields:
        tuple[FunctionNode, bool]: Each public function and whether it is a method.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not child.name.startswith("_"):
                yield child, in_class
        elif isinstance(child, ast.ClassDef):
            if not child.name.startswith("_"):
                yield from _public_functions(child, in_class=True)
        else:
            yield from _public_functions(child, in_class)


def _has_parameters(function: FunctionNode, is_method: bool) -> bool:
    """
    Report whether a function takes any parameter a caller passes explicitly.

    A method's first positional parameter is bound implicitly unless it is a staticmethod. It is
    dropped by position, so a staticmethod parameter that happens to be named `cls` still counts.

    Args:
        function (FunctionNode): The function to inspect.
        is_method (bool): Whether the function is defined in a class body.

    Returns:
        bool: True when the function has a parameter that needs documenting.
    """
    arguments = function.args
    positional = arguments.posonlyargs + arguments.args
    if is_method and not _is_staticmethod(function):
        positional = positional[1:]

    return bool(positional or arguments.kwonlyargs or arguments.vararg or arguments.kwarg)


def _is_staticmethod(function: FunctionNode) -> bool:
    """Return whether the function is decorated with a bare `@staticmethod`."""
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "staticmethod" for decorator in function.decorator_list
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
