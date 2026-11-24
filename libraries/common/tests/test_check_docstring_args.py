"""Tests for scripts/check_docstring_args.py, which `just lint` runs against every library and service."""

import importlib.util
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_docstring_args.py"


@pytest.fixture(scope="module")
def check_docstring_args() -> ModuleType:
    """Return the check_docstring_args script, imported as a module."""
    return _load_script()


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            '''
            def fetch(url):
                """Fetch a URL."""
            ''',
            id="positional",
        ),
        pytest.param(
            '''
            async def fetch(url):
                """Fetch a URL."""
            ''',
            id="async",
        ),
        pytest.param(
            '''
            def fetch(*, timeout):
                """Fetch something."""
            ''',
            id="keyword-only",
        ),
        pytest.param(
            '''
            def fetch(*urls):
                """Fetch URLs."""
            ''',
            id="varargs",
        ),
        pytest.param(
            '''
            def fetch(**options):
                """Fetch something."""
            ''',
            id="kwargs",
        ),
        pytest.param(
            '''
            class Client:
                """A client."""

                def fetch(self, url):
                    """Fetch a URL."""
            ''',
            id="method",
        ),
        pytest.param(
            '''
            class Client:
                """A client."""

                @staticmethod
                def fetch(cls):
                    """Fetch a URL."""
            ''',
            id="staticmethod-first-param-named-cls",
        ),
        pytest.param(
            '''
            class Client:
                """A client."""

                def fetch(self, url, /):
                    """Fetch a URL."""
            ''',
            id="method-positional-only",
        ),
        pytest.param(
            '''
            class Client:
                """A client."""

                @classmethod
                def fetch(cls, url):
                    """Fetch a URL."""
            ''',
            id="classmethod-with-parameter",
        ),
        pytest.param(
            '''
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                def fetch(url):
                    """Fetch a URL."""
            ''',
            id="under-if-block",
        ),
        pytest.param(
            '''
            class Outer:
                """Outer class."""

                class Inner:
                    """Inner class."""

                    def fetch(self, url):
                        """Fetch a URL."""
            ''',
            id="nested-public-class",
        ),
    ],
)
def test_public_function_with_parameters_and_no_args_section_is_flagged(
    check_docstring_args: ModuleType, tmp_path: Path, source: str
) -> None:
    """A public function that takes parameters but has no `Args:` section is reported."""
    assert _flagged_names(check_docstring_args, _write_source(tmp_path, source)) == ["fetch"]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            '''
            def fetch(url):
                """
                Fetch a URL.

                Args:
                    url (str): The URL.
                """
            ''',
            id="has-args-section",
        ),
        pytest.param(
            '''
            def fetch():
                """Fetch the default URL."""
            ''',
            id="no-parameters",
        ),
        pytest.param(
            """
            def fetch(url):
                pass
            """,
            id="no-docstring",
        ),
        pytest.param(
            '''
            def _fetch(url):
                """Fetch a URL."""
            ''',
            id="private-function",
        ),
        pytest.param(
            '''
            class _Client:
                """A private client."""

                def fetch(self, url):
                    """Fetch a URL."""
            ''',
            id="method-of-private-class",
        ),
        pytest.param(
            '''
            def outer():
                """Wrap a nested function."""

                def fetch(url):
                    """Fetch a URL."""
            ''',
            id="nested-in-function",
        ),
        pytest.param(
            '''
            class Client:
                """A client."""

                def fetch(self):
                    """Fetch the default URL."""

                @classmethod
                def create(cls):
                    """Create a client."""

                @property
                def url(self):
                    """The client's URL."""
            ''',
            id="only-implicit-first-parameter",
        ),
        pytest.param(
            '''
            def test_fetch(client):
                """Fetching works."""
            ''',
            id="test-function",
        ),
    ],
)
def test_function_not_needing_args_section_is_not_flagged(
    check_docstring_args: ModuleType, tmp_path: Path, source: str
) -> None:
    """Functions that are private, nested, undocumented, tests, or take no explicit parameters pass."""
    assert _flagged_names(check_docstring_args, _write_source(tmp_path, source)) == []


def test_main_exits_nonzero_and_prints_offenders_found_in_directory(
    check_docstring_args: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given a directory, main searches it recursively, prints each offender and returns 1."""
    nested = tmp_path / "pkg"
    nested.mkdir()
    offender = _write_source(
        nested,
        '''
        def fetch(url):
            """Fetch a URL."""
        ''',
    )

    exit_code = check_docstring_args.main([str(tmp_path)])

    assert exit_code == 1
    assert f"{offender}:2: fetch" in capsys.readouterr().out


def test_main_exits_zero_when_nothing_is_flagged(
    check_docstring_args: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """main returns 0 and prints nothing when every file passes."""
    clean = _write_source(
        tmp_path,
        '''
        def fetch():
            """Fetch the default URL."""
        ''',
    )

    exit_code = check_docstring_args.main([str(clean)])

    assert exit_code == 0
    assert capsys.readouterr().out == ""


def _load_script() -> ModuleType:
    """
    Import the script from its file path, since `scripts/` is not an installed package.

    Returns:
        ModuleType: The loaded check_docstring_args module.
    """
    spec = importlib.util.spec_from_file_location("check_docstring_args", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_source(directory: Path, source: str, name: str = "module.py") -> Path:
    """
    Write dedented Python source to a file.

    Args:
        directory (Path): Directory to write into.
        source (str): Python source, dedented before writing.
        name (str): File name to write.

    Returns:
        Path: The written file.
    """
    path = directory / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def _flagged_names(script: ModuleType, path: Path) -> list[str]:
    """
    Run the check on a file and return the names of the functions it reports.

    Args:
        script (ModuleType): The loaded check_docstring_args module.
        path (Path): Python file to check.

    Returns:
        list[str]: Function names in the order they are reported.
    """
    return [problem.split(": ")[1].split()[0] for problem in script.find_missing_args_sections(path)]
