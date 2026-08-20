"""AgentCheck ships as its own distribution, so it may not import its host repo.

AgentCheck is developed alongside AgentLens but released separately and
publicly. A single ``from agentlens_sdk...`` line is enough to make the public
package unimportable for anyone who installed it from PyPI, and it would not
fail here, because the private sources are on the path during development. The
static scan below is the thing that fails instead.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


PRIVATE_ROOTS = ("agentlens", "agentlens_sdk", "agentlens_collector")
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "agentcheck"


def _python_sources() -> list[pathlib.Path]:
    sources = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert sources, f"no AgentCheck sources found under {PACKAGE_ROOT}"
    return sources


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import cannot escape the package, so only absolute
            # ones can name a private distribution.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


@pytest.mark.parametrize("source", _python_sources(), ids=lambda path: path.name)
def test_agentcheck_never_imports_private_agentlens_modules(
    source: pathlib.Path,
) -> None:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    forbidden = _imported_roots(tree).intersection(PRIVATE_ROOTS)
    assert not forbidden, (
        f"{source.relative_to(PACKAGE_ROOT.parent)} imports {sorted(forbidden)}; "
        "AgentCheck must not depend on private AgentLens code"
    )


def test_agentcheck_imports_without_any_private_module_available() -> None:
    """Import the package the way a PyPI installation would: with nothing else."""

    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent(
        f"""
        import importlib, sys

        class Blocked:
            def find_spec(self, name, path=None, target=None):
                if name.split(".", 1)[0] in {PRIVATE_ROOTS!r}:
                    raise ImportError(f"private module {{name}} is not installable")
                return None

        sys.meta_path.insert(0, Blocked())
        for module in ("agentcheck", "agentcheck.cli", "agentcheck.application",
                       "agentcheck.privacy", "agentcheck.redaction",
                       "agentcheck.adapters"):
            importlib.import_module(module)
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
