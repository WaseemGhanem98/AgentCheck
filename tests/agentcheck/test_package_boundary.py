"""AgentCheck may import only its public runtime dependency surface.

A private sibling package can be importable on a maintainer's machine and absent
from the published distribution. Static allowlisting makes that error fail in
the repository instead of after installation from PyPI.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest


PUBLIC_IMPORT_ROOTS = frozenset(
    {
        "agentcheck",
        "agents",
        "jsonschema",
        "openai",
        "pydantic",
        "pydantic_ai",
        "referencing",
        "requests",
        "typing_extensions",
    }
)
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
            if node.level == 0 and node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


@pytest.mark.parametrize("source", _python_sources(), ids=lambda path: path.name)
def test_agentcheck_imports_only_public_runtime_modules(source: pathlib.Path) -> None:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    unexpected = _imported_roots(tree) - sys.stdlib_module_names - PUBLIC_IMPORT_ROOTS
    assert not unexpected, (
        f"{source.relative_to(PACKAGE_ROOT.parent)} imports undeclared module root(s) "
        f"{sorted(unexpected)}; published AgentCheck must stand alone"
    )
