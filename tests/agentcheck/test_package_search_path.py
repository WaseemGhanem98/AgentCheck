"""A target finder must respect an already imported package's search path."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from collections.abc import Iterator

import pytest

from agentcheck.inspect.extractor import TargetLoadError, _ContainedTargetFinder


PACKAGE = "_ac_package_path_probe"


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Path, Path]]:
    root = tmp_path / "target"
    external = tmp_path / "external"
    root.mkdir()
    external.mkdir()
    monkeypatch.syspath_prepend(str(external))
    try:
        yield root, external
    finally:
        for name in list(sys.modules):
            if name == PACKAGE or name.startswith(PACKAGE + "."):
                sys.modules.pop(name, None)


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _enable(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setattr(sys, "meta_path", [_ContainedTargetFinder(root), *sys.meta_path])


@pytest.mark.parametrize("single_pass", [False, True])
def test_target_does_not_replace_a_submodule_of_an_external_package(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, single_pass: bool,
) -> None:
    root, external = paths
    _write(external / PACKAGE / "__init__.py")
    _write(external / PACKAGE / "child.py", "VALUE = 'external'\n")
    parent = importlib.import_module(PACKAGE)
    original_path = list(parent.__path__)
    if single_pass:
        parent.__path__ = iter(original_path)
    _write(root / PACKAGE / "child.py", "VALUE = 'target'\n")
    _enable(root, monkeypatch)

    child = importlib.import_module(PACKAGE + ".child")

    assert child.VALUE == "external"
    assert Path(child.__file__) == external / PACKAGE / "child.py"
    if not single_pass:
        assert list(parent.__path__) == original_path


@pytest.mark.parametrize("parent_path", ["external", "empty", "non-string"])
def test_excluded_target_child_does_not_rescue_a_missing_external_submodule(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, parent_path: str,
) -> None:
    root, external = paths
    _write(external / PACKAGE / "__init__.py")
    parent = importlib.import_module(PACKAGE)
    if parent_path == "empty":
        parent.__path__ = []
    elif parent_path == "non-string":
        parent.__path__ = [object()]
    _write(root / PACKAGE / "child.py", "VALUE = 'target'\n")
    _enable(root, monkeypatch)

    with pytest.raises(ModuleNotFoundError, match=PACKAGE + r"\.child"):
        importlib.import_module(PACKAGE + ".child")


def test_changed_target_package_search_path_is_authoritative(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _external = paths
    _write(root / PACKAGE / "__init__.py")
    _write(root / PACKAGE / "child.py", "VALUE = 'original-location'\n")
    _write(root / "alternate" / "child.py", "VALUE = 'chosen-location'\n")
    _enable(root, monkeypatch)
    parent = importlib.import_module(PACKAGE)
    parent.__path__ = [str(root / "alternate")]

    assert importlib.import_module(PACKAGE + ".child").VALUE == "chosen-location"


def test_finder_still_refuses_keyword_components_in_full_module_names(
    paths: tuple[Path, Path],
) -> None:
    root, _external = paths
    _write(root / PACKAGE / "child.py")

    assert _ContainedTargetFinder(root).find_spec("for.child", [str(root / PACKAGE)]) is None


@pytest.mark.parametrize("target_first", [False, True])
def test_mixed_parent_paths_preserve_concrete_module_order(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, target_first: bool,
) -> None:
    root, external = paths
    _write(root / PACKAGE / "__init__.py")
    _write(root / PACKAGE / "child.py", "VALUE = 'target'\n")
    _write(external / PACKAGE / "child.py", "VALUE = 'external'\n")
    _enable(root, monkeypatch)
    parent = importlib.import_module(PACKAGE)
    parent.__path__ = [str(p / PACKAGE) for p in ((root, external) if target_first else (external, root))]

    assert importlib.import_module(PACKAGE + ".child").VALUE == (
        "target" if target_first else "external"
    )


def test_target_namespace_candidate_does_not_hide_later_external_module(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, external = paths
    _write(root / PACKAGE / "__init__.py")
    (root / PACKAGE / "child").mkdir()
    _write(external / PACKAGE / "child.py", "VALUE = 'external-concrete'\n")
    _enable(root, monkeypatch)
    parent = importlib.import_module(PACKAGE)
    parent.__path__ = [str(root / PACKAGE), str(external / PACKAGE)]

    assert importlib.import_module(PACKAGE + ".child").VALUE == "external-concrete"


@pytest.mark.parametrize("single_pass", [False, True])
def test_mixed_namespace_child_keeps_both_search_locations(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, single_pass: bool,
) -> None:
    root, external = paths
    _write(root / PACKAGE / "__init__.py")
    _write(root / PACKAGE / "child" / "one.py", "VALUE = 'target'\n")
    _write(external / PACKAGE / "child" / "two.py", "VALUE = 'external'\n")
    _enable(root, monkeypatch)
    parent = importlib.import_module(PACKAGE)
    parent.__path__ = [str(root / PACKAGE), str(external / PACKAGE)]
    if single_pass:
        parent.__path__ = iter(parent.__path__)

    assert importlib.import_module(PACKAGE + ".child.one").VALUE == "target"
    assert importlib.import_module(PACKAGE + ".child.two").VALUE == "external"


@pytest.mark.parametrize("namespace", [False, True])
def test_target_packages_keep_nested_relative_imports_and_top_level_priority(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, namespace: bool,
) -> None:
    root, external = paths
    _write(external / PACKAGE / "__init__.py", "raise AssertionError('wrong top-level package')\n")
    if not namespace:
        _write(root / PACKAGE / "__init__.py")
        _write(root / PACKAGE / "nested" / "__init__.py")
    _write(root / PACKAGE / "value.py", "VALUE = 'target-relative'\n")
    _write(root / PACKAGE / "nested" / "leaf.py", "from ..value import VALUE\n")
    _enable(root, monkeypatch)

    assert importlib.import_module(PACKAGE + ".nested.leaf").VALUE == "target-relative"


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks required")
def test_in_target_package_alias_uses_its_resolved_parent_path(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _external = paths
    _write(root / "real_package" / "__init__.py")
    _write(root / "real_package" / "child.py", "VALUE = 'alias'\n")
    (root / PACKAGE).symlink_to(root / "real_package", target_is_directory=True)
    _enable(root, monkeypatch)

    assert importlib.import_module(PACKAGE + ".child").VALUE == "alias"


@pytest.mark.parametrize("bad_child", ["missing", "ambiguous", "escape"])
def test_target_submodule_failures_keep_containment_diagnostics(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, bad_child: str,
) -> None:
    if bad_child == "escape" and os.name != "posix":
        pytest.skip("POSIX symlinks required")
    root, external = paths
    _write(root / PACKAGE / "__init__.py")
    if bad_child == "ambiguous":
        _write(root / PACKAGE / "child.py")
        (root / PACKAGE / "child").mkdir()
    elif bad_child == "escape":
        _write(external / "child.py", "raise AssertionError('escaped module executed')\n")
        (root / PACKAGE / "child.py").symlink_to(external / "child.py")
    _enable(root, monkeypatch)
    importlib.import_module(PACKAGE)
    message = {"missing": "no module named", "ambiguous": "ambiguous import", "escape": "escapes the target directory"}[bad_child]

    with pytest.raises(TargetLoadError, match=message):
        importlib.import_module(PACKAGE + ".child")


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks required")
@pytest.mark.parametrize("parent_kind", ["iterator", "external-alias", "escaping-parent"])
def test_target_parent_paths_cannot_hide_an_outbound_child(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, parent_kind: str,
) -> None:
    root, external = paths
    _write(root / PACKAGE / "__init__.py")
    _write(external / "child.py", "VALUE = 'outside'\n")
    (root / PACKAGE / "child.py").symlink_to(external / "child.py")
    _enable(root, monkeypatch)
    parent = importlib.import_module(PACKAGE)
    if parent_kind == "iterator":
        parent.__path__ = iter([str(root / PACKAGE)])
    elif parent_kind == "external-alias":
        (external / "alias").symlink_to(root / PACKAGE, target_is_directory=True)
        parent.__path__ = [str(external / "alias")]
    else:
        (root / "escape").symlink_to(external, target_is_directory=True)
        parent.__path__ = [str(root / "escape")]

    with pytest.raises(TargetLoadError, match="escapes the target directory"):
        importlib.import_module(PACKAGE + ".child")
