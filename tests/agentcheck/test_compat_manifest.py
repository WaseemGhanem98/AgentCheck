"""Keep the cross-version suite from quietly shrinking.

The full suite runs on one interpreter; the manifest decides what still runs on
all of them. That makes the manifest a coverage decision, and coverage decisions
that live only in a YAML argument list get eroded by whoever is trying to make
CI faster next.

These tests fail if a category loses its files, if a listed file disappears, or
if a listed file stops containing the symbols that made it evidence for its
category in the first place. The last one matters most: a file can be gutted
while keeping its name, and a manifest that only checks paths would not notice.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.compat_manifest import COMPATIBILITY_SUITE, compatibility_paths


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Losing any of these means the product's safety story stops being verified on
# an interpreter it claims to support.
REQUIRED_CATEGORIES = frozenset(
    {
        "domain-and-serialization",
        "worker-and-isolation",
        "tool-gateway-and-fail-closed",
        "prerequisites-confirmation-followup",
        "replay-and-source-integrity",
        "adapters",
        "cli-and-import",
    }
)


def test_every_required_category_is_present() -> None:
    missing = REQUIRED_CATEGORIES - set(COMPATIBILITY_SUITE)
    assert not missing, (
        f"the cross-version suite no longer covers {sorted(missing)}. Those "
        "categories are the interpreter-facing surfaces; dropping one means "
        "3.10 and 3.11 stop verifying it."
    )


@pytest.mark.parametrize("category", sorted(REQUIRED_CATEGORIES))
def test_category_has_files_and_a_stated_reason(category: str) -> None:
    entry = COMPATIBILITY_SUITE[category]
    assert entry["files"], f"{category} has no test files"
    assert str(entry["why"]).strip(), (
        f"{category} has no stated reason. If it is not clear why the category "
        "is interpreter-sensitive, it should not be costing two extra CI runs."
    )


@pytest.mark.parametrize("path", compatibility_paths())
def test_listed_file_exists(path: str) -> None:
    assert (REPOSITORY_ROOT / path).is_file(), (
        f"{path} is in the cross-version manifest but does not exist. Either "
        "restore it or move its coverage to another file in the same category."
    )


@pytest.mark.parametrize("category", sorted(REQUIRED_CATEGORIES))
def test_each_file_still_evidences_its_category(category: str) -> None:
    """A file can keep its name and lose its meaning; check the content."""

    entry = COMPATIBILITY_SUITE[category]
    symbols = tuple(entry["symbols"])  # type: ignore[arg-type]
    for path in entry["files"]:  # type: ignore[union-attr]
        text = (REPOSITORY_ROOT / str(path)).read_text(encoding="utf-8")
        assert any(symbol in text for symbol in symbols), (
            f"{path} is listed under {category} but no longer mentions any of "
            f"{list(symbols)}. It may have been rewritten into something that "
            "no longer provides that category's cross-version evidence."
        )


def test_manifest_matches_what_ci_runs() -> None:
    """The workflow must run exactly the manifest, not a drifting copy of it."""

    workflow = (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "compat_manifest" in workflow, (
        "ci.yml no longer derives the compatibility suite from "
        "tests/compat_manifest.py. Hard-coding the list in the workflow lets the "
        "manifest and the matrix drift apart silently."
    )
