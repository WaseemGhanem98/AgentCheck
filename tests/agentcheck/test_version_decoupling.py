"""Releasing the package must not re-identify suites that assert the same thing.

A frozen suite's fingerprint covers its provenance, and provenance recorded the
package release. So publishing 0.1.1 -> 0.2.0 moved every ``suite_id`` while the
generated cases stayed byte-identical: identity tracked which wheel wrote the
document rather than what the document asserts, and a routine release was
indistinguishable from a behavioural change.

Generation semantics now carry their own version. These tests pin the boundary
in both directions -- a release must not move identity, and a semantics bump
must.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, function_tool

import agentcheck
import agentcheck.generate.suite as suite_module
from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.generate.suite import (
    GENERATOR_COMPATIBILITY_VERSION,
    build_frozen_suite,
    encode_frozen_suite,
    load_frozen_suite,
)


SEED = 1729


@function_tool
def delete_record(record_id: str, reason: str) -> str:
    """Delete a record permanently. This removes it for good."""
    raise AssertionError("original handler must never run")


@function_tool
def lookup_record(record_id: str) -> str:
    """Look up a stored record."""
    raise AssertionError("original handler must never run")


def _spec(*tools: Any):
    return OpenAIAgentsAdapter().inspect(
        Agent(name="T", instructions="Assist.", tools=list(tools), model="gpt-4.1-mini")
    )


def _suite(spec: Any):
    return build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)


@pytest.fixture
def released(monkeypatch: pytest.MonkeyPatch):
    """Simulate publishing a new package release and nothing else."""

    def _release(version: str) -> None:
        monkeypatch.setattr(agentcheck, "__version__", version, raising=False)
        monkeypatch.setattr(suite_module, "__version__", version, raising=False)

    return _release


# --- a release must not move identity --------------------------------------


def test_a_package_release_alone_does_not_move_suite_identity(released) -> None:
    spec = _spec(delete_record, lookup_record)

    before = _suite(spec)
    released("9.9.9")
    after = _suite(spec)

    assert after.fingerprint == before.fingerprint
    assert after.suite_id == before.suite_id


def test_a_package_release_alone_does_not_move_scenario_fingerprints(released) -> None:
    spec = _spec(delete_record, lookup_record)

    before = [case.scenario.fingerprint for case in _suite(spec).cases]
    released("9.9.9")
    after = [case.scenario.fingerprint for case in _suite(spec).cases]

    assert after == before


def test_a_package_release_alone_does_not_move_the_spec_identity(released) -> None:
    """The target did not change, so neither may the identity derived from it."""

    before = _spec(delete_record).spec_id
    released("9.9.9")
    after = _spec(delete_record).spec_id

    assert after == before


def test_provenance_records_generation_semantics_not_the_release(released) -> None:
    spec = _spec(delete_record)

    released("9.9.9")
    provenance = _suite(spec).provenance

    assert provenance.generator_version == GENERATOR_COMPATIBILITY_VERSION
    assert provenance.generator_version != agentcheck.__version__


# --- a semantics bump must move exactly the intended boundary ---------------


def test_a_generator_semantics_bump_moves_suite_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(delete_record, lookup_record)
    before = _suite(spec)

    monkeypatch.setattr(suite_module, "GENERATOR_COMPATIBILITY_VERSION", "2")
    after = _suite(spec)

    assert after.fingerprint != before.fingerprint
    assert after.suite_id != before.suite_id
    assert after.provenance.generator_version == "2"


def test_a_generator_semantics_bump_does_not_reach_the_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The boundary is suite identity. Individual cases are unchanged data."""

    spec = _spec(delete_record, lookup_record)
    before = [case.scenario.fingerprint for case in _suite(spec).cases]

    monkeypatch.setattr(suite_module, "GENERATOR_COMPATIBILITY_VERSION", "2")
    after = [case.scenario.fingerprint for case in _suite(spec).cases]

    assert after == before


# --- artifacts already on disk ---------------------------------------------


def test_a_suite_frozen_by_an_older_release_still_loads(
    tmp_path: Path, released
) -> None:
    """A stored suite validates against what it recorded, not against today."""

    spec = _spec(delete_record)
    legacy = json.loads(encode_frozen_suite(_suite(spec)))
    # Exactly the shape releases used to write: the package version verbatim.
    legacy["provenance"]["generator_version"] = "0.1.0"
    legacy["fingerprint"] = ""
    legacy["suite_id"] = ""
    path = tmp_path / "legacy-suite.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    rebuilt = load_frozen_suite(path)
    stored = tmp_path / "stored.json"
    stored.write_bytes(encode_frozen_suite(rebuilt))

    released("9.9.9")
    reloaded = load_frozen_suite(stored)

    assert reloaded.provenance.generator_version == "0.1.0"
    assert reloaded.fingerprint == rebuilt.fingerprint
    assert len(reloaded.cases) == len(rebuilt.cases)


def test_a_tampered_suite_is_still_rejected(tmp_path: Path) -> None:
    """Decoupling identity must not soften integrity."""

    spec = _spec(delete_record)
    forged = json.loads(encode_frozen_suite(_suite(spec)))
    # Keep the recorded fingerprint, change what it covers.
    forged["provenance"]["generator_version"] = "not-what-was-hashed"
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(Exception) as caught:
        load_frozen_suite(path)
    assert "fingerprint" in str(caught.value).lower()


def test_a_suite_round_trips_unchanged_across_a_release(
    tmp_path: Path, released
) -> None:
    spec = _spec(delete_record, lookup_record)
    path = tmp_path / "suite.json"
    path.write_bytes(encode_frozen_suite(_suite(spec)))
    before = load_frozen_suite(path)

    released("9.9.9")
    after = load_frozen_suite(path)

    assert after.fingerprint == before.fingerprint
    assert after.suite_id == before.suite_id


# --- the release version is still reported where it belongs ----------------


def test_the_cli_still_reports_the_package_release() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "agentcheck", "--version"],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert agentcheck.__version__ in (result.stdout + result.stderr)
    assert GENERATOR_COMPATIBILITY_VERSION not in result.stdout.split()[-1:]


# --- the two hand-maintained declarations must not drift apart -------------


def test_the_release_version_is_declared_once_in_effect() -> None:
    """``pyproject.toml`` and ``agentcheck.__version__`` are edited by hand.

    Nothing generates one from the other, so a bump that touches only one leaves
    the wheel's metadata disagreeing with what ``--version`` prints and with what
    ``provenance.release_version`` records. That is silent: both files are
    individually valid.
    """
    import re

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    # Match the project table's own key, not a pinned dependency's version.
    match = re.search(r"(?m)^version = \"([^\"]+)\"$", text)
    assert match is not None, "no top-level version key in pyproject.toml"

    assert match.group(1) == agentcheck.__version__, (
        f"pyproject.toml declares {match.group(1)!r} but "
        f"agentcheck.__version__ is {agentcheck.__version__!r}"
    )


def test_contributing_docs_keep_release_and_generator_versions_separate() -> None:
    contributing = Path(__file__).resolve().parents[2] / "CONTRIBUTING.md"
    text = contributing.read_text(encoding="utf-8")
    prose = " ".join(text.split())

    assert (
        "Package releases and frozen-suite generation compatibility use separate"
        in prose
    )
    assert "A package release alone is not a suite compatibility event" in prose
    assert (
        "Package and frozen-suite generator versions are currently coupled" not in text
    )
