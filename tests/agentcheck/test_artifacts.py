from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from agentcheck.artifacts import ArtifactStore
from agentcheck.domain import Scenario
from agentcheck.generate import build_account_support_suite


def test_artifacts_are_private_redacted_and_valid(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, ".agentcheck", "run-1")
    output = store.write_json(
        "summary.json",
        {"authorization": "Bearer secret-value", "message": "api_key=secret-value"},
    )

    value = json.loads(output.read_text(encoding="utf-8"))
    assert value == {"authorization": "[REDACTED]", "message": "api_key=[REDACTED]"}
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.root.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.root.parent.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_jsonl_has_one_redacted_object_per_line(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, ".agentcheck", "run-2")

    output = store.write_jsonl("runs.jsonl", [{"id": 1}, {"password": "nope"}])

    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"id": 1},
        {"password": "[REDACTED]"},
    ]


@pytest.mark.parametrize("run_id", ("../escape", "..", "contains/slash", ""))
def test_artifact_run_id_cannot_escape_run_directory(
    tmp_path: Path,
    run_id: str,
) -> None:
    with pytest.raises(ValueError, match="safe single path component"):
        ArtifactStore(tmp_path, ".agentcheck", run_id)


def test_artifact_symlink_cannot_escape_target(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / ".agentcheck").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="remain inside"):
        ArtifactStore(tmp_path, ".agentcheck", "safe-run")


def test_artifact_runs_symlink_cannot_escape_target(tmp_path: Path) -> None:
    artifact_root = tmp_path / ".agentcheck"
    artifact_root.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-runs-outside"
    outside.mkdir()
    (artifact_root / "runs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="runs directory must remain inside"):
        ArtifactStore(tmp_path, ".agentcheck", "safe-run")


def test_complete_suite_artifact_is_not_cut_off_by_telemetry_budget(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path, ".agentcheck", "complete-suite")
    scenarios = build_account_support_suite()

    output = store.write_json("suite.json", {"scenarios": scenarios})
    raw = output.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert "[TRUNCATED" not in raw
    assert len(payload["scenarios"]) == 12
    assert payload["scenarios"][-1]["scenario_id"] == "ambiguous_delete_clarification"
    restored = tuple(
        Scenario.model_validate_json(json.dumps(item)) for item in payload["scenarios"]
    )
    assert [item.fingerprint for item in restored] == [
        item.fingerprint for item in scenarios
    ]
