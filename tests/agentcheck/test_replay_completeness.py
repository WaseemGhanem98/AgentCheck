from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agentcheck.application as application
import agentcheck.cli as cli
from agentcheck.baseline.service import create_baseline
from agentcheck.config import AgentCheckConfig
from agentcheck.errors import ConfigurationError, InfrastructureError
from agentcheck.gate import EXIT_NOT_CERTIFIABLE, GateDecision, run_gate
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.replay import (
    ReplayManifest,
    SourceFileEntry,
    SourceFileSet,
    build_replay_manifest,
    load_replay_manifest,
)
from agentcheck.replay.fileset import SourceSnapshot
from agentcheck.report import load_stored_run


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
SECRET = "sk-thisisafakesecretvalue12"


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "account_agent"
    shutil.copytree(EXAMPLE, target, symlinks=False)
    return target


def _stub_spec() -> SimpleNamespace:
    return SimpleNamespace(
        spec_id="agentspec-replay-completeness",
        identity=SimpleNamespace(
            framework=SimpleNamespace(value="openai_agents"),
            framework_version=SimpleNamespace(value="0.20.0"),
        ),
    )


def _source_snapshot() -> SourceSnapshot:
    digest = "sha256:" + "a" * 64
    file_set = SourceFileSet(
        mode="local_files",
        complete=True,
        files=(SourceFileEntry(path="agent.py", digest=digest),),
    )
    return SourceSnapshot(
        git_revision=None,
        entrypoint_path="agent.py",
        entrypoint_digest=digest,
        file_set=file_set,
        commit_bindable=False,
        commit_unbound_paths=("agent.py",),
    )


def _taint(scenario_index: int) -> Any:
    scenario = build_account_support_suite(seed=1729)[scenario_index]
    payload = json.loads(scenario.model_dump_json())
    payload["conversation_turns"][0]["content"] = SECRET
    payload["fingerprint"] = ""
    return type(scenario).model_validate_json(json.dumps(payload))


def _require_replay(
    root: Path,
    scenarios: tuple[Any, ...],
    *,
    run_id: str = "complete-replay",
) -> Path:
    return application._require_complete_replay_manifest(
        root=root,
        config=AgentCheckConfig(),
        spec=_stub_spec(),  # type: ignore[arg-type]
        run_id=run_id,
        seed=1729,
        scenarios=scenarios,
        source_snapshot=_source_snapshot(),
        policy_pack_ids=(),
    )


def test_complete_replay_uses_the_initial_snapshot_and_every_case(
    tmp_path: Path,
) -> None:
    scenarios = tuple(build_account_support_suite(seed=1729)[:2])

    replay_path = _require_replay(tmp_path, scenarios)
    manifest = load_replay_manifest(
        tmp_path,
        replay_path.relative_to(tmp_path).as_posix(),
    )

    assert manifest.omitted == ()
    assert tuple(
        (scenario.scenario_id, scenario.fingerprint) for scenario in manifest.cases
    ) == tuple((scenario.scenario_id, scenario.fingerprint) for scenario in scenarios)
    assert manifest.source_binding.entrypoint_digest == _source_snapshot().entrypoint_digest
    assert manifest.source_binding.file_set == _source_snapshot().file_set


@pytest.mark.parametrize("include_clean_case", [False, True])
def test_secret_omissions_refuse_before_any_replay_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_clean_case: bool,
) -> None:
    suite = build_account_support_suite(seed=1729)
    scenarios = ((_taint(1),) if not include_clean_case else (suite[0], _taint(1)))
    writer_called = False

    def record_write(*_args: object, **_kwargs: object) -> Path:
        nonlocal writer_called
        writer_called = True
        return tmp_path / "unexpected.json"

    monkeypatch.setattr(application, "write_replay_manifest", record_write)

    with pytest.raises(ConfigurationError, match="complete replay manifest") as exc_info:
        _require_replay(tmp_path, scenarios)

    assert writer_called is False
    assert SECRET not in str(exc_info.value)
    assert not (tmp_path / ".agentcheck").exists()


def test_case_mismatch_refuses_even_when_builder_reports_no_omissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = tuple(build_account_support_suite(seed=1729)[:2])
    real_builder = application.build_replay_manifest
    writer_called = False

    def incomplete_builder(**kwargs: Any) -> tuple[ReplayManifest | None, tuple[Any, ...]]:
        kwargs["scenarios"] = kwargs["scenarios"][:1]
        return real_builder(**kwargs)

    def record_write(*_args: object, **_kwargs: object) -> Path:
        nonlocal writer_called
        writer_called = True
        return tmp_path / "unexpected.json"

    monkeypatch.setattr(application, "build_replay_manifest", incomplete_builder)
    monkeypatch.setattr(application, "write_replay_manifest", record_write)

    with pytest.raises(ConfigurationError, match="exactly once"):
        _require_replay(tmp_path, scenarios)

    assert writer_called is False


def test_unexpected_replay_failure_is_redacted_and_propagated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(**_kwargs: Any) -> Any:
        raise RuntimeError(f"writer exposed {SECRET}")

    monkeypatch.setattr(application, "build_replay_manifest", explode)

    with pytest.raises(ConfigurationError, match="unable to produce") as exc_info:
        _require_replay(tmp_path, (build_account_support_suite(seed=1729)[0],))

    assert SECRET not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_forced_replay_write_failure_leaves_no_loadable_or_baselineable_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _copy_example(tmp_path)
    run_id = "forced-replay-write-failure"
    persisted = False

    def fail_replay_write(*_args: object, **_kwargs: object) -> Path:
        raise ConfigurationError("forced replay write failure")

    def record_persist(*_args: object, **_kwargs: object) -> None:
        nonlocal persisted
        persisted = True

    monkeypatch.setattr(application, "write_replay_manifest", fail_replay_write)
    monkeypatch.setattr(application, "_persist_execution", record_persist)

    with pytest.raises(ConfigurationError, match="forced replay write failure"):
        run_gate(target, run_id=run_id, persist_store=True)

    run_path = target / ".agentcheck" / "runs" / run_id
    replay_path = target / ".agentcheck" / "replay" / f"{run_id}.json"
    assert run_path.is_dir()
    assert tuple(run_path.iterdir()) == ()
    assert replay_path.exists() is False
    assert persisted is False
    with pytest.raises(ConfigurationError, match="missing required artifact"):
        load_stored_run(target, run_id=run_id, latest=False)
    with pytest.raises(ConfigurationError, match="missing required artifact"):
        create_baseline(target, run_id=run_id)
    assert not (target / "agentcheck-baseline.json").exists()


def test_run_id_collision_precedes_workers_and_preserves_prior_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _copy_example(tmp_path)
    run_id = "existing-run"
    run_path = target / ".agentcheck" / "runs" / run_id
    replay_path = target / ".agentcheck" / "replay" / f"{run_id}.json"
    run_path.mkdir(parents=True)
    replay_path.parent.mkdir(parents=True)
    replay_path.write_bytes(b"prior replay evidence\n")
    prior_replay = replay_path.read_bytes()

    def worker_must_not_run(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("scenario worker ran after a run-ID collision")

    monkeypatch.setattr(application, "run_scenario_in_subprocess", worker_must_not_run)

    with pytest.raises(InfrastructureError, match="unable to create run artifact"):
        application.execute_suite(target, run_id=run_id, persist_store=False)

    assert replay_path.read_bytes() == prior_replay
    assert tuple(run_path.iterdir()) == ()


def test_gate_json_reports_no_certifiable_result_after_late_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_gate(*_args: object, **_kwargs: object) -> Any:
        raise ConfigurationError("forced replay write failure")

    monkeypatch.setattr(cli, "run_gate", fail_gate)

    exit_code = cli._gate_command(
        str(tmp_path),
        baseline=None,
        seed=None,
        run_id="late-failure",
        persist_store=False,
        as_json=True,
        python_executable=None,
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == EXIT_NOT_CERTIFIABLE
    assert payload["decision"] == GateDecision.BLOCK
    assert payload["counts"] == {}
    assert any("no certifiable result" in item for item in payload["detail"])
    assert "suite did not execute" not in captured.out


def test_test_cli_redacts_replay_failure_and_exits_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_execution(*_args: object, **_kwargs: object) -> Any:
        raise ConfigurationError(f"writer exposed {SECRET}")

    monkeypatch.setattr(cli, "execute_suite", fail_execution)

    assert cli.main(["test", str(tmp_path), "--run-id", "late-failure"]) == 2
    captured = capsys.readouterr()
    assert SECRET not in captured.err
    assert "[REDACTED]" in captured.err


def test_lower_level_partial_manifest_contract_remains_readable() -> None:
    suite = build_account_support_suite(seed=1729)
    manifest, omitted = build_replay_manifest(
        run_id="legacy-partial",
        seed=1729,
        spec=_stub_spec(),  # type: ignore[arg-type]
        config=AgentCheckConfig(),
        scenarios=(suite[0], _taint(1)),
        git_revision=None,
        entrypoint_digest=_source_snapshot().entrypoint_digest,
        file_set=_source_snapshot().file_set,
    )

    assert manifest is not None
    assert len(manifest.cases) == 1
    assert len(omitted) == 1
    assert ReplayManifest.model_validate_json(manifest.model_dump_json()) == manifest
