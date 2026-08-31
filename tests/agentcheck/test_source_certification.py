from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import pytest
from pydantic import ValidationError

import agentcheck.application as application
from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig, load_config
from agentcheck.domain import (
    AssertionResult,
    CanonicalRun,
    CaseEvaluation,
    RunTermination,
    Scenario,
    Verdict,
    utc_now,
)
from agentcheck.errors import ConfigurationError
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.inspect import load_target
from agentcheck.replay import load_replay_manifest
from agentcheck.replay import bind as replay_bind
from agentcheck.replay.fileset import git_command_env
from agentcheck.runner.orchestrator import ProcessResult


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"


def _git(target: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(target), *args],
        check=True,
        capture_output=True,
        text=True,
        env=git_command_env(),
    )


def _copy_committed_target(
    tmp_path: Path,
    *,
    import_helper: bool = False,
    ignore_helper: bool = False,
    create_helper_before_commit: bool = True,
) -> tuple[Path, Path]:
    target = tmp_path / "target"
    shutil.copytree(EXAMPLE, target, symlinks=False)
    helper = target / "runtime_helper.py"
    if import_helper:
        source = target / "agent.py"
        source.write_text(
            source.read_text(encoding="utf-8").replace(
                "import json\n", "import json\nimport runtime_helper\n", 1
            ),
            encoding="utf-8",
        )
    if ignore_helper:
        (target / ".gitignore").write_text(
            ".agentcheck/\nruntime_helper.py\n", encoding="utf-8"
        )
    if create_helper_before_commit:
        helper.write_text("VALUE = 'initial'\n", encoding="utf-8")
    _git(target, "init", "-q")
    _git(target, "add", ".")
    _git(
        target,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=AgentCheck Test",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    if not create_helper_before_commit:
        helper.write_text("VALUE = 'outside-commit'\n", encoding="utf-8")
    return target, helper


def _copy_committed_package_target(tmp_path: Path) -> tuple[Path, Path]:
    target = tmp_path / "target"
    package = target / "pkg"
    package.mkdir(parents=True)
    (target / "agent.py").write_text(
        "from pkg import helper\nagent = helper.VALUE\n", encoding="utf-8"
    )
    (target / "agentcheck.json").write_text(
        '{"schema_version":"agentcheck.config.v1","entrypoint":"agent.py:agent"}\n',
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helper.py").write_text("VALUE = 'committed'\n", encoding="utf-8")
    _git(target, "init", "-q")
    _git(target, "add", ".")
    _git(
        target,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=AgentCheck Test",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    return target, package


def _example_spec() -> object:
    loaded, source = load_target(EXAMPLE)
    return OpenAIAgentsAdapter().inspect(loaded, source=source)


def _passing_evaluation(scenario: Scenario, run: CanonicalRun) -> CaseEvaluation:
    now = utc_now()
    return CaseEvaluation(
        evaluation_id=f"evaluation-{run.run_id}",
        scenario_id=scenario.scenario_id,
        run_id=run.run_id,
        verdict=Verdict.PASS,
        assertions=(
            AssertionResult(
                assertion_id="source-certification-test",
                criterion="controlled test evaluation passes",
                result=Verdict.PASS,
                oracle_ids=("source-certification-test",),
                rationale="This test isolates source certification ordering.",
            ),
        ),
        started_at=now,
        completed_at=now,
        summary="Controlled passing evaluation.",
    )


def _install_fast_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run: Callable[..., ProcessResult[CanonicalRun]] | None = None,
) -> Scenario:
    spec = _example_spec()
    scenario = build_account_support_suite()[0]

    def default_run(
        _root: Path,
        _config: AgentCheckConfig,
        case: Scenario,
        case_run_id: str,
        *,
        expected_target_id: str,
    ) -> ProcessResult[CanonicalRun]:
        now = utc_now()
        return ProcessResult(
            value=CanonicalRun(
                run_id=case_run_id,
                scenario_id=case.scenario_id,
                target_id=expected_target_id,
                started_at=now,
                ended_at=now,
                termination=RunTermination.COMPLETED,
                initial_world_state=case.initial_world_state,
                final_world_state=case.initial_world_state,
            ),
            infrastructure_error=None,
        )

    monkeypatch.setattr(
        application,
        "inspect_in_subprocess",
        lambda *_args, **_kwargs: ProcessResult(
            value=spec,  # type: ignore[arg-type]
            infrastructure_error=None,
        ),
    )
    monkeypatch.setattr(application, "_suite", lambda *_args: (scenario,))
    monkeypatch.setattr(application, "run_scenario_in_subprocess", run or default_run)
    monkeypatch.setattr(application, "evaluate_run", _passing_evaluation)
    return scenario


@pytest.mark.parametrize(
    "change",
    [
        "modified",
        "staged",
        "deleted",
        "staged_deleted",
        "staged_renamed_to_excluded",
    ],
)
def test_tracked_source_change_refuses_before_target_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    target, helper = _copy_committed_target(tmp_path, import_helper=True)
    if change in {"deleted", "staged_deleted"}:
        helper.unlink()
        if change == "staged_deleted":
            _git(target, "add", "-u", "--", "runtime_helper.py")
    elif change == "staged_renamed_to_excluded":
        (target / "dist").mkdir()
        _git(target, "mv", "runtime_helper.py", "dist/runtime_helper.py")
    else:
        helper.write_text("VALUE = 'changed'\n", encoding="utf-8")
        if change == "staged":
            _git(target, "add", "runtime_helper.py")

    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> None:
        calls.append("called")
        raise AssertionError(
            "source refusal must precede target inspection and workers"
        )

    monkeypatch.setattr(application, "inspect_in_subprocess", forbidden)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", forbidden)

    with pytest.raises(ConfigurationError, match="source|commit"):
        application.execute_suite(target, run_id=f"tracked-{change}")

    assert calls == []
    assert not (target / ".agentcheck" / "runs" / f"tracked-{change}").exists()


def test_nonignored_untracked_import_refuses_before_target_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, _helper = _copy_committed_target(
        tmp_path,
        import_helper=True,
        create_helper_before_commit=False,
    )
    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> None:
        calls.append("called")
        raise AssertionError("untracked imported source must refuse before inspection")

    monkeypatch.setattr(application, "inspect_in_subprocess", forbidden)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", forbidden)

    with pytest.raises(ConfigurationError, match="untracked source.*runtime_helper.py"):
        application.execute_suite(target, run_id="untracked-source")

    assert calls == []
    assert not (target / ".agentcheck" / "runs" / "untracked-source").exists()


@pytest.mark.skipif(os.name != "posix", reason="directory symlink probe")
@pytest.mark.parametrize(
    "with_venv_marker", [False, True], ids=["plain", "venv-marker"]
)
def test_outbound_intermediate_directory_symlink_refuses_before_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_venv_marker: bool,
) -> None:
    target, package = _copy_committed_package_target(tmp_path)
    outside = tmp_path / "outside-package"
    package.rename(outside)
    if with_venv_marker:
        (outside / "helper.py").write_text(
            "VALUE = 'outside-and-different'\n", encoding="utf-8"
        )
        (outside / "pyvenv.cfg").write_text("version = 3.12\n", encoding="utf-8")
    os.symlink(outside, package, target_is_directory=True)
    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> None:
        calls.append("inspect")
        raise AssertionError("outbound source symlink must refuse before inspection")

    monkeypatch.setattr(application, "inspect_in_subprocess", forbidden)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", forbidden)

    with pytest.raises(ConfigurationError, match="symlink|outside|contain"):
        application.execute_suite(
            target,
            run_id=f"outbound-directory-symlink-{with_venv_marker}",
            persist_store=False,
        )

    assert calls == []


def test_untracked_venv_marker_cannot_hide_tracked_source_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, package = _copy_committed_package_target(tmp_path)
    (package / "helper.py").write_text(
        "VALUE = 'modified-and-executed'\n", encoding="utf-8"
    )
    (package / "pyvenv.cfg").write_text("version = 3.12\n", encoding="utf-8")
    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> None:
        calls.append("inspect")
        raise AssertionError("tracked source mismatch must refuse before inspection")

    monkeypatch.setattr(application, "inspect_in_subprocess", forbidden)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", forbidden)

    with pytest.raises(ConfigurationError, match="tracked source bytes differ"):
        application.execute_suite(
            target,
            run_id="untracked-venv-marker",
            persist_store=False,
        )

    assert calls == []


@pytest.mark.parametrize(
    "tracked_package_init", [True, False], ids=["regular-package", "namespace-package"]
)
def test_venv_marker_cannot_hide_ignored_relevant_source(
    tmp_path: Path, tracked_package_init: bool
) -> None:
    target = tmp_path / "target"
    package = target / "pkg"
    package.mkdir(parents=True)
    (target / "agent.py").write_text(
        "from pkg import ignored_runtime\nagent = ignored_runtime.VALUE\n",
        encoding="utf-8",
    )
    (target / "agentcheck.json").write_text(
        '{"schema_version":"agentcheck.config.v1","entrypoint":"agent.py:agent"}\n',
        encoding="utf-8",
    )
    (target / ".gitignore").write_text("pkg/ignored_runtime.py\n", encoding="utf-8")
    if tracked_package_init:
        (package / "__init__.py").write_text("", encoding="utf-8")
    _git(target, "init", "-q")
    _git(target, "add", ".")
    _git(
        target,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=AgentCheck Test",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    ignored = package / "ignored_runtime.py"
    ignored.write_text("VALUE = 'ignored-runtime-source'\n", encoding="utf-8")
    (package / "pyvenv.cfg").write_text("version = 3.12\n", encoding="utf-8")
    assert _git(target, "check-ignore", "pkg/ignored_runtime.py").stdout.strip()

    root, config = load_config(target)
    snapshot = replay_bind.capture_source_snapshot(root, config)
    entries = {item.path: item.digest for item in snapshot.file_set.files}

    assert entries["pkg/ignored_runtime.py"] == (
        "sha256:" + hashlib.sha256(ignored.read_bytes()).hexdigest()
    )
    assert snapshot.commit_bindable is False
    assert snapshot.commit_unbound_paths == ("pkg/ignored_runtime.py",)


def test_live_gitignore_cannot_hide_untracked_namespace_source(tmp_path: Path) -> None:
    target = tmp_path / "target"
    package = target / "pkg"
    package.mkdir(parents=True)
    (target / "agent.py").write_text(
        "from pkg import ignored_runtime\nagent = ignored_runtime.VALUE\n",
        encoding="utf-8",
    )
    (target / "agentcheck.json").write_text(
        '{"schema_version":"agentcheck.config.v1","entrypoint":"agent.py:agent"}\n',
        encoding="utf-8",
    )
    gitignore = target / ".gitignore"
    gitignore.write_text("# committed ignore policy\n", encoding="utf-8")
    _git(target, "init", "-q")
    _git(target, "add", ".")
    _git(
        target,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=AgentCheck Test",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    gitignore.write_text("pkg/\n", encoding="utf-8")
    ignored = package / "ignored_runtime.py"
    ignored.write_text("VALUE = 'live-ignore-source'\n", encoding="utf-8")
    (package / "pyvenv.cfg").write_text("version = 3.12\n", encoding="utf-8")
    assert _git(target, "check-ignore", "pkg/ignored_runtime.py").stdout.strip()

    root, config = load_config(target)
    snapshot = replay_bind.capture_source_snapshot(root, config)
    entries = {item.path: item.digest for item in snapshot.file_set.files}

    assert entries["pkg/ignored_runtime.py"] == (
        "sha256:" + hashlib.sha256(ignored.read_bytes()).hexdigest()
    )
    assert snapshot.commit_bindable is False
    assert snapshot.commit_unbound_paths == ("pkg/ignored_runtime.py",)


def test_git_replace_ref_cannot_redefine_named_revision_bytes(tmp_path: Path) -> None:
    target, package = _copy_committed_package_target(tmp_path)
    original_revision = _git(target, "rev-parse", "HEAD").stdout.strip()
    (package / "helper.py").write_text(
        "VALUE = 'replacement-object'\n", encoding="utf-8"
    )
    _git(target, "add", "pkg/helper.py")
    _git(
        target,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=AgentCheck Test",
        "commit",
        "-q",
        "-m",
        "replacement",
    )
    replacement_revision = _git(target, "rev-parse", "HEAD").stdout.strip()
    _git(target, "checkout", "-q", "--detach", original_revision)
    (package / "helper.py").write_text(
        "VALUE = 'replacement-object'\n", encoding="utf-8"
    )
    _git(target, "replace", original_revision, replacement_revision)
    assert _git(target, "replace", "-l").stdout.strip() == original_revision

    root, config = load_config(target)
    with pytest.raises(ConfigurationError, match="tracked source bytes differ"):
        replay_bind.capture_source_snapshot(root, config)


def test_ignored_relevant_source_is_exactly_bound_but_not_commit_bindable(
    tmp_path: Path,
) -> None:
    target, helper = _copy_committed_target(
        tmp_path,
        import_helper=True,
        ignore_helper=True,
        create_helper_before_commit=False,
    )
    root, config = load_config(target)

    assert _git(target, "status", "--porcelain").stdout == ""
    assert (
        subprocess.run(
            ["git", "-C", str(target), "cat-file", "-e", "HEAD:./runtime_helper.py"],
            check=False,
            capture_output=True,
            env=git_command_env(),
        ).returncode
        != 0
    )

    snapshot = replay_bind.capture_source_snapshot(root, config)
    entries = {item.path: item.digest for item in snapshot.file_set.files}

    assert snapshot.schema_version == "agentcheck.source_snapshot.v1"
    assert snapshot.commit_bindable is False
    assert snapshot.commit_unbound_paths == ("runtime_helper.py",)
    assert entries["runtime_helper.py"] == (
        "sha256:" + hashlib.sha256(helper.read_bytes()).hexdigest()
    )


def test_clean_tracked_source_is_byte_identical_and_commit_bindable(
    tmp_path: Path,
) -> None:
    target, _helper = _copy_committed_target(tmp_path, import_helper=True)
    root, config = load_config(target)

    snapshot = replay_bind.capture_source_snapshot(root, config)

    assert snapshot.commit_bindable is True
    assert snapshot.commit_unbound_paths == ()
    assert snapshot.git_revision == _git(target, "rev-parse", "HEAD").stdout.strip()
    for entry in snapshot.file_set.files:
        committed = subprocess.run(
            [
                "git",
                "-C",
                str(target),
                "cat-file",
                "blob",
                f"{snapshot.git_revision}:./{entry.path}",
            ],
            check=True,
            capture_output=True,
            env=git_command_env(),
        ).stdout
        assert entry.digest == "sha256:" + hashlib.sha256(committed).hexdigest()


@pytest.mark.parametrize("callback_name", ["on_inspected", "on_prepared"])
def test_inspection_and_preparation_mutation_refuses_before_scenario_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    callback_name: str,
) -> None:
    target, helper = _copy_committed_target(tmp_path, import_helper=True)
    _install_fast_execution(monkeypatch)
    worker_calls: list[str] = []
    callback_calls: list[str] = []

    def forbidden_worker(*_args: object, **_kwargs: object) -> None:
        worker_calls.append("worker")
        raise AssertionError("phase mismatch must refuse before scenario workers")

    def mutate(*_args: object, **_kwargs: object) -> None:
        callback_calls.append(callback_name)
        helper.write_text("VALUE = 'initial'\n# phase mutation\n", encoding="utf-8")

    monkeypatch.setattr(application, "run_scenario_in_subprocess", forbidden_worker)

    with pytest.raises(ConfigurationError, match="source|commit"):
        application.execute_suite(
            target,
            run_id=f"phase-{callback_name}",
            persist_store=False,
            **{callback_name: mutate},
        )

    assert worker_calls == []
    assert callback_calls == [callback_name]
    assert helper.read_text(encoding="utf-8").endswith("# phase mutation\n")
    assert not (target / ".agentcheck" / "runs" / f"phase-{callback_name}").exists()


def test_during_worker_mutation_refuses_after_passing_behavior_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, helper = _copy_committed_target(tmp_path, import_helper=True)
    worker_calls: list[str] = []
    evaluated_verdicts: list[Verdict] = []

    def mutating_run(
        _root: Path,
        _config: AgentCheckConfig,
        scenario: Scenario,
        case_run_id: str,
        *,
        expected_target_id: str,
    ) -> ProcessResult[CanonicalRun]:
        worker_calls.append(scenario.scenario_id)
        helper.write_text("VALUE = 'initial'\n# worker mutation\n", encoding="utf-8")
        now = utc_now()
        return ProcessResult(
            value=CanonicalRun(
                run_id=case_run_id,
                scenario_id=scenario.scenario_id,
                target_id=expected_target_id,
                started_at=now,
                ended_at=now,
                termination=RunTermination.COMPLETED,
                initial_world_state=scenario.initial_world_state,
                final_world_state=scenario.initial_world_state,
            ),
            infrastructure_error=None,
        )

    scenario = _install_fast_execution(monkeypatch, run=mutating_run)

    def recorded_passing_evaluation(
        evaluated_scenario: Scenario, run: CanonicalRun
    ) -> CaseEvaluation:
        result = _passing_evaluation(evaluated_scenario, run)
        evaluated_verdicts.append(result.verdict)
        return result

    monkeypatch.setattr(application, "evaluate_run", recorded_passing_evaluation)

    with pytest.raises(ConfigurationError, match="source|commit"):
        application.execute_suite(
            target,
            run_id="during-worker-mutation",
            persist_store=False,
        )

    assert worker_calls == [scenario.scenario_id]
    assert evaluated_verdicts == [Verdict.PASS]
    assert not (target / ".agentcheck" / "runs" / "during-worker-mutation").exists()
    assert not (
        target / ".agentcheck" / "replay" / "during-worker-mutation.json"
    ).exists()


def test_clean_execution_replay_uses_initial_snapshot_and_outputs_do_not_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target, _helper = _copy_committed_target(tmp_path, import_helper=True)
    _install_fast_execution(monkeypatch)
    emitted_snapshots: list[object] = []
    captured_snapshots: list[object] = []
    events: list[str] = []
    original_emit = application._emit_replay_manifest
    original_capture = application.capture_source_snapshot
    original_inspect = application.inspect_in_subprocess

    def traced_capture(*args: object, **kwargs: object) -> object:
        snapshot = original_capture(*args, **kwargs)  # type: ignore[arg-type]
        captured_snapshots.append(snapshot)
        events.append("capture")
        return snapshot

    def traced_inspect(*args: object, **kwargs: object) -> object:
        events.append("inspect")
        return original_inspect(*args, **kwargs)

    def traced_emit(**kwargs: object) -> Path | None:
        emitted_snapshots.append(kwargs["source_snapshot"])
        events.append("emit")
        return original_emit(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(application, "capture_source_snapshot", traced_capture)
    monkeypatch.setattr(replay_bind, "capture_source_snapshot", traced_capture)
    monkeypatch.setattr(application, "inspect_in_subprocess", traced_inspect)
    monkeypatch.setattr(application, "_emit_replay_manifest", traced_emit)
    first = application.execute_suite(
        target,
        run_id="source-snapshot-clean-1",
        persist_store=False,
    )
    second = application.execute_suite(
        target,
        run_id="source-snapshot-clean-2",
        persist_store=False,
    )

    assert first.source_snapshot.commit_bindable is True
    assert first.source_snapshot == second.source_snapshot
    assert events == [
        "capture",
        "inspect",
        "capture",
        "capture",
        "emit",
        "capture",
        "inspect",
        "capture",
        "capture",
        "emit",
    ]
    assert len(captured_snapshots) == 6
    assert emitted_snapshots == [first.source_snapshot, second.source_snapshot]
    assert emitted_snapshots[0] is captured_snapshots[0]
    assert emitted_snapshots[0] is not captured_snapshots[1]
    assert emitted_snapshots[0] is not captured_snapshots[2]
    assert emitted_snapshots[1] is captured_snapshots[3]
    assert first.replay_manifest_path is not None
    manifest = load_replay_manifest(
        target,
        first.replay_manifest_path.relative_to(target).as_posix(),
    )
    assert manifest.source_binding.git_revision == first.source_snapshot.git_revision
    assert manifest.source_binding.entrypoint_digest == (
        first.source_snapshot.entrypoint_digest
    )
    assert manifest.source_binding.file_set == first.source_snapshot.file_set

    (target / "agentcheck-suite.json").write_text("{}\n", encoding="utf-8")
    (target / "agentcheck-baseline.json").write_text("{}\n", encoding="utf-8")
    replay_bind.verify_source_snapshot(
        first.source_snapshot,
        root=target,
        config=first.config,
        phase="after generated outputs",
    )


def test_source_snapshot_contract_rejects_tampering_and_unknown_versions(
    tmp_path: Path,
) -> None:
    target, _helper = _copy_committed_target(tmp_path, import_helper=True)
    root, config = load_config(target)
    snapshot = replay_bind.capture_source_snapshot(root, config)
    snapshot_type = type(snapshot)

    document = snapshot.model_dump(mode="json")
    document["unexpected"] = True
    with pytest.raises(ValidationError):
        snapshot_type.model_validate_json(json.dumps(document))

    document = snapshot.model_dump(mode="json")
    document["schema_version"] = "agentcheck.source_snapshot.v0"
    with pytest.raises(ValidationError):
        snapshot_type.model_validate_json(json.dumps(document))

    document = snapshot.model_dump(mode="json")
    document["fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="fingerprint"):
        snapshot_type.model_validate_json(json.dumps(document))

    document = snapshot.model_dump(mode="json")
    document["commit_bindable"] = False
    with pytest.raises(ValidationError, match="commit_bindable"):
        snapshot_type.model_validate_json(json.dumps(document))

    document = snapshot.model_dump(mode="json")
    document["commit_unbound_paths"] = ["runtime_helper.py"]
    with pytest.raises(ValidationError, match="commit_bindable"):
        snapshot_type.model_validate_json(json.dumps(document))

    document = snapshot.model_dump(mode="json")
    document["commit_bindable"] = False
    document["commit_unbound_paths"] = ["runtime_helper.py"]
    with pytest.raises(ValidationError, match="fingerprint"):
        snapshot_type.model_validate_json(json.dumps(document))

    ignored_parent = tmp_path / "ignored"
    ignored_parent.mkdir()
    ignored_target, _ignored_helper = _copy_committed_target(
        ignored_parent,
        import_helper=True,
        ignore_helper=True,
        create_helper_before_commit=False,
    )
    ignored_root, ignored_config = load_config(ignored_target)
    ignored = replay_bind.capture_source_snapshot(ignored_root, ignored_config)
    assert ignored.commit_bindable is False
    assert ignored.commit_unbound_paths == ("runtime_helper.py",)

    document = ignored.model_dump(mode="json")
    document["commit_bindable"] = True
    with pytest.raises(ValidationError, match="commit_bindable"):
        snapshot_type.model_validate_json(json.dumps(document))

    document = ignored.model_dump(mode="json")
    document["commit_unbound_paths"] = []
    with pytest.raises(ValidationError, match="commit_bindable"):
        snapshot_type.model_validate_json(json.dumps(document))

    document = ignored.model_dump(mode="json")
    document["commit_bindable"] = True
    document["commit_unbound_paths"] = []
    with pytest.raises(ValidationError, match="fingerprint"):
        snapshot_type.model_validate_json(json.dumps(document))
