from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentcheck.application import execute_suite, replay_suite
from agentcheck.cli import main
from agentcheck.config import AgentCheckConfig
from agentcheck.domain import Scenario, Verdict
from agentcheck.errors import ConfigurationError
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.replay import (
    MAX_MANIFEST_BYTES,
    REPLAY_MANIFEST_CONTRACT_VERSION,
    EnvironmentRequirements,
    ReplayManifest,
    SourceBinding,
    SpecBinding,
    build_replay_manifest,
    encode_replay_manifest,
    load_replay_manifest,
    secret_shaped_reason,
)
from agentcheck.replay.load import load_replay_manifest_path


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes required")
SEED = 1729


def _scenario() -> Scenario:
    return build_account_support_suite(seed=SEED)[0]


def _manifest(**updates: object) -> ReplayManifest:
    payload: dict[str, object] = dict(
        created_from_run_id="unit-replay-001",
        agentcheck_version="0.1.0",
        seed=SEED,
        spec_binding=SpecBinding(
            spec_id="agentspec-unit-test",
            adapter="openai_agents",
            entrypoint="agent.py:agent",
        ),
        source_binding=SourceBinding(
            git_revision="a" * 40,
            entrypoint_digest="sha256:" + "b" * 64,
            framework="openai_agents",
            framework_version="0.20.0",
        ),
        environment_requirements=EnvironmentRequirements(names=()),
        cases=(_scenario(),),
    )
    payload.update(updates)
    return ReplayManifest.model_validate(payload)


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "account_agent"
    shutil.copytree(EXAMPLE, target, symlinks=False)
    return target


def _stub_spec(spec_id: str = "agentspec-unit-test") -> SimpleNamespace:
    return SimpleNamespace(
        spec_id=spec_id,
        identity=SimpleNamespace(
            framework=SimpleNamespace(value="openai_agents"),
            framework_version=SimpleNamespace(value="0.20.0"),
        ),
    )


def test_replay_manifest_round_trip_is_deterministic() -> None:
    first = _manifest()
    encoded = encode_replay_manifest(first)
    loaded = ReplayManifest.model_validate_json(encoded)
    assert loaded.schema_version == REPLAY_MANIFEST_CONTRACT_VERSION
    assert loaded.fingerprint == first.fingerprint
    assert loaded.manifest_id == first.manifest_id
    assert loaded.cases[0].fingerprint == first.cases[0].fingerprint
    assert encode_replay_manifest(loaded) == encoded


def test_replay_manifest_rejects_extra_fields() -> None:
    document = json.loads(encode_replay_manifest(_manifest()))
    document["unexpected"] = True
    with pytest.raises(ValidationError):
        ReplayManifest.model_validate_json(json.dumps(document))


def test_replay_manifest_detects_document_tampering() -> None:
    document = json.loads(encode_replay_manifest(_manifest()))
    document["seed"] = 1
    with pytest.raises(ValidationError, match="fingerprint"):
        ReplayManifest.model_validate_json(json.dumps(document))


def test_replay_manifest_detects_scenario_tampering() -> None:
    document = json.loads(encode_replay_manifest(_manifest()))
    document["cases"][0]["conversation_turns"][0]["content"] = "tampered user text"
    with pytest.raises(ValidationError, match="fingerprint"):
        ReplayManifest.model_validate_json(json.dumps(document))


def test_replay_manifest_rejects_unsupported_version() -> None:
    document = json.loads(encode_replay_manifest(_manifest()))
    document["schema_version"] = "agentcheck.replay_manifest.v0"
    with pytest.raises(ValidationError):
        ReplayManifest.model_validate_json(json.dumps(document))


def test_loader_refuses_extra_fields(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    document = json.loads(encode_replay_manifest(_manifest()))
    document["unexpected"] = True
    (target / "extra.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid replay manifest"):
        load_replay_manifest(target, "extra.json")


def test_loader_refuses_unsupported_version(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "bad.json").write_text(
        json.dumps({"schema_version": "agentcheck.replay_manifest.v0"}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="unsupported replay manifest"):
        load_replay_manifest(target, "bad.json")


def test_loader_refuses_frozen_suite(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "agentcheck-suite.json").write_text(
        json.dumps({"schema_version": "agentcheck.frozen_suite.v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="frozen suites are not replay"):
        load_replay_manifest(target, "agentcheck-suite.json")


def test_loader_refuses_baseline_and_review_documents(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "agentcheck-baseline.json").write_text(
        json.dumps({"schema_version": "agentcheck.baseline.v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="evaluation baselines are not replay"):
        load_replay_manifest(target, "agentcheck-baseline.json")
    (target / "review.json").write_text(
        json.dumps({"schema_version": "agentcheck.human_review.v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="human reviews are not replay"):
        load_replay_manifest(target, "review.json")


def test_loader_refuses_disclosure_suite(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "suite.json").write_text(
        json.dumps({"schema_version": "agentcheck.suite.v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="disclosure suite.json"):
        load_replay_manifest(target, "suite.json")


def test_loader_refuses_summary_artifact(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "summary.json").write_text(
        json.dumps({"schema_version": "agentcheck.summary.v1"}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="summary artifacts"):
        load_replay_manifest(target, "summary.json")


def test_loader_refuses_sqlite(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "agentcheck.sqlite").write_bytes(b"SQLite format 3\x00" + b"\x00" * 20)
    with pytest.raises(ConfigurationError, match="SQLite"):
        load_replay_manifest(target, "agentcheck.sqlite")


def test_loader_refuses_html_report(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "report.html").write_text(
        "<!doctype html><html><body>nope</body></html>\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="HTML"):
        load_replay_manifest(target, "report.html")


@POSIX_ONLY
def test_loader_refuses_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "real.json").write_bytes(encode_replay_manifest(_manifest()))
    os.symlink(target / "real.json", target / "link.json")
    with pytest.raises(ConfigurationError, match="symlink"):
        load_replay_manifest(target, "link.json")


def test_loader_enforces_size_cap(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "huge.json").write_bytes(b"{" + b"a" * (MAX_MANIFEST_BYTES + 2))
    with pytest.raises(ConfigurationError, match="exceeds"):
        load_replay_manifest(target, "huge.json")


def test_loader_does_not_inspect_on_malformed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    target = _copy_example(tmp_path)
    (target / "broken.json").write_text("{", encoding="utf-8")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("target must not be imported for a malformed manifest")

    monkeypatch.setattr(application, "inspect_in_subprocess", explode)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    with pytest.raises(ConfigurationError, match="invalid replay manifest"):
        replay_suite(target, "broken.json")


def test_environment_names_are_stored_without_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-thisisafakesecretvalue12")
    manifest, omitted = build_replay_manifest(
        run_id="unit-replay-001",
        seed=SEED,
        spec=_stub_spec(),  # type: ignore[arg-type]
        config=AgentCheckConfig(environment_allowlist=("OPENAI_API_KEY",)),
        scenarios=(_scenario(),),
        git_revision=None,
        entrypoint_digest="sha256:" + "b" * 64,
        policy_pack_ids=(),
    )
    assert omitted == ()
    assert manifest is not None
    assert manifest.environment_requirements.names == ("OPENAI_API_KEY",)
    encoded = encode_replay_manifest(manifest).decode("utf-8")
    assert "OPENAI_API_KEY" in encoded
    assert "sk-thisisafakesecretvalue12" not in encoded


def test_secret_shaped_scenario_is_omitted_not_redacted() -> None:
    payload = json.loads(_scenario().model_dump_json())
    payload["conversation_turns"][0]["content"] = "sk-thisisafakesecretvalue12"
    payload["fingerprint"] = ""
    tainted = Scenario.model_validate_json(json.dumps(payload))
    assert secret_shaped_reason(tainted) is not None
    manifest, omitted = build_replay_manifest(
        run_id="unit-replay-001",
        seed=SEED,
        spec=_stub_spec(),  # type: ignore[arg-type]
        config=AgentCheckConfig(),
        scenarios=(tainted,),
        git_revision=None,
        entrypoint_digest="sha256:" + "b" * 64,
    )
    assert manifest is None
    assert len(omitted) == 1
    assert omitted[0].scenario_id == tainted.scenario_id


def test_secret_shaped_case_is_omitted_while_others_are_kept() -> None:
    suite = build_account_support_suite(seed=SEED)
    payload = json.loads(suite[1].model_dump_json())
    payload["conversation_turns"][0]["content"] = "sk-thisisafakesecretvalue12"
    payload["fingerprint"] = ""
    tainted = Scenario.model_validate_json(json.dumps(payload))
    manifest, omitted = build_replay_manifest(
        run_id="unit-replay-001",
        seed=SEED,
        spec=_stub_spec(),  # type: ignore[arg-type]
        config=AgentCheckConfig(),
        scenarios=(suite[0], tainted),
        git_revision=None,
        entrypoint_digest="sha256:" + "b" * 64,
    )
    assert omitted[0].scenario_id == tainted.scenario_id
    assert manifest is not None
    assert [case.scenario_id for case in manifest.cases] == [suite[0].scenario_id]
    encoded = encode_replay_manifest(manifest).decode("utf-8")
    assert "sk-thisisafakesecretvalue12" not in encoded


def test_spec_id_mismatch_is_configuration_error(tmp_path: Path) -> None:
    from agentcheck.replay.bind import entrypoint_digest
    from agentcheck.replay.fileset import collect_source_file_set

    target = _copy_example(tmp_path)
    digest = entrypoint_digest(target, "agent.py:agent")
    manifest = ReplayManifest(
        created_from_run_id="unit-replay-001",
        agentcheck_version="0.1.0",
        seed=SEED,
        spec_binding=SpecBinding(
            spec_id="agentspec-unit-test",
            adapter="openai_agents",
            entrypoint="agent.py:agent",
        ),
        source_binding=SourceBinding(
            git_revision=None,
            entrypoint_digest=digest,
            framework="openai_agents",
            framework_version="0.20.0",
            file_set=collect_source_file_set(target),
        ),
        environment_requirements=EnvironmentRequirements(names=()),
        cases=(_scenario(),),
    )
    (target / ".agentcheck" / "replay").mkdir(parents=True, exist_ok=True)
    (target / ".agentcheck" / "replay" / "replay-unit.json").write_bytes(
        encode_replay_manifest(manifest)
    )
    with pytest.raises(ConfigurationError, match="bound to spec"):
        replay_suite(target, ".agentcheck/replay/replay-unit.json")


def test_git_revision_mismatch_is_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application
    from agentcheck.adapters import OpenAIAgentsAdapter
    from agentcheck.inspect import load_target
    from agentcheck.replay import bind as replay_bind
    from agentcheck.replay.bind import entrypoint_digest

    target = _copy_example(tmp_path)
    loaded, source = load_target(target)
    spec = OpenAIAgentsAdapter().inspect(loaded, source=source)
    digest = entrypoint_digest(target, "agent.py:agent")
    manifest, omitted = build_replay_manifest(
        run_id="unit-replay-001",
        seed=SEED,
        spec=spec,
        config=AgentCheckConfig(),
        scenarios=(_scenario(),),
        git_revision="a" * 40,
        entrypoint_digest=digest,
    )
    assert omitted == ()
    assert manifest is not None
    (target / "replay-unit.json").write_bytes(encode_replay_manifest(manifest))
    monkeypatch.setattr(replay_bind, "git_revision", lambda _root: "c" * 40)
    monkeypatch.setattr(application, "_git_revision", lambda _root: "c" * 40)
    with pytest.raises(ConfigurationError, match="git revision"):
        replay_suite(target, "replay-unit.json")


def test_missing_required_environment_is_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentcheck.adapters import OpenAIAgentsAdapter
    from agentcheck.inspect import load_target
    from agentcheck.replay.bind import entrypoint_digest

    target = _copy_example(tmp_path)
    loaded, source = load_target(target)
    spec = OpenAIAgentsAdapter().inspect(loaded, source=source)
    digest = entrypoint_digest(target, "agent.py:agent")
    config = AgentCheckConfig(environment_allowlist=("OPENAI_API_KEY",))
    (target / "agentcheck.json").write_text(
        config.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
    )
    manifest, omitted = build_replay_manifest(
        run_id="unit-replay-001",
        seed=SEED,
        spec=spec,
        config=config,
        scenarios=(_scenario(),),
        git_revision=None,
        entrypoint_digest=digest,
    )
    assert omitted == ()
    assert manifest is not None
    (target / "replay-unit.json").write_bytes(encode_replay_manifest(manifest))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="environment variables that are not set"):
        replay_suite(target, "replay-unit.json")


def test_allowlist_mismatch_is_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentcheck.adapters import OpenAIAgentsAdapter
    from agentcheck.inspect import load_target
    from agentcheck.replay.bind import entrypoint_digest

    target = _copy_example(tmp_path)
    loaded, source = load_target(target)
    spec = OpenAIAgentsAdapter().inspect(loaded, source=source)
    digest = entrypoint_digest(target, "agent.py:agent")
    monkeypatch.setenv("OPENAI_API_KEY", "present-but-must-not-be-stored")
    manifest, omitted = build_replay_manifest(
        run_id="unit-replay-001",
        seed=SEED,
        spec=spec,
        config=AgentCheckConfig(environment_allowlist=("OPENAI_API_KEY",)),
        scenarios=(_scenario(),),
        git_revision=None,
        entrypoint_digest=digest,
    )
    assert omitted == ()
    assert manifest is not None
    (target / "replay-unit.json").write_bytes(encode_replay_manifest(manifest))
    with pytest.raises(ConfigurationError, match="environment_allowlist"):
        replay_suite(target, "replay-unit.json")


def test_entrypoint_digest_mismatch_is_configuration_error(tmp_path: Path) -> None:
    from agentcheck.adapters import OpenAIAgentsAdapter
    from agentcheck.inspect import load_target

    target = _copy_example(tmp_path)
    loaded, source = load_target(target)
    spec = OpenAIAgentsAdapter().inspect(loaded, source=source)
    manifest, omitted = build_replay_manifest(
        run_id="unit-replay-001",
        seed=SEED,
        spec=spec,
        config=AgentCheckConfig(),
        scenarios=(_scenario(),),
        git_revision=None,
        entrypoint_digest="sha256:" + "d" * 64,
    )
    assert omitted == ()
    assert manifest is not None
    (target / "replay-unit.json").write_bytes(encode_replay_manifest(manifest))
    with pytest.raises(ConfigurationError, match="entrypoint source digest"):
        replay_suite(target, "replay-unit.json")


def test_dirty_git_worktree_is_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application
    from agentcheck.adapters import OpenAIAgentsAdapter
    from agentcheck.inspect import load_target
    from agentcheck.replay import bind as replay_bind
    from agentcheck.replay.bind import entrypoint_digest

    target = _copy_example(tmp_path)
    loaded, source = load_target(target)
    spec = OpenAIAgentsAdapter().inspect(loaded, source=source)
    digest = entrypoint_digest(target, "agent.py:agent")
    revision = "a" * 40
    manifest, omitted = build_replay_manifest(
        run_id="unit-replay-001",
        seed=SEED,
        spec=spec,
        config=AgentCheckConfig(),
        scenarios=(_scenario(),),
        git_revision=revision,
        entrypoint_digest=digest,
    )
    assert omitted == ()
    assert manifest is not None
    (target / "replay-unit.json").write_bytes(encode_replay_manifest(manifest))
    monkeypatch.setattr(replay_bind, "git_revision", lambda _root: revision)
    monkeypatch.setattr(replay_bind, "git_worktree_is_dirty", lambda _root: True)
    monkeypatch.setattr(application, "_git_revision", lambda _root: revision)
    with pytest.raises(ConfigurationError, match="dirty"):
        replay_suite(target, "replay-unit.json")


def test_execute_suite_emits_replay_manifest_and_replay_reproduces(
    tmp_path: Path,
) -> None:
    target = _copy_example(tmp_path)
    execution = execute_suite(target, run_id="replay-source", persist_store=False)
    assert execution.replay_manifest_path is not None
    assert execution.replay_manifest_path.parent.name == "replay"
    assert execution.replay_manifest_path.parent != execution.artifact_directory
    if os.name == "posix":
        assert stat.S_IMODE(execution.replay_manifest_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(execution.replay_manifest_path.parent.stat().st_mode) == 0o700

    loaded = load_replay_manifest_path(execution.replay_manifest_path)
    assert loaded.schema_version == REPLAY_MANIFEST_CONTRACT_VERSION
    assert loaded.source_binding.file_set is not None
    assert loaded.source_binding.file_set.schema_version == "agentcheck.source_file_set.v1"
    assert "agent.py" in {item.path for item in loaded.source_binding.file_set.files}
    assert loaded.created_from_run_id == "replay-source"
    assert loaded.environment_requirements.names == ()
    assert loaded.seed == execution.seed
    encoded = execution.replay_manifest_path.read_text(encoding="utf-8")
    assert "sk-" not in encoded
    assert "on_invoke_tool" not in encoded
    assert tuple(case.fingerprint for case in loaded.cases) == tuple(
        case.fingerprint for case in execution.scenarios
    )
    assert [case.initial_world_state for case in loaded.cases] == [
        case.initial_world_state for case in execution.scenarios
    ]
    assert [case.tool_fixtures for case in loaded.cases] == [
        case.tool_fixtures for case in execution.scenarios
    ]
    assert [case.injected_faults for case in loaded.cases] == [
        case.injected_faults for case in execution.scenarios
    ]
    assert [case.resource_budgets for case in loaded.cases] == [
        case.resource_budgets for case in execution.scenarios
    ]

    with pytest.raises(ConfigurationError, match="disclosure suite.json"):
        replay_suite(target, ".agentcheck/runs/replay-source/suite.json")

    replayed = replay_suite(
        target,
        ".agentcheck/replay/replay-source.json",
        run_id="replay-again",
        persist_store=False,
    )
    assert replayed.run_id == "replay-again"
    assert replayed.artifact_directory != execution.artifact_directory
    assert [item.verdict for item in replayed.evaluations] == [
        item.verdict for item in execution.evaluations
    ]
    assert [item.scenario_id for item in replayed.evaluations] == [
        item.scenario_id for item in execution.evaluations
    ]
    assert Verdict.FAIL in {item.verdict for item in replayed.evaluations}
    assert all(
        left.fingerprint == right.fingerprint
        for left, right in zip(replayed.scenarios, execution.scenarios, strict=True)
    )


def test_replay_cli_never_calls_original_handlers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _copy_example(tmp_path)
    source = (target / "agent.py").read_text(encoding="utf-8")
    tripwire = "    ORIGINAL_TOOL_CALLS.append((tool_name, arguments))\n"
    probe = (
        "    with open(__file__ + \".agentcheck-original-tool-invoked\", "
        '"a", encoding="utf-8") as _probe:\n'
        "        _probe.write(f\"{tool_name}\\n\")\n"
    )
    assert tripwire in source
    (target / "agent.py").write_text(
        source.replace(tripwire, probe + tripwire, 1), encoding="utf-8"
    )
    assert main(["test", str(target), "--run-id", "cli-replay-src", "--no-store"]) == 1
    capsys.readouterr()
    probe_path = target / "agent.py.agentcheck-original-tool-invoked"
    assert not probe_path.exists()

    relative = ".agentcheck/replay/cli-replay-src.json"
    assert (target / relative).is_file()
    assert (
        main(
            [
                "replay",
                str(target),
                "--manifest",
                relative,
                "--run-id",
                "cli-replay",
                "--no-store",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "Replaying stored scenarios" in captured.out
    assert "FAIL" in captured.out
    assert not probe_path.exists()
    assert (target / ".agentcheck" / "runs" / "cli-replay" / "report.html").is_file()
    run_files = {
        path.name
        for path in (target / ".agentcheck" / "runs" / "cli-replay-src").iterdir()
    }
    assert "suite.json" in run_files
    assert all(not name.endswith(".json") or name != "replay-manifest.json" for name in run_files)


def test_replay_help_states_it_is_not_a_sandbox(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["replay", "--help"])
    assert excinfo.value.code == 0
    text = " ".join(capsys.readouterr().out.split()).lower()
    assert "not a sandbox" in text
    assert "frozen" in text
    assert "suite.json" in text
