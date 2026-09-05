"""Offline negative controls for the release artifact gate; installs are stubbed."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from scripts import check_release_artifact as gate
from scripts import check_workflow_safety as workflow_checker
from scripts.check_workflow_safety import release_artifact_gate_failures


VERSION = "0.5.4"
SOURCE = "108cde008b4e1ae7e329ab22efaaf7a6eb1d321f"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def distributions(tmp_path):
    dist = tmp_path / "dist with spaces"
    dist.mkdir()
    wheel = dist / f"agentcheck_ai-{VERSION}-py3-none-any.whl"
    wheel.write_bytes(b"stub wheel bytes; not an installed-artifact execution")
    (dist / f"agentcheck_ai-{VERSION}.tar.gz").write_bytes(b"stub sdist")
    return dist, wheel, hashlib.sha256(wheel.read_bytes()).hexdigest()


def download_info(wheel, digest):
    return {"url": wheel.as_uri(), "archive_info": {"hashes": {"sha256": digest}}}


def install_receipt(wheel, digest):
    return {"version": "1", "install": [{
        "metadata": {"name": "agentcheck-ai", "version": VERSION},
        "is_direct": True, "requested": True, "download_info": download_info(wheel, digest),
    }]}


@pytest.fixture
def fake_runner(monkeypatch, distributions):
    _, wheel, digest = distributions
    calls = []

    def execute(command, scratch):
        calls.append((command, scratch))
        if "install" in command:
            receipt = Path(command[command.index("--report") + 1])
            receipt.write_text(json.dumps(install_receipt(wheel, digest)))
        if "--probe" in command:
            extra = command[command.index("--extra") + 1]
            return json.dumps(gate.probe_receipt(digest, VERSION, extra, list(gate.SEMANTIC_CASES)))
        if command[-1] == "--version":
            return f"agentcheck {VERSION}"
        return ""

    monkeypatch.setattr(gate, "run", execute)
    return execute, calls


def test_qualifies_the_same_two_files_in_three_fresh_external_environments(distributions, fake_runner):
    dist, wheel, digest = distributions
    before = gate.artifact_hashes(dist, VERSION)
    result = gate.qualify(dist, VERSION, SOURCE)
    assert result["qualification"] == "PASS"
    assert result["artifacts"] == before == gate.artifact_hashes(dist, VERSION)
    assert result["source_sha"] == SOURCE
    assert result["environments"] == ["base", "openai-agents", "pydantic-ai"]
    _, calls = fake_runner
    installs = [command for command, _ in calls if "install" in command]
    assert len(installs) == 3
    for extra, command in zip(gate.EXTRAS, installs, strict=True):
        suffix = f"[{extra}]" if extra else ""
        assert command[-1] == f"agentcheck-ai{suffix} @ {wheel.as_uri()}#sha256={digest}"
        assert "-I" in command and "--no-cache-dir" in command
        assert "--only-binary=:all:" in command
        assert Path(command[command.index("--report") + 1]).parent != dist
    assert len({command[0] for command in installs}) == 3
    assert all(not scratch.is_relative_to(ROOT) for _, scratch in calls)
    assert all("build" not in command for command, _ in calls)


@pytest.mark.parametrize("mutation", ["missing", "extra", "directory", "symlink"])
def test_requires_exact_regular_artifact_set(distributions, mutation):
    dist, wheel, _ = distributions
    if mutation == "missing":
        wheel.unlink()
    elif mutation == "extra":
        (dist / "receipt.json").write_text("{}")
    elif mutation == "directory":
        wheel.unlink()
        wheel.mkdir()
    else:
        target = dist.parent / "other.whl"
        target.write_bytes(wheel.read_bytes())
        wheel.unlink()
        wheel.symlink_to(target)
    with pytest.raises(ValueError):
        gate.artifact_hashes(dist, VERSION)


@pytest.mark.parametrize("phase", ["venv", "install-base", "install-openai", "install-pydantic",
                                   "check", "probe", "--version", "--help"])
def test_child_failure_is_not_qualified(distributions, fake_runner, monkeypatch, phase):
    execute, _ = fake_runner
    triggered = []

    def fail(command, scratch):
        selected = phase in command or (phase == "probe" and "--probe" in command)
        if phase.startswith("install-"):
            name = {"install-base": "base", "install-openai": "openai-agents",
                    "install-pydantic": "pydantic-ai"}[phase]
            selected = "install" in command and name in Path(command[0]).parts
        if selected:
            triggered.append(True)
            raise subprocess.CalledProcessError(9, command, "failure", "diagnostic")
        return execute(command, scratch)

    monkeypatch.setattr(gate, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        gate.qualify(distributions[0], VERSION, SOURCE)
    assert triggered


@pytest.mark.parametrize("proof", ["", "{}", '{"qualification":"PASS"}',
                                   '{"semantic_cases":[]}'])
def test_noop_or_incomplete_child_receipt_cannot_pass(distributions, fake_runner, monkeypatch, proof):
    execute, _ = fake_runner
    monkeypatch.setattr(gate, "run", lambda command, scratch:
                        proof if "--probe" in command else execute(command, scratch))
    with pytest.raises(ValueError):
        gate.qualify(distributions[0], VERSION, SOURCE)


@pytest.mark.parametrize("name", ["wheel", "sdist"])
def test_modified_distribution_fails_before_qualification_returns(
    distributions, fake_runner, monkeypatch, name,
):
    dist, wheel, _ = distributions
    execute, _ = fake_runner

    def swap(command, scratch):
        if "--probe" in command:
            path = wheel if name == "wheel" else dist / f"agentcheck_ai-{VERSION}.tar.gz"
            path.write_bytes(b"different same-version bytes")
        return execute(command, scratch)

    monkeypatch.setattr(gate, "run", swap)
    with pytest.raises(ValueError, match="artifacts changed"):
        gate.qualify(dist, VERSION, SOURCE)


@pytest.mark.parametrize("mutation", ["hash", "url", "version", "not-direct", "duplicate", "missing"])
def test_installer_receipt_binds_exact_wheel_not_version_only(distributions, tmp_path, mutation):
    _, wheel, digest = distributions
    receipt = install_receipt(wheel, digest)
    item = receipt["install"][0]
    if mutation == "hash":
        item["download_info"]["archive_info"]["hashes"]["sha256"] = "b" * 64
    elif mutation == "url":
        item["download_info"]["url"] = "https://example.invalid/same-version.whl"
    elif mutation == "version":
        item["metadata"]["version"] = "0.0.0"
    elif mutation == "not-direct":
        item["is_direct"] = False
    elif mutation == "duplicate":
        receipt["install"].append(deepcopy(item))
    else:
        receipt["install"] = []
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        gate.check_install_report(path, wheel, digest, VERSION)


def test_subprocess_is_isolated_bounded_and_does_not_inherit_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-never-forward")
    monkeypatch.setenv("PYTHONPATH", "/synthetic-shadow")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://example.invalid")
    observed = {}

    def invoke(command, **kwargs):
        observed.update(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(gate.subprocess, "run", invoke)
    with pytest.raises(subprocess.TimeoutExpired):
        gate.run(["unused"], tmp_path)
    assert observed["check"] is True
    assert observed["cwd"] == tmp_path
    assert observed["timeout"] == 180
    assert observed["env"] == gate.clean_environment(tmp_path)
    assert not {"OPENAI_API_KEY", "PYTHONPATH", "PIP_EXTRA_INDEX_URL"} & observed["env"].keys()
    assert observed["env"]["PIP_CONFIG_FILE"] == gate.os.devnull
    assert observed["env"]["NETRC"] == gate.os.devnull
    assert observed["env"]["PIP_KEYRING_PROVIDER"] == "disabled"


@pytest.mark.parametrize("swallow", [False, True])
def test_bootstrap_network_attempt_cannot_pass_qualification(
    distributions, monkeypatch, swallow,
):
    import sys

    dist, wheel, digest = distributions
    real_run = gate.run
    # Synthetic audit event only: no socket, DNS lookup, or real endpoint is used.
    code = """
import runpy, sys
helper = runpy.run_path(sys.argv[1])
check = helper['deny_probe_network']()
try:
    sys.audit('socket.connect', None, ('synthetic.invalid', 0))
except ValueError:
    if sys.argv[2] == 'False':
        raise
check()
"""

    def execute(command, scratch):
        if "install" in command:
            path = Path(command[command.index("--report") + 1])
            path.write_text(json.dumps(install_receipt(wheel, digest)))
        if "--probe" in command:
            return real_run([sys.executable, "-I", "-c", code, str(gate.SCRIPT), str(swallow)], scratch)
        return ""

    monkeypatch.setattr(gate, "run", execute)
    with pytest.raises(subprocess.CalledProcessError, match="non-zero") as error:
        gate.qualify(dist, VERSION, SOURCE)
    assert "network attempt" in error.value.stderr


@pytest.fixture
def identity(monkeypatch, tmp_path, distributions):
    _, wheel, digest = distributions
    environment = tmp_path / "environment"
    site = environment / "lib/site-packages"
    state = SimpleNamespace(
        origin=site / "agentcheck/__init__.py", metadata=site / "dist-info/METADATA",
        package=site / "agentcheck/__init__.py", version=VERSION,
        direct=json.dumps(download_info(wheel, digest)),
    )
    flags = SimpleNamespace(isolated=1)
    monkeypatch.setattr(gate, "sys", SimpleNamespace(flags=flags, prefix=str(environment)))
    monkeypatch.setattr(gate.sysconfig, "get_path", lambda key: str(site))
    monkeypatch.setattr(gate.importlib.util, "find_spec", lambda name:
                        SimpleNamespace(origin=str(state.origin)))
    dist = SimpleNamespace(
        version=VERSION, files=[Path("dist-info/METADATA")],
        locate_file=lambda p: state.metadata if Path(p).name == "METADATA" else state.package,
        read_text=lambda name: state.direct,
    )
    monkeypatch.setattr(gate.metadata, "distribution", lambda name: dist)
    import sys

    package = SimpleNamespace(__version__=VERSION)
    monkeypatch.setitem(sys.modules, "agentcheck", package)
    return wheel, digest, environment, state, dist, package


def test_identity_accepts_only_bound_installed_module_and_distribution(identity):
    wheel, digest, environment, *_ = identity
    gate.check_identity(wheel, digest, VERSION, environment)


@pytest.mark.parametrize("mutation", ["shadow", "distribution", "metadata", "direct-missing",
                                   "direct-hash", "package-version", "dist-version", "not-isolated",
                                   "wrong-environment"])
def test_installed_identity_rejects_checkout_and_same_version_substitution(identity, mutation):
    wheel, digest, environment, state, dist, package = identity
    if mutation == "shadow":
        state.origin = ROOT / "agentcheck/__init__.py"
    elif mutation == "distribution":
        state.package = ROOT / "agentcheck/__init__.py"
    elif mutation == "metadata":
        state.metadata = ROOT / "agentcheck_ai.egg-info/METADATA"
    elif mutation == "direct-missing":
        state.direct = None
    elif mutation == "direct-hash":
        state.direct = json.dumps(download_info(wheel, "c" * 64))
    elif mutation == "package-version":
        package.__version__ = "0.0.0"
    elif mutation == "dist-version":
        dist.version = "0.0.0"
    elif mutation == "not-isolated":
        gate.sys.flags.isolated = 0
    else:
        gate.sys.prefix = str(ROOT)
    with pytest.raises(ValueError):
        gate.check_identity(wheel, digest, VERSION, environment)


@pytest.mark.parametrize("extra", gate.EXTRAS)
@pytest.mark.parametrize("mutation", ["none", "leak", "missing", "unsupported", "unactionable"])
def test_framework_checks_fail_closed_without_provider_calls(monkeypatch, extra, mutation):
    import sys
    from agentcheck.adapters import AdapterDependencyError

    present = {name: name == extra for name in ("openai-agents", "pydantic-ai")}
    if mutation == "leak":
        present[next(name for name in present if name != extra)] = True
    elif mutation == "missing":
        if not extra:
            present["openai-agents"] = True
        else:
            present[extra] = False
    for name, module in (("openai-agents", "openai_agents"), ("pydantic-ai", "pydantic_ai")):
        def need_sdk(name=name):
            if not present[name]:
                message = "missing" if mutation == "unactionable" else "pip install the extra"
                raise AdapterDependencyError(message)

        adapter = SimpleNamespace(
            _require_sdk=need_sdk, _sdk_version=lambda: "supported-version",
            _supported_sdk_version=lambda version: mutation != "unsupported",
        )
        monkeypatch.setitem(sys.modules, f"agentcheck.adapters.{module}", adapter)
    modules = {"agents": "openai-agents", "pydantic_ai": "pydantic-ai"}
    monkeypatch.setattr(gate.importlib.util, "find_spec", lambda module:
                        object() if present[modules[module]] else None)
    if mutation == "none" or (mutation == "unsupported" and not extra):
        gate.check_frameworks(extra)
    else:
        with pytest.raises(ValueError):
            gate.check_frameworks(extra)


def test_main_failure_does_not_emit_passing_receipt(monkeypatch, capsys):
    monkeypatch.setattr(gate.sys, "argv", ["helper", "--version", VERSION, "--source-sha", SOURCE])

    def fail(*args):
        raise subprocess.CalledProcessError(7, ["installed-check"], "child output", "failed check")

    monkeypatch.setattr(gate, "qualify", fail)
    assert gate.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "failed check" in captured.err


def test_checks_are_not_disabled_by_python_optimization():
    import sys

    result = subprocess.run([
        sys.executable, "-O", "-c",
        "from scripts.check_release_artifact import require; require(False, 'must fail')",
    ], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "must fail" in result.stderr


@pytest.mark.parametrize("context", ["withheld", "absent", "established"])
@pytest.mark.parametrize("call", [False, True])
def test_confirmation_fixture_has_authored_context_and_complete_public_records(context, call):
    scenario, record = gate.confirmation_fixture(context, call)
    assert scenario.allowed_tool_behavior[0].min_calls == 0
    assert scenario.allowed_tool_behavior[0].confirmation_required_before_call == (context == "withheld")
    assert bool(scenario.followup_turns) == (context == "established")
    assert bool(record.tool_attempts) is call
    assert record.events[-1].payload["text"] == record.final_output
    assert record.termination.value == "completed"
    for turn, event in zip((*scenario.conversation_turns, *scenario.followup_turns), record.events):
        assert event.event_type.value == "user_turn"
        assert event.payload == {"turn_id": turn.turn_id, "text": turn.content}
        assert event.metadata == {**turn.metadata, "scenario_input": True}
    if call:
        attempt = record.tool_attempts[0]
        event = record.events[attempt.sequence]
        assert event.event_id == attempt.event_id
        assert event.payload["arguments"] == attempt.arguments
        assert event.payload["tool_name"] == attempt.tool_name


@pytest.mark.parametrize("mutation", ["none", "false-pass", "false-fail", "missing-criterion",
                                      "unmeasured", "vacuous-evidence"])
def test_semantic_smoke_enforces_fixed_verdict_and_evidence_expectations(monkeypatch, mutation):
    import agentcheck.evaluate

    authored = gate.confirmation_fixture
    actual_evaluate = agentcheck.evaluate.evaluate_run
    state = {}

    def fixture(context, call, **kwargs):
        state.update(context=context, call=call)
        result = authored(context, call, **kwargs)
        if kwargs.get("prose"):
            assert result[0].conversation_turns[0].metadata == {}
            assert result[0].followup_turns == ()
        return result

    def evaluate(scenario, record):
        if record.run_id != "release-smoke":
            return actual_evaluate(scenario, record)
        context, call = state["context"], state["call"]
        expected = "INCONCLUSIVE" if context == "absent" else (
            "FAIL" if context == "withheld" and call else "PASS"
        )
        if mutation == "false-pass" and context == "absent":
            expected = "PASS"
        if mutation == "false-fail" and context == "withheld" and not call:
            expected = "FAIL"
        result = SimpleNamespace(value=expected)
        data = {"confirmation_context": context, "confirmation_exercised": call}
        if mutation == "unmeasured":
            data["confirmation_exercised"] = None
        if mutation == "vacuous-evidence":
            data["confirmed_before_every_call"] = True
        assertion = SimpleNamespace(assertion_id="confirm", result=result, supporting_evidence_ids=("e",))
        return SimpleNamespace(
            verdict=result, assertions=[] if mutation == "missing-criterion" else [assertion],
            evidence=[SimpleNamespace(evidence_id="e", data=data)],
        )

    monkeypatch.setattr(gate, "confirmation_fixture", fixture)
    monkeypatch.setattr(agentcheck.evaluate, "evaluate_run", evaluate)
    if mutation == "none":
        assert gate.semantic_smoke() == list(gate.SEMANTIC_CASES)
    else:
        with pytest.raises(ValueError):
            gate.semantic_smoke()


@pytest.mark.parametrize("omitted", [None, 0, 1])
def test_retry_fixture_is_coherent_and_omits_only_the_named_outcome(omitted):
    from agentcheck.evaluate.confirmation import observed_completion, tool_evidence_is_consistent

    scenario, record = gate.ambiguous_retry_fixture(omitted)
    assert tool_evidence_is_consistent(scenario, record)
    assert observed_completion(scenario, record)
    assert len(record.tool_attempts) == 2
    assert [o.attempt_id for o in record.tool_outcomes] == [f"a{i}" for i in (0, 1) if i != omitted]
    assert [o.status.value for o in record.tool_outcomes] == [s for i, s in enumerate(("timeout", "success")) if i != omitted]
    assert {e.event_id for e in record.events if e.event_type.value == "tool_result"} == {o.event_id for o in record.tool_outcomes}
    if omitted != 0:
        assert record.tool_outcomes[0].error.code == "ambiguous_timeout"


@pytest.mark.parametrize("mutation", ["none", "false-pass", "missing-evidence",
                                      "lost-known-fail", "missing-criterion", "missing-retry-id"])
def test_retry_release_smoke_rejects_wrong_verdicts_and_missing_evidence(monkeypatch, mutation):
    import agentcheck.evaluate
    from agentcheck.domain import Verdict

    actual = agentcheck.evaluate.evaluate_run

    def evaluate(scenario, record):
        result = actual(scenario, record)
        if scenario.scenario_id != "release-ambiguous-retry" or mutation == "none":
            return result
        assertion = next(a for a in result.assertions if a.assertion_id == "retry")
        missing_first = [o.attempt_id for o in record.tool_outcomes] == ["a1"]
        missing_second = [o.attempt_id for o in record.tool_outcomes] == ["a0"]
        verdict = result.verdict
        if missing_first and mutation == "false-pass":
            verdict = Verdict.PASS
            assertion = assertion.model_copy(update={"result": verdict})
        if missing_first and mutation == "missing-evidence":
            assertion = assertion.model_copy(update={"missing_evidence": ()})
        if missing_second and mutation == "lost-known-fail":
            verdict = Verdict.INCONCLUSIVE
            assertion = assertion.model_copy(update={"result": verdict})
        assertions = tuple(assertion if a.assertion_id == "retry" else a for a in result.assertions)
        if mutation == "missing-criterion":
            assertions = tuple(a for a in assertions if a.assertion_id != "retry")
        evidence = result.evidence
        if mutation == "missing-retry-id" and not missing_first:
            evidence = tuple(e.model_copy(update={"data": {**e.data, "retry_attempt_ids": []}})
                             if e.evidence_id in assertion.supporting_evidence_ids else e for e in evidence)
        return result.model_copy(update={"verdict": verdict, "assertions": assertions, "evidence": evidence})

    monkeypatch.setattr(agentcheck.evaluate, "evaluate_run", evaluate)
    if mutation == "none":
        assert gate.semantic_smoke() == [
            "withheld-no-call", "withheld-call", "absent-no-call", "absent-call",
            "prose-not-consent", "established-consent", "established-no-call",
            "ambiguous-retry-complete", "ambiguous-retry-missing-origin",
            "ambiguous-retry-known-violation",
            "authored-sample-mismatch", "generated-exact-mismatch", "authored-schema-failure",
            "authored-retry-missing-origin", "authored-retry-known-violation",
        ]
    else:
        with pytest.raises(ValueError):
            gate.semantic_smoke()


@pytest.mark.parametrize("kind", [
    "authored-sample-mismatch", "generated-exact-mismatch", "authored-schema-failure",
    "authored-retry-missing-origin", "authored-retry-known-violation",
])
def test_argument_authority_release_fixtures_bind_generated_contract_and_record(kind):
    from agentcheck.evaluate.confirmation import observed_completion, tool_evidence_is_consistent

    scenario, record = gate.argument_authority_fixture(kind)
    assert tool_evidence_is_consistent(scenario, record)
    assert observed_completion(scenario, record)
    sample_oracles = [o for o in scenario.oracle_provenance if o.oracle_id.endswith(":argument-sample")]
    assert len(sample_oracles) == (0 if kind == "generated-exact-mismatch" else 1)
    if sample_oracles:
        assert not sample_oracles[0].supports_hard_failure and sample_oracles[0].confidence == 0.0
        assert scenario.allowed_tool_behavior[0].oracle_ids == (sample_oracles[0].oracle_id,)
    assert record.tool_attempts[0].arguments == {
        "record_id": 3 if kind == "authored-schema-failure" else "record-1", "note": "", "labels": None,
    }
    if "retry" in kind:
        expected_ids = ["a1"] if kind == "authored-retry-missing-origin" else ["a0"]
        assert [o.attempt_id for o in record.tool_outcomes] == expected_ids
        assert len(record.tool_attempts) == 2


@pytest.mark.parametrize("mutation", [
    "none", "hard-sample", "false-pass", "missing-criterion", "missing-evidence",
    "missing-values", "missing-schema", "missing-retry", "missing-retry-data",
])
def test_argument_authority_release_smoke_rejects_wrong_verdicts_and_missing_evidence(monkeypatch, mutation):
    import agentcheck.evaluate
    from agentcheck.domain import Verdict

    actual = agentcheck.evaluate.evaluate_run

    def evaluate(scenario, record):
        result = actual(scenario, record)
        if record.run_id != "release-authority" or mutation == "none":
            return result
        criterion = "tool_contract:cancel_record:unexpected_arguments"
        assertions, evidence, verdict = list(result.assertions), list(result.evidence), result.verdict
        for index, assertion in enumerate(assertions):
            if assertion.assertion_id == criterion:
                if mutation in {"hard-sample", "false-pass"}:
                    verdict = Verdict.FAIL if mutation == "hard-sample" else Verdict.PASS
                    assertions[index] = assertion.model_copy(update={"result": verdict})
                if mutation == "missing-evidence":
                    assertions[index] = assertion.model_copy(update={"missing_evidence": ()})
                if mutation == "missing-values":
                    evidence = [e.model_copy(update={"data": {}})
                                if e.evidence_id in assertion.supporting_evidence_ids else e for e in evidence]
            if mutation == "missing-retry-data" and assertion.assertion_id.endswith(":no_retry"):
                evidence = [e.model_copy(update={"data": {}})
                            if e.evidence_id in assertion.supporting_evidence_ids else e for e in evidence]
        if mutation == "missing-criterion":
            assertions = [a for a in assertions if a.assertion_id != criterion]
        if mutation == "missing-schema":
            assertions = [a for a in assertions if not a.assertion_id.startswith("schema:")]
        if mutation == "missing-retry":
            assertions = [a for a in assertions if not a.assertion_id.endswith(":no_retry")]
        return result.model_copy(update={"verdict": verdict, "assertions": tuple(assertions), "evidence": tuple(evidence)})

    monkeypatch.setattr(agentcheck.evaluate, "evaluate_run", evaluate)
    if mutation == "none":
        assert gate.semantic_smoke() == list(gate.SEMANTIC_CASES)
    else:
        with pytest.raises(ValueError):
            gate.semantic_smoke()


def release_workflow():
    return yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())


def test_release_workflow_has_executable_same_artifact_gate():
    assert release_artifact_gate_failures(release_workflow()) == []


@pytest.mark.parametrize("scope", ["workflow", "build"])
@pytest.mark.parametrize("setting", ["shell", "working-directory"])
def test_release_guard_rejects_inherited_execution_defaults(scope, setting):
    workflow = release_workflow()
    target = workflow if scope == "workflow" else workflow["jobs"]["build"]
    value = "bash -c 'bash {0}; exit 0'" if setting == "shell" else "other-checkout"
    target["defaults"] = {"run": {setting: value}}
    assert release_artifact_gate_failures(workflow)


@pytest.mark.parametrize("mutation", ["extra-producer", "overwrite"])
def test_release_guard_requires_sole_non_overwriting_producer(mutation):
    workflow = release_workflow()
    upload = workflow["jobs"]["build"]["steps"][-1]
    if mutation == "extra-producer":
        upload = deepcopy(upload)
        upload["with"]["path"] = "different-artifacts/"
        workflow["jobs"]["replacement"] = {
            "runs-on": "ubuntu-latest", "needs": "build",
            "permissions": {"contents": "read"}, "steps": [upload],
        }
    upload["with"]["overwrite"] = True
    assert release_artifact_gate_failures(workflow)


@pytest.mark.parametrize("mutation", ["workflow-defaults", "build-defaults", "replacement-producer"])
def test_full_workflow_checker_rejects_reviewed_bypasses(monkeypatch, mutation):
    workflow = release_workflow()
    if mutation == "replacement-producer":
        upload = deepcopy(workflow["jobs"]["build"]["steps"][-1])
        upload["with"].update(path="different-artifacts/", overwrite=True)
        workflow["jobs"]["replacement"] = {
            "runs-on": "ubuntu-latest", "permissions": {"contents": "read"},
            "needs": "build", "steps": [upload],
        }
    else:
        target = workflow if mutation == "workflow-defaults" else workflow["jobs"]["build"]
        target["defaults"] = {"run": {"shell": "bash -c 'bash {0}; exit 0'"}}
    directory = ROOT / ".github/workflows"
    text = (directory / "release.yml").read_text()
    original = yaml.safe_load
    monkeypatch.setattr(workflow_checker, "WORKFLOW_DIRECTORY", directory)
    monkeypatch.setattr(workflow_checker.yaml, "safe_load", lambda value:
                        deepcopy(workflow) if value == text else original(value))
    assert workflow_checker.main() == 1


@pytest.mark.parametrize("mutation", ["remove", "noop", "ignore", "conditional", "after-upload",
    "before-audit", "upload-always", "publish-always", "missing-needs", "rebuild", "different-dist",
    "different-download", "wrong-sha", "wrong-command", "ignore-job", "new-budget"])
def test_release_workflow_gate_mutations_fail(mutation):
    workflow = release_workflow()
    build, publish = workflow["jobs"]["build"], workflow["jobs"]["publish"]
    steps = build["steps"]
    index = next(i for i, step in enumerate(steps) if step.get("id") == "qualify_artifacts")
    qualifier, upload = steps[index], steps[index + 1]
    if mutation == "remove":
        steps.pop(index)
    elif mutation == "noop":
        qualifier["run"] = 'echo "qualification PASS"'
    elif mutation == "ignore":
        qualifier["continue-on-error"] = True
    elif mutation == "conditional":
        qualifier["if"] = "false"
    elif mutation == "after-upload":
        steps[index], steps[index + 1] = steps[index + 1], steps[index]
    elif mutation == "before-audit":
        steps[index], steps[index - 1] = steps[index - 1], steps[index]
    elif mutation == "upload-always":
        upload["if"] = "always()"
    elif mutation == "publish-always":
        publish["if"] = "always()"
    elif mutation == "missing-needs":
        del publish["needs"]
    elif mutation == "rebuild":
        publish["steps"].insert(1, {"run": "python -m build"})
    elif mutation == "different-dist":
        upload["with"]["path"] = "other-dist/"
    elif mutation == "different-download":
        publish["steps"][0]["with"]["name"] = "untested-artifacts"
    elif mutation == "wrong-sha":
        qualifier["env"]["RELEASE_SHA"] = "a" * 40
    elif mutation == "wrong-command":
        qualifier["run"] += "\ntrue"
    elif mutation == "ignore-job":
        build["continue-on-error"] = True
    else:
        build["timeout-minutes"] = 30
    assert release_artifact_gate_failures(workflow), mutation
