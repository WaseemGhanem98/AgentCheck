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


def release_workflow():
    return yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())


def test_release_workflow_has_executable_same_artifact_gate():
    assert release_artifact_gate_failures(release_workflow()) == []


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
