from __future__ import annotations

import io
import json
import stat
import time
from pathlib import Path

import pytest

from agentcheck.config import AgentCheckConfig, load_config
from agentcheck.domain import (
    CanonicalRun,
    FaultType,
    InjectedFault,
    RunTermination,
    Scenario,
    ToolOutcomeStatus,
    utc_now,
)
from agentcheck.generate import build_account_support_suite
from agentcheck.initialize import write_initial_config
from agentcheck.runner import inspect_in_subprocess, run_scenario_in_subprocess
from agentcheck.runner.orchestrator import (
    WORKER_REQUEST_VERSION,
    WORKER_RESPONSE_VERSION,
    _write_private_json,
)
from agentcheck.runner import orchestrator
from agentcheck.runner.worker import execute_request


EXAMPLE = Path(__file__).parents[2] / "examples" / "evaluation" / "account_agent"


def _write_target(root: Path, source: str) -> AgentCheckConfig:
    (root / "agent.py").write_text(source, encoding="utf-8")
    return AgentCheckConfig()


def test_inspection_runs_in_a_fresh_process_each_time() -> None:
    root, config = load_config(EXAMPLE)

    first = inspect_in_subprocess(root, config)
    second = inspect_in_subprocess(root, config)

    assert first.ok and second.ok
    assert first.require_value().identity.name.value == "Account Support Agent"
    assert {item.value.name for item in first.require_value().tools.items} == {
        "lookup_account",
        "update_email",
        "cancel_subscription",
        "delete_account",
    }
    assert first.worker_pid is not None
    assert second.worker_pid is not None
    assert first.worker_pid != second.worker_pid
    assert first.preflight_issues == ()
    assert second.preflight_issues == ()


def test_scenario_run_returns_canonical_run_from_intercepted_tools() -> None:
    root, config = load_config(EXAMPLE)
    scenario = next(
        item
        for item in build_account_support_suite(seed=config.seed)
        if item.scenario_id == "happy_lookup"
    )

    result = run_scenario_in_subprocess(root, config, scenario, "isolated-happy")

    run = result.require_value()
    assert run.run_id == "isolated-happy"
    assert run.scenario_id == scenario.scenario_id
    assert run.termination == RunTermination.COMPLETED
    assert [attempt.tool_name for attempt in run.tool_attempts] == ["lookup_account"]
    assert run.initial_world_state == scenario.initial_world_state
    assert run.final_world_state == scenario.initial_world_state
    assert "alex@example.com" in (run.final_output or "")


def test_child_environment_is_allowlisted_and_diagnostics_are_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTCHECK_ALLOWED_TEST", "visible-to-worker")
    parent_canary = "".join(("sk", "-", "parent", "secret", "12345"))
    stdout_canary = "".join(("sk", "-", "stdout", "secret", "12345"))
    monkeypatch.setenv("OPENAI_API_KEY", parent_canary)
    config = _write_target(
        tmp_path,
        f"""
import os
from agents import Agent

print("api_key={stdout_canary}")
print("x" * 100_000)
if "OPENAI_API_KEY" in os.environ:
    raise RuntimeError("provider credential leaked")
agent = Agent(
    name=os.environ.get("AGENTCHECK_ALLOWED_TEST", "not-allowed"),
    instructions="Inspect only.",
    model="gpt-4.1-mini",
)
""".lstrip(),
    ).model_copy(update={"environment_allowlist": ("AGENTCHECK_ALLOWED_TEST",)})

    result = inspect_in_subprocess(tmp_path, config)

    assert result.require_value().identity.name.value == "visible-to-worker"
    assert stdout_canary not in result.stdout
    assert "[REDACTED]" in result.stdout
    assert "...[TRUNCATED]" in result.stdout
    assert len(result.stdout) < 17_000
    assert "parentsecret" not in result.stdout + result.stderr


def test_wall_clock_timeout_kills_worker_and_is_infrastructure_error(
    tmp_path: Path,
) -> None:
    config = _write_target(
        tmp_path,
        """
import time

print("authorization=Bearer timeout-secret-value", flush=True)
time.sleep(10)
agent = object()
""".lstrip(),
    )
    started = time.monotonic()

    result = inspect_in_subprocess(tmp_path, config, timeout_seconds=0.2)

    assert time.monotonic() - started < 3
    assert result.value is None
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.code == "worker_timeout"
    assert result.infrastructure_error.phase in {"worker_start", "load"}
    assert result.timed_out is True
    assert result.returncode is not None and result.returncode < 0
    assert "timeout-secret-value" not in result.stdout


def test_target_import_failure_is_not_an_agent_failure(tmp_path: Path) -> None:
    config = _write_target(
        tmp_path,
        'raise RuntimeError("api_key=sk-importsecret12345\\n'
        'Authorization: Basic dXNlcjpwYXNz\\n'
        'Cookie: session=worker-cookie-secret\\n'
        'Set-Cookie: refresh=worker-refresh-secret; Path=/")\n',
    )

    result = inspect_in_subprocess(tmp_path, config)

    assert result.ok is False
    assert result.value is None
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.code == "target_import_failed"
    assert result.infrastructure_error.phase == "load"
    assert "sk-importsecret12345" not in result.infrastructure_error.message
    assert "dXNlcjpwYXNz" not in result.infrastructure_error.message
    assert "worker-cookie-secret" not in result.infrastructure_error.message
    assert "worker-refresh-secret" not in result.infrastructure_error.message
    assert "[REDACTED]" in result.infrastructure_error.message


def test_request_and_response_files_are_private(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    _write_private_json(
        request_path,
        {
            "contract_version": WORKER_REQUEST_VERSION,
            "operation": "not-supported",
        },
    )

    returncode = execute_request(request_path, response_path)

    assert returncode == 1
    assert stat.S_IMODE(request_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(response_path.stat().st_mode) == 0o600
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "error"
    assert response["error"]["phase"] == "request"


def test_inspect_worker_response_includes_preflight_report(tmp_path: Path) -> None:
    root, config = load_config(EXAMPLE)
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    _write_private_json(
        request_path,
        {
            "contract_version": WORKER_REQUEST_VERSION,
            "operation": "inspect",
            "root": str(root),
            "config": config.model_dump(mode="json"),
        },
    )

    assert execute_request(request_path, response_path) == 0
    response = json.loads(response_path.read_text(encoding="utf-8"))
    assert response["status"] == "ok"
    assert response["preflight"] == {"framework": "openai_agents", "issues": []}
    assert "topology" not in response
    assert response["result"]["identity"]["name"]["value"] == "Account Support Agent"


def test_malformed_inspect_preflight_is_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentcheck.adapters import OpenAIAgentsAdapter
    from agentcheck.inspect import load_target

    root, config = load_config(EXAMPLE)
    target, source = load_target(EXAMPLE)
    spec = OpenAIAgentsAdapter().inspect(target, source=source)

    class FakeProcess:
        pid = 4545
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def poll(self) -> int:
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        del kwargs
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "contract_version": WORKER_RESPONSE_VERSION,
                    "status": "ok",
                    "phase": "complete",
                    "operation": "inspect",
                    "worker_pid": FakeProcess.pid,
                    "result": spec.model_dump(mode="json"),
                    "preflight": {"framework": "openai_agents", "issues": "nope"},
                }
            ),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr(orchestrator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        orchestrator, "_kill_remaining_process_group", lambda process: None
    )

    result = inspect_in_subprocess(root, config)
    assert result.ok is False
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.code == "invalid_worker_result"


def test_malformed_inspect_topology_is_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentcheck.adapters import OpenAIAgentsAdapter
    from agentcheck.inspect import load_target

    root, config = load_config(EXAMPLE)
    target, source = load_target(EXAMPLE)
    spec = OpenAIAgentsAdapter().inspect(target, source=source)

    class FakeProcess:
        pid = 4747
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def poll(self) -> int:
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        del kwargs
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "contract_version": WORKER_RESPONSE_VERSION,
                    "status": "ok",
                    "phase": "complete",
                    "operation": "inspect",
                    "worker_pid": FakeProcess.pid,
                    "result": spec.model_dump(mode="json"),
                    "preflight": {"framework": "openai_agents", "issues": []},
                    "topology": {"framework": "openai_agents", "agents": [{"name": "x"}]},
                }
            ),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr(orchestrator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        orchestrator, "_kill_remaining_process_group", lambda process: None
    )

    result = inspect_in_subprocess(root, config)
    assert result.ok is False
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.code == "invalid_worker_result"


def test_inspect_response_without_preflight_is_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentcheck.adapters import OpenAIAgentsAdapter
    from agentcheck.inspect import load_target

    root, config = load_config(EXAMPLE)
    target, source = load_target(EXAMPLE)
    spec = OpenAIAgentsAdapter().inspect(target, source=source)

    class FakeProcess:
        pid = 4646
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def poll(self) -> int:
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        del kwargs
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "contract_version": WORKER_RESPONSE_VERSION,
                    "status": "ok",
                    "phase": "complete",
                    "operation": "inspect",
                    "worker_pid": FakeProcess.pid,
                    "result": spec.model_dump(mode="json"),
                }
            ),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr(orchestrator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        orchestrator, "_kill_remaining_process_group", lambda process: None
    )

    result = inspect_in_subprocess(root, config)
    assert result.ok is False
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.code == "invalid_worker_result"


def test_invalid_worker_result_is_structured_infrastructure_error(
    tmp_path: Path,
) -> None:
    config = _write_target(tmp_path, "agent = object()\n")

    result = inspect_in_subprocess(tmp_path, config)

    assert result.value is None
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.phase == "load"
    assert result.returncode == 1


def test_parent_private_json_rejects_non_json_values(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        _write_private_json(tmp_path / "bad.json", {"not_json": object()})

    assert not (tmp_path / "bad.json").exists()


def test_worker_does_not_inherit_parent_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/definitely/untrusted/path")
    config = _write_target(
        tmp_path,
        """
import os
from agents import Agent

agent = Agent(
    name="safe" if "/definitely/untrusted/path" not in os.environ.get("PYTHONPATH", "") else "unsafe",
    instructions="Inspect only.",
    model="gpt-4.1-mini",
)
""".lstrip(),
    )

    result = inspect_in_subprocess(tmp_path, config)

    assert result.require_value().identity.name.value == "safe"


def test_target_cannot_shadow_agentcheck_worker_at_python_startup(
    tmp_path: Path,
) -> None:
    shadow = tmp_path / "agentcheck"
    shadow.mkdir()
    marker = tmp_path / "shadow-imported"
    (shadow / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('unsafe')\n",
        encoding="utf-8",
    )
    config = _write_target(
        tmp_path,
        """
from agents import Agent

agent = Agent(name="safe", instructions="Inspect only.", model="gpt-4.1-mini")
""".lstrip(),
    )

    result = inspect_in_subprocess(tmp_path, config)

    assert result.require_value().identity.name.value == "safe"
    assert not marker.exists()


def test_injected_fault_overrides_fixture_for_the_same_invocation() -> None:
    root, config = load_config(EXAMPLE)
    baseline = next(
        item
        for item in build_account_support_suite(seed=config.seed)
        if item.scenario_id == "happy_lookup"
    )
    payload = baseline.model_dump(mode="json")
    payload["tool_fixtures"][0]["invocation_index"] = 1
    payload["injected_faults"] = [
        InjectedFault(
            fault_id="forced-timeout",
            tool_name="lookup_account",
            fault_type=FaultType.TIMEOUT,
            invocation_index=1,
            message="Controlled timeout.",
        ).model_dump(mode="json")
    ]
    payload["fingerprint"] = ""
    scenario = Scenario.model_validate_json(json.dumps(payload))

    run = run_scenario_in_subprocess(
        root, config, scenario, "injected-timeout"
    ).require_value()

    assert run.termination == RunTermination.COMPLETED
    assert len(run.tool_outcomes) == 1
    assert run.tool_outcomes[0].status == ToolOutcomeStatus.TIMEOUT
    assert run.tool_outcomes[0].error is not None
    assert run.tool_outcomes[0].error.code == "timeout"


def test_valid_canonical_adapter_error_is_preserved_for_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = load_config(EXAMPLE)
    scenario = next(
        item
        for item in build_account_support_suite(seed=config.seed)
        if item.scenario_id == "happy_lookup"
    )
    now = utc_now()
    canonical_error = CanonicalRun(
        run_id="adapter-error-run",
        scenario_id=scenario.scenario_id,
        target_id="target",
        started_at=now,
        ended_at=now,
        termination=RunTermination.ADAPTER_ERROR,
        termination_reason="Controlled adapter failure.",
        initial_world_state=scenario.initial_world_state,
        final_world_state=scenario.initial_world_state,
    )

    class FakeProcess:
        pid = 4444
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def poll(self) -> int:
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        del kwargs
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "contract_version": WORKER_RESPONSE_VERSION,
                    "status": "ok",
                    "phase": "complete",
                    "operation": "run",
                    "worker_pid": FakeProcess.pid,
                    "result": canonical_error.model_dump(mode="json"),
                }
            ),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr(orchestrator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        orchestrator, "_kill_remaining_process_group", lambda process: None
    )

    result = run_scenario_in_subprocess(root, config, scenario, "adapter-error-run")

    run = result.require_value()
    assert run.termination == RunTermination.ADAPTER_ERROR
    assert run.termination_reason == "Controlled adapter failure."
    assert result.infrastructure_error is None


def test_parent_interruption_kills_and_reaps_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_target(tmp_path, "agent = object()\n")

    class FakeProcess:
        pid = 4242
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        returncode: int | None = None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is not None:
                return self.returncode
            raise KeyboardInterrupt

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    process = FakeProcess()
    killed: list[int] = []

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        del args, kwargs
        return process

    def fake_kill(value: FakeProcess) -> None:
        killed.append(value.pid)
        value.returncode = -9

    monkeypatch.setattr(orchestrator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(orchestrator, "_kill_process_group", fake_kill)
    monkeypatch.setattr(
        orchestrator, "_kill_remaining_process_group", lambda process: None
    )

    with pytest.raises(KeyboardInterrupt):
        inspect_in_subprocess(tmp_path, config)

    assert killed == [process.pid]
    assert process.returncode == -9


def test_mismatched_canonical_run_identity_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = load_config(EXAMPLE)
    scenario = next(
        item
        for item in build_account_support_suite(seed=config.seed)
        if item.scenario_id == "happy_lookup"
    )
    now = utc_now()
    forged = CanonicalRun(
        run_id="wrong-run",
        scenario_id="wrong-scenario",
        target_id="wrong-target",
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
        initial_world_state=scenario.initial_world_state,
        final_world_state=scenario.initial_world_state,
    )

    class FakeProcess:
        pid = 4343
        stdout = io.BytesIO()
        stderr = io.BytesIO()
        returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def poll(self) -> int:
            return 0

        def kill(self) -> None:
            return None

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        assert "-P" not in command
        assert command[1] == "-c"
        assert "sys.path" in command[2]
        assert kwargs["cwd"] == root
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        package_root = str(Path(orchestrator.__file__).resolve().parents[2])
        assert environment.get("PYTHONPATH") == package_root
        assert "AGENTCHECK_PACKAGE_ROOT" not in environment
        response_path = Path(command[-1])
        response_path.write_text(
            json.dumps(
                {
                    "contract_version": WORKER_RESPONSE_VERSION,
                    "status": "ok",
                    "phase": "complete",
                    "operation": "run",
                    "worker_pid": FakeProcess.pid,
                    "result": forged.model_dump(mode="json"),
                }
            ),
            encoding="utf-8",
        )
        return FakeProcess()

    monkeypatch.setattr(orchestrator.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        orchestrator, "_kill_remaining_process_group", lambda process: None
    )

    result = run_scenario_in_subprocess(
        root,
        config,
        scenario,
        "expected-run",
        expected_target_id="expected-target",
    )

    assert result.value is None
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.code == "worker_result_identity_mismatch"
    mismatches = result.infrastructure_error.details["mismatches"]
    assert isinstance(mismatches, dict)
    assert set(mismatches) == {"run_id", "scenario_id", "target_id"}


def test_no_worker_temp_directory_is_left_in_target(tmp_path: Path) -> None:
    config = _write_target(
        tmp_path,
        """
from agents import Agent

agent = Agent(name="clean", instructions="Inspect only.", model="gpt-4.1-mini")
""".lstrip(),
    )

    assert inspect_in_subprocess(tmp_path, config).ok

    assert not any(
        path.name.startswith("agentcheck-worker-") for path in tmp_path.iterdir()
    )


def test_target_cannot_shadow_agentcheck_worker_module(tmp_path: Path) -> None:
    marker = tmp_path / "shadow-worker-ran"
    shadow = tmp_path / "agentcheck" / "runner"
    shadow.mkdir(parents=True)
    (tmp_path / "agentcheck" / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "worker.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('unsafe')\n",
        encoding="utf-8",
    )
    config = _write_target(
        tmp_path,
        """
from agents import Agent

agent = Agent(name="safe-worker", instructions="Inspect only.", model="gpt-4.1-mini")
""".lstrip(),
    )

    result = inspect_in_subprocess(tmp_path, config)

    assert result.require_value().identity.name.value == "safe-worker"
    assert not marker.exists()


def test_in_process_execute_request_restores_cwd(tmp_path: Path) -> None:
    root, config = load_config(EXAMPLE)
    previous = Path.cwd()
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    _write_private_json(
        request_path,
        {
            "contract_version": WORKER_REQUEST_VERSION,
            "operation": "inspect",
            "root": str(root),
            "config": config.model_dump(mode="json"),
        },
    )

    assert execute_request(request_path, response_path) == 0
    assert Path.cwd() == previous


def test_worker_import_does_not_observe_host_cwd_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = tmp_path / "host"
    host.mkdir()
    (host / ".env").write_text(
        "AGENTCHECK_CWD_SENTINEL=leaked-from-host\n", encoding="utf-8"
    )
    monkeypatch.chdir(host)

    target = tmp_path / "target"
    target.mkdir()
    write_initial_config(target, entrypoint="agent.py:create_agent()")
    (target / "agent.py").write_text(
        """
from pathlib import Path
import json
from agents import Agent

OBSERVED = Path(__file__).with_name("observed.json")


def _observe(stage: str) -> None:
    cwd = Path.cwd()
    dotenv = Path(".env")
    existing = json.loads(OBSERVED.read_text(encoding="utf-8")) if OBSERVED.is_file() else []
    existing.append(
        {
            "stage": stage,
            "cwd": str(cwd),
            "dotenv_exists": dotenv.is_file(),
            "dotenv_text": dotenv.read_text(encoding="utf-8") if dotenv.is_file() else "",
        }
    )
    OBSERVED.write_text(json.dumps(existing), encoding="utf-8")


_observe("import")


def create_agent():
    _observe("factory")
    return Agent(
        name="cwd-probe",
        instructions="Inspect only.",
        model="gpt-4.1-mini",
    )
""".lstrip(),
        encoding="utf-8",
    )
    _, config = load_config(target)

    result = inspect_in_subprocess(target, config)

    assert result.require_value().identity.name.value == "cwd-probe"
    assert Path.cwd() == host.resolve()
    observed = json.loads((target / "observed.json").read_text(encoding="utf-8"))
    assert [item["stage"] for item in observed] == ["import", "factory"]
    for item in observed:
        assert item["cwd"] == str(target.resolve())
        assert item["dotenv_exists"] is False
        assert "leaked-from-host" not in item["dotenv_text"]
