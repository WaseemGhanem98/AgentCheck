"""Real-world target loading: interpreter selection, factories, and diagnostics."""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.cli import main
from agentcheck.config import (
    apply_python_executable,
    load_config,
    portable_entrypoint,
)
from agentcheck.errors import ConfigurationError
from agentcheck.initialize import write_initial_config
from agentcheck.inspect import TargetLoadError, load_target
from agentcheck.runner import inspect_in_subprocess
from agentcheck.runner import orchestrator

POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX interpreter paths required")
EXAMPLE = Path(__file__).parents[2] / "examples" / "evaluation" / "account_agent"

MODULE_LEVEL_AGENT = """
from agents import Agent

agent = Agent(name="Standalone", instructions="Static instructions.", model="gpt-4.1-mini")
"""

FACTORY_AGENT = """
from pathlib import Path
from agents import Agent, function_tool

PROBE = Path(__file__).with_name("FACTORY_SIDE_EFFECT")
HANDLER = Path(__file__).with_name("HANDLER_RAN")


@function_tool
def ping() -> str:
    HANDLER.write_text("original-handler", encoding="utf-8")
    return "pong"


def create_agent():
    PROBE.write_text("factory-ran", encoding="utf-8")
    return Agent(
        name="FactoryAgent",
        instructions="Call ping when asked.",
        tools=[ping],
        model="gpt-4.1-mini",
    )


def unused_factory():
    PROBE.write_text("unused-factory", encoding="utf-8")
    return create_agent()


agent = Agent(name="ModuleAgent", instructions="Module-level.", model="gpt-4.1-mini")
"""

FACTORY_REQUIRES_ARGS = """
from agents import Agent


def build_agent(config):
    return Agent(name="NeedsConfig", instructions="x", model="gpt-4.1-mini")
"""

FACTORY_RETURNS_INT = """
def create_agent():
    return 123
"""

FACTORY_THROWS = """
def create_agent():
    raise RuntimeError("factory exploded api_key=sk-factorysecret12345")
"""

FACTORY_ASYNC = """
async def create_agent():
    return object()
"""

FACTORY_PROVIDER = """
import os


def create_agent():
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY is required during factory construction")
    return object()
"""

MISSING_DEP = """
import definitely_not_an_installed_agentcheck_module  # noqa: F401
from agents import Agent

agent = Agent(name="Missing", instructions="x", model="gpt-4.1-mini")
"""

UNIQUE_DEP_AGENT = """
import agentcheck_unique_dep_mod
from agents import Agent

agent = Agent(
    name=agentcheck_unique_dep_mod.MARKER,
    instructions="Uses a prepared extra package.",
    model="gpt-4.1-mini",
)
"""

HANDOFF_FACTORY = """
from agents import Agent


def create_agent():
    faq = Agent(name="FAQ", instructions="Answer FAQ questions.", model="gpt-4.1-mini")
    return Agent(
        name="Triage",
        instructions="Route the request.",
        handoffs=[faq],
        model="gpt-4.1-mini",
    )
"""

CALLBACK_HANDOFF_FACTORY = """
from agents import Agent, handoff


async def on_seat_booking_handoff(context):
    del context


def create_agent():
    faq = Agent(name="FAQ", instructions="Answer FAQ questions.", model="gpt-4.1-mini")
    return Agent(
        name="Triage",
        instructions="Route the request.",
        handoffs=[handoff(faq, on_handoff=on_seat_booking_handoff)],
        model="gpt-4.1-mini",
    )
"""


def _init(root: Path, entrypoint: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_initial_config(root, entrypoint=entrypoint)
    return root


def test_module_level_agent_still_loads(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "agent.py:agent")
    (root / "agent.py").write_text(MODULE_LEVEL_AGENT.lstrip(), encoding="utf-8")

    loaded, source = load_target(root)
    assert loaded.name == "Standalone"
    assert source.endswith("agent.py:agent")
    assert not source.endswith("()")

    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.ok
    assert result.require_value().identity.name.value == "Standalone"
    assert result.preflight_issues == ()


def test_account_agent_spec_id_remains_stable() -> None:
    """In-process and worker inspection must agree on identity.

    They agree only when both derive it the same way, so this also pins the
    relationship between the two identity modes: supplying the portable
    locator yields the pipeline's ``spec_id``, and omitting it reproduces
    exactly the location-bound value the worker records as ``legacy_spec_id``.
    """

    loaded, source = load_target(EXAMPLE)
    root, config = load_config(EXAMPLE)
    portable = OpenAIAgentsAdapter().inspect(
        loaded, source=source, identity_locator=portable_entrypoint(root, config.entrypoint)
    )
    location_bound = OpenAIAgentsAdapter().inspect(loaded, source=source)

    result = inspect_in_subprocess(root, config)
    worker_spec = result.require_value()

    assert worker_spec.spec_id == portable.spec_id
    assert worker_spec.legacy_spec_id == location_bound.spec_id
    assert worker_spec.spec_id != worker_spec.legacy_spec_id


def test_missing_dependency_fails_clearly_without_pip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _init(tmp_path / "target", "agent.py:agent")
    (root / "agent.py").write_text(MISSING_DEP.lstrip(), encoding="utf-8")
    recorded: list[list[str]] = []
    real_popen = subprocess.Popen
    real_run = subprocess.run

    def _popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        argv = args[0] if args else kwargs.get("args")
        recorded.append([str(item) for item in argv] if argv is not None else [])
        return real_popen(*args, **kwargs)

    def _run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        argv = args[0] if args else kwargs.get("args")
        recorded.append([str(item) for item in argv] if argv is not None else [])
        return real_run(*args, **kwargs)

    monkeypatch.setattr(orchestrator.subprocess, "Popen", _popen)
    monkeypatch.setattr(orchestrator.subprocess, "run", _run)

    with pytest.raises(TargetLoadError, match="definitely_not_an_installed_agentcheck_module"):
        load_target(root)

    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.ok is False
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.code == "target_dependency_missing"
    assert "does not install" in result.infrastructure_error.message
    assert "python_executable" in result.infrastructure_error.message
    assert all("pip" not in " ".join(argv) for argv in recorded)

    assert main(["inspect", str(root)]) == 2
    err = capsys.readouterr().err
    assert "target_dependency_missing" in err
    assert "does not install" in err


def test_explicit_same_interpreter_still_works(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "agent.py:agent")
    (root / "agent.py").write_text(MODULE_LEVEL_AGENT.lstrip(), encoding="utf-8")
    _, config = load_config(root)
    config = apply_python_executable(config, sys.executable)
    result = inspect_in_subprocess(root, config)
    assert result.require_value().identity.name.value == "Standalone"


def test_missing_python_executable_fails_closed(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "agent.py:agent")
    (root / "agent.py").write_text(MODULE_LEVEL_AGENT.lstrip(), encoding="utf-8")
    _, config = load_config(root)
    config = config.model_copy(update={"python_executable": str(root / "missing-python")})
    with pytest.raises(ConfigurationError, match="not an existing interpreter"):
        inspect_in_subprocess(root, config)


@POSIX_ONLY
def test_prepared_venv_with_extra_package_is_used(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "agent.py:agent")
    (root / "agent.py").write_text(UNIQUE_DEP_AGENT.lstrip(), encoding="utf-8")
    venv_dir = root / ".venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        check=True,
        capture_output=True,
    )
    venv_python = venv_dir / "bin" / "python"
    site = subprocess.check_output(
        [str(venv_python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    dep = Path(site) / "agentcheck_unique_dep_mod"
    dep.mkdir()
    (dep / "__init__.py").write_text('MARKER = "PreparedExtra"\n', encoding="utf-8")
    (Path(site) / "agentcheck-parent.pth").write_text(
        sysconfig.get_path("purelib") + "\n", encoding="utf-8"
    )

    _, default_config = load_config(root)
    missing = inspect_in_subprocess(root, default_config)
    assert missing.ok is False
    assert missing.infrastructure_error is not None
    assert missing.infrastructure_error.code == "target_dependency_missing"

    prepared = apply_python_executable(default_config, ".venv/bin/python")
    result = inspect_in_subprocess(root, prepared)
    assert result.ok
    assert result.require_value().identity.name.value == "PreparedExtra"


CUSTOM_AGENT = """
from agentcheck import ToolRuntime, TurnResult


class MinimalCustomAgent:
    name = "Minimal"
    instructions = "Do nothing."
    tools = ()

    def start(self, message: str, tools: ToolRuntime) -> TurnResult:
        return TurnResult(output="ok")


agent = MinimalCustomAgent()
"""


@POSIX_ONLY
def test_worker_python_probe_does_not_require_unrelated_framework_sdks(
    tmp_path: Path,
) -> None:
    """A worker interpreter for one adapter must not need another adapter's SDK.

    Reproduces a real bug found validating an independent PydanticAI target
    (a Slack bolt starter agent) with a dedicated, framework-isolated
    ``--python`` interpreter -- exactly the setup the CLI's own ``--python``
    help text recommends. The probe used to hardcode ``import ... agents``
    unconditionally, so a legitimate pydantic-ai-only (or, as tested here,
    framework-free ``custom``-adapter) worker venv failed closed with a
    misleading "could not import ... openai-agents" error even though that
    adapter needs no such package. A bare venv with only the base AgentCheck
    package installed -- no framework extra at all -- is the sharpest
    reproduction: it must work for a ``custom``-adapter target, which needs
    neither SDK.
    """
    root = _init(tmp_path / "target", "agent.py:agent")
    (root / "agent.py").write_text(CUSTOM_AGENT.lstrip(), encoding="utf-8")

    venv_dir = tmp_path / "bare-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(venv_dir)],
        check=True,
        capture_output=True,
    )
    venv_python = venv_dir / "bin" / "python"
    package_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-q", str(package_root)],
        check=True,
        capture_output=True,
        timeout=180,
    )
    # Confirm the bare venv genuinely lacks both framework SDKs -- otherwise
    # this test would pass for the wrong reason.
    missing = subprocess.run(
        [str(venv_python), "-c", "import agents"],
        capture_output=True,
    )
    assert missing.returncode != 0, "bare venv unexpectedly has openai-agents installed"

    _, config = load_config(root)
    config = config.model_copy(update={"adapter": "custom"})
    config = apply_python_executable(config, str(venv_python))
    result = inspect_in_subprocess(root, config)
    assert result.ok, result.infrastructure_error
    assert result.require_value().identity.name.value == "Minimal"


def test_child_environment_still_omits_credentials_with_explicit_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    root = _init(tmp_path / "target", "agent.py:agent")
    (root / "agent.py").write_text(
        """
import os
from agents import Agent

if "OPENAI_API_KEY" in os.environ:
    raise RuntimeError("provider credential leaked")
agent = Agent(name="Isolated", instructions="Inspect only.", model="gpt-4.1-mini")
""".lstrip(),
        encoding="utf-8",
    )
    _, config = load_config(root)
    config = apply_python_executable(config, sys.executable)
    result = inspect_in_subprocess(root, config)
    assert result.require_value().identity.name.value == "Isolated"


def test_explicit_factory_entrypoint_is_called_once(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "agent.py:create_agent()")
    (root / "agent.py").write_text(FACTORY_AGENT.lstrip(), encoding="utf-8")

    loaded, source = load_target(root)
    assert loaded.name == "FactoryAgent"
    assert source.endswith("agent.py:create_agent()")
    assert (root / "FACTORY_SIDE_EFFECT").read_text(encoding="utf-8") == "factory-ran"
    assert not (root / "HANDLER_RAN").exists()

    (root / "FACTORY_SIDE_EFFECT").unlink()
    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.ok
    spec = result.require_value()
    assert spec.identity.name.value == "FactoryAgent"
    assert [item.value.name for item in spec.tools.items] == ["ping"]
    assert result.preflight_issues == ()
    assert (root / "FACTORY_SIDE_EFFECT").exists()
    assert not (root / "HANDLER_RAN").exists()


def test_arbitrary_factory_functions_are_not_auto_executed(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "agent.py:agent")
    (root / "agent.py").write_text(FACTORY_AGENT.lstrip(), encoding="utf-8")

    loaded, source = load_target(root)
    assert loaded.name == "ModuleAgent"
    assert source.endswith("agent.py:agent")
    assert not (root / "FACTORY_SIDE_EFFECT").exists()

    undeclared = _init(tmp_path / "undeclared", "agent.py:create_agent")
    (undeclared / "agent.py").write_text(FACTORY_AGENT.lstrip(), encoding="utf-8")
    with pytest.raises(TargetLoadError, match="does not auto-call"):
        load_target(undeclared)
    assert not (undeclared / "FACTORY_SIDE_EFFECT").exists()
    assert not (undeclared / "HANDLER_RAN").exists()


def test_factory_returning_non_agent_fails_closed(tmp_path: Path) -> None:
    """A wrong-type target now fails through the same code as any other.

    Previously this specific case (a bare ``TypeError`` raised by
    ``OpenAIAgentsAdapter.inspect``) was pattern-matched into an
    ``unsupported_agent_shape`` code by the worker's error mapping, distinct
    from ``preflight()``'s own ``unsupported_agent_type`` for what is the same
    underlying defect. ``inspect()`` now raises ``UnsupportedTargetError``
    with ``preflight()``'s own issue directly, so both paths agree.
    """

    root = _init(tmp_path / "target", "agent.py:create_agent()")
    (root / "agent.py").write_text(FACTORY_RETURNS_INT.lstrip(), encoding="utf-8")
    loaded, _source = load_target(root)
    assert loaded == 123
    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.ok is False
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.code == "unsupported_agent_type"


def test_factory_exception_is_bounded_and_redacted(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "agent.py:create_agent()")
    (root / "agent.py").write_text(FACTORY_THROWS.lstrip(), encoding="utf-8")
    with pytest.raises(TargetLoadError, match="factory exploded"):
        load_target(root)
    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.code == "target_factory_failed"
    assert "sk-factorysecret12345" not in result.infrastructure_error.message
    assert "[REDACTED]" in result.infrastructure_error.message


def test_factory_requiring_arguments_is_not_called(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "agent.py:build_agent()")
    (root / "agent.py").write_text(FACTORY_REQUIRES_ARGS.lstrip(), encoding="utf-8")
    with pytest.raises(TargetLoadError, match="requires arguments"):
        load_target(root)


def test_async_factory_is_rejected(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "agent.py:create_agent()")
    (root / "agent.py").write_text(FACTORY_ASYNC.lstrip(), encoding="utf-8")
    with pytest.raises(TargetLoadError, match="async"):
        load_target(root)


def test_factory_provider_requirement_does_not_become_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _init(tmp_path / "target", "agent.py:create_agent()")
    (root / "agent.py").write_text(FACTORY_PROVIDER.lstrip(), encoding="utf-8")
    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.ok is False
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.code == "provider_required_during_import"
    assert main(["inspect", str(root)]) == 2
    assert "PASS" not in capsys.readouterr().out


def test_inspect_prints_runtime_mutation_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _init(tmp_path / "target", "agent.py:agent")
    (root / "agent.py").write_text(MODULE_LEVEL_AGENT.lstrip(), encoding="utf-8")
    assert main(["inspect", str(root)]) == 0
    output = capsys.readouterr().out
    assert "Inspection scope:" in output
    assert "not proven absent" in output
    assert "Preflight: supported" in output


def test_factory_handoff_topology_and_callback_still_fail_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    supported = _init(tmp_path / "supported", "agent.py:create_agent()")
    (supported / "agent.py").write_text(HANDOFF_FACTORY.lstrip(), encoding="utf-8")
    assert main(["inspect", str(supported)]) == 0
    supported_out = capsys.readouterr().out
    assert "Handoff topology (2 reachable agents):" in supported_out
    assert "Preflight: supported" in supported_out

    callback = _init(tmp_path / "callback", "agent.py:create_agent()")
    (callback / "agent.py").write_text(CALLBACK_HANDOFF_FACTORY.lstrip(), encoding="utf-8")
    assert main(["inspect", str(callback)]) == 0
    callback_out = capsys.readouterr().out
    assert "handoff_callback" in callback_out
    assert main(["generate", str(callback)]) == 2
    assert "handoff_callback" in capsys.readouterr().err


def test_cli_python_flag_rejects_host_pythonpath_escape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _init(tmp_path / "target", "agent.py:agent")
    (root / "agent.py").write_text(MODULE_LEVEL_AGENT.lstrip(), encoding="utf-8")
    assert main(["inspect", str(root), "--python", "../usr/bin/python"]) == 2
    assert "python_executable" in capsys.readouterr().err
