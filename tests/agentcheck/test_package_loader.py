from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.cli import main
from agentcheck.config import load_config
from agentcheck.initialize import write_initial_config
from agentcheck.inspect import TargetLoadError, load_target
from agentcheck.runner import inspect_in_subprocess


POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX symlinks required")

STANDALONE_AGENT = """
from agents import Agent

agent = Agent(name="Standalone", instructions="Static instructions.", model="gpt-4.1-mini")
"""

PACKAGE_TOOLS = """
from pathlib import Path
from agents import function_tool

PROBE = Path(__file__).with_name("HANDLER_RAN")


@function_tool
def ping() -> str:
    PROBE.write_text("original-handler", encoding="utf-8")
    return "pong"
"""

PACKAGE_AGENTS = """
from agents import Agent
from .tools import ping

agent = Agent(
    name="Packaged",
    instructions="Call ping when asked.",
    tools=[ping],
    model="gpt-4.1-mini",
)
"""

NESTED_UTIL = """
def helper() -> str:
    return "from-parent-package"
"""

NESTED_SIB = """
VALUE = "from-sibling"
"""

NESTED_LEAF = """
from agents import Agent
from ..util import helper
from .sib import VALUE

agent = Agent(
    name="Nested",
    instructions=f"{helper()} {VALUE}",
    model="gpt-4.1-mini",
)
"""

ROOT_HELPER = """
MARKER = "from-examples-auto-mode"
"""

ROOT_IMPORT_AGENT = """
from agents import Agent
from examples.auto_mode import MARKER

agent = Agent(name="RootImport", instructions=MARKER, model="gpt-4.1-mini")
"""

MISSING_RELATIVE = """
from agents import Agent
from .missing import ghost

agent = Agent(name="Broken", instructions="unused", model="gpt-4.1-mini")
"""

HANDOFF_PACKAGE = """
from agents import Agent

faq = Agent(name="FAQ", instructions="Answer FAQ questions.", model="gpt-4.1-mini")
agent = Agent(
    name="Triage",
    instructions="Route the request.",
    handoffs=[faq],
    model="gpt-4.1-mini",
)
"""

PARENT_SECRET = """
SECRET = "must-not-import"
"""

ESCAPE_IMPORT = """
from agents import Agent
from secret import SECRET

agent = Agent(name="Escape", instructions=SECRET, model="gpt-4.1-mini")
"""


def _init(root: Path, entrypoint: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_initial_config(root, entrypoint=entrypoint)
    return root


def test_standalone_file_target_still_loads(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "agent.py:agent")
    (root / "agent.py").write_text(STANDALONE_AGENT.lstrip(), encoding="utf-8")

    loaded, source = load_target(root)
    assert loaded.name == "Standalone"
    assert source.endswith("agent.py:agent")

    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.ok
    assert result.require_value().identity.name.value == "Standalone"
    assert result.preflight_issues == ()


def test_package_relative_tool_import_and_uncalled_handler(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "airline/agents.py:agent")
    package = root / "airline"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "tools.py").write_text(PACKAGE_TOOLS.lstrip(), encoding="utf-8")
    (package / "agents.py").write_text(PACKAGE_AGENTS.lstrip(), encoding="utf-8")

    loaded, _source = load_target(root)
    assert loaded.name == "Packaged"
    assert [tool.name for tool in loaded.tools] == ["ping"]
    assert not (package / "HANDLER_RAN").exists()

    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.ok
    spec = result.require_value()
    assert spec.identity.name.value == "Packaged"
    assert [item.value.name for item in spec.tools.items] == ["ping"]
    assert result.preflight_issues == ()
    assert not (package / "HANDLER_RAN").exists()

    adapter = OpenAIAgentsAdapter()
    report = adapter.preflight(loaded)
    assert report.supported is True


def test_nested_package_relative_imports(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "pkg/sub/leaf.py:agent")
    pkg = root / "pkg"
    sub = pkg / "sub"
    sub.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (sub / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "util.py").write_text(NESTED_UTIL.lstrip(), encoding="utf-8")
    (sub / "sib.py").write_text(NESTED_SIB.lstrip(), encoding="utf-8")
    (sub / "leaf.py").write_text(NESTED_LEAF.lstrip(), encoding="utf-8")

    loaded, _source = load_target(root)
    assert loaded.name == "Nested"
    assert loaded.instructions == "from-parent-package from-sibling"

    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.ok
    assert result.require_value().identity.name.value == "Nested"


def test_repo_root_import_from_examples_auto_mode(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "examples/demo/agent.py:agent")
    examples = root / "examples"
    demo = examples / "demo"
    demo.mkdir(parents=True)
    (examples / "auto_mode.py").write_text(ROOT_HELPER.lstrip(), encoding="utf-8")
    (demo / "agent.py").write_text(ROOT_IMPORT_AGENT.lstrip(), encoding="utf-8")

    loaded, _source = load_target(root)
    assert loaded.name == "RootImport"
    assert loaded.instructions == "from-examples-auto-mode"

    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.ok
    assert result.require_value().identity.name.value == "RootImport"
    assert result.preflight_issues == ()


def test_path_traversal_entrypoint_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "target"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text(STANDALONE_AGENT.lstrip(), encoding="utf-8")
    (root / "agentcheck.json").write_text(
        json.dumps({"entrypoint": "../outside.py:agent"}),
        encoding="utf-8",
    )

    with pytest.raises(TargetLoadError, match="escapes the target directory"):
        load_target(root)


def test_missing_relative_module_is_a_deterministic_error(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "airline/agents.py:agent")
    package = root / "airline"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "agents.py").write_text(MISSING_RELATIVE.lstrip(), encoding="utf-8")

    with pytest.raises(TargetLoadError, match="missing"):
        load_target(root)

    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.ok is False
    assert result.infrastructure_error is not None
    assert result.infrastructure_error.phase == "load"
    assert "missing" in result.infrastructure_error.message


def test_packaged_handoff_target_still_runs_preflight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _init(tmp_path / "target", "airline/agents.py:agent")
    package = root / "airline"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "agents.py").write_text(HANDOFF_PACKAGE.lstrip(), encoding="utf-8")

    assert main(["inspect", str(root)]) == 0
    output = capsys.readouterr().out
    assert "Preflight: supported" in output
    assert "Handoff topology (2 reachable agents):" in output
    assert "- handoffs (agent.handoffs):" not in output

    loaded, _source = load_target(root)
    report = OpenAIAgentsAdapter().preflight(loaded)
    assert report.supported is True
    assert "handoffs" not in {issue.code for issue in report.issues}


def test_parent_directory_is_not_added_to_sys_path(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "airline/agents.py:agent")
    (tmp_path / "secret.py").write_text(PARENT_SECRET.lstrip(), encoding="utf-8")
    package = root / "airline"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "agents.py").write_text(ESCAPE_IMPORT.lstrip(), encoding="utf-8")

    with pytest.raises(TargetLoadError, match="secret"):
        load_target(root)


@POSIX_ONLY
def test_entrypoint_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "target"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text(STANDALONE_AGENT.lstrip(), encoding="utf-8")
    (root / "agent.py").symlink_to(outside)
    (root / "agentcheck.json").write_text(
        json.dumps({"entrypoint": "agent.py:agent"}),
        encoding="utf-8",
    )

    with pytest.raises(TargetLoadError, match="escapes the target directory"):
        load_target(root)


@POSIX_ONLY
def test_package_relative_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = _init(tmp_path / "target", "airline/agents.py:agent")
    outside = tmp_path / "outside_tools.py"
    outside.write_text("ping = object()\n", encoding="utf-8")
    package = root / "airline"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "tools.py").symlink_to(outside)
    (package / "agents.py").write_text(
        "from agents import Agent\nfrom .tools import ping\n"
        "agent = Agent(name='Unsafe', instructions='x', tools=[ping], model='gpt-4.1-mini')\n",
        encoding="utf-8",
    )

    with pytest.raises(TargetLoadError, match="escapes the target directory"):
        load_target(root)

    _, config = load_config(root)
    result = inspect_in_subprocess(root, config)
    assert result.ok is False
    assert result.infrastructure_error is not None
    assert "escapes the target directory" in result.infrastructure_error.message
