from __future__ import annotations

import json
from pathlib import Path

import pytest

from pydantic import ValidationError

from agentcheck.config import (
    AgentCheckConfig,
    child_environment,
    contained_path,
    entrypoint_location,
    load_config,
    resolve_entrypoint,
)
from agentcheck.errors import ConfigurationError


def test_load_config_uses_versioned_strict_contract(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("agent = object()\n", encoding="utf-8")
    (tmp_path / "agentcheck.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.config.v1",
                "entrypoint": "agent.py:agent",
                "seed": 41,
            }
        ),
        encoding="utf-8",
    )

    root, config = load_config(tmp_path)

    assert root == tmp_path.resolve()
    assert config.seed == 41
    assert config.adapter == "openai_agents"


def test_entrypoint_cannot_escape_target(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"
    outside.write_text("agent = object()\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="inside the target"):
        resolve_entrypoint(tmp_path, "../outside.py:agent")


def test_entrypoint_location_contains_paths_without_requiring_the_source(
    tmp_path: Path,
) -> None:
    source, attribute = entrypoint_location(tmp_path, "src/agent.py:agent")

    assert source == tmp_path.resolve() / "src" / "agent.py"
    assert attribute == "agent"
    assert not source.exists()

    with pytest.raises(ConfigurationError, match="inside the target"):
        entrypoint_location(tmp_path, "../outside.py:agent")


def test_resolve_entrypoint_still_requires_an_existing_source(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        resolve_entrypoint(tmp_path, "agent.py:agent")


def test_child_environment_omits_unapproved_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("AGENTCHECK_ALLOWED_TEST", "visible")
    monkeypatch.setenv("LANG", "C.UTF-8")

    environment = child_environment(
        AgentCheckConfig(environment_allowlist=("AGENTCHECK_ALLOWED_TEST",))
    )

    assert environment["AGENTCHECK_ALLOWED_TEST"] == "visible"
    assert environment["LANG"] == "C.UTF-8"
    assert "OPENAI_API_KEY" not in environment


def test_suite_path_is_optional_and_keeps_config_v1(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("agent = object()\n", encoding="utf-8")
    (tmp_path / "agentcheck.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.config.v1",
                "entrypoint": "agent.py:agent",
            }
        ),
        encoding="utf-8",
    )

    _, config = load_config(tmp_path)

    assert config.suite_path is None
    assert AgentCheckConfig().suite_path is None
    assert AgentCheckConfig(suite_path="suites/frozen.json").suite_path == (
        "suites/frozen.json"
    )


def test_suite_path_rejects_unsafe_locations() -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        AgentCheckConfig(suite_path="../escape.json")
    with pytest.raises(ValueError, match="safe relative path"):
        AgentCheckConfig(suite_path="/tmp/frozen.json")
    with pytest.raises(ValueError, match="must not be empty"):
        AgentCheckConfig(suite_path="   ")
    with pytest.raises(ValidationError):
        AgentCheckConfig.model_validate(
            {
                "schema_version": "agentcheck.config.v1",
                "suite_path": "ok.json",
                "unexpected": True,
            }
        )


def test_contained_path_refuses_traversal_and_outbound_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "target"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")

    assert contained_path(root, "agentcheck-suite.json") == root / "agentcheck-suite.json"
    with pytest.raises(ConfigurationError, match="relative"):
        contained_path(root, str(outside))
    with pytest.raises(ConfigurationError, match="safe relative path"):
        contained_path(root, "../outside.json")
    with pytest.raises(ConfigurationError, match="must not be empty"):
        contained_path(root, "")

    (root / "suite.json").symlink_to(outside)
    with pytest.raises(ConfigurationError, match="inside the target"):
        contained_path(root, "suite.json")
