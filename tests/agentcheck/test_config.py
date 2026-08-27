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


def test_suite_accepts_only_the_one_bundled_reference_suite() -> None:
    """``suite`` identifies which bundled reference suite applies -- the same
    kind of forward-compatible identifier ``schema_version`` is -- not a
    general suite selector, so a name that matches no bundled suite is
    refused with an explanation, rather than pydantic's bare Literal-mismatch
    text or (worse) being silently accepted."""

    assert AgentCheckConfig().suite == "account_support_v1"
    assert AgentCheckConfig(suite="account_support_v1").suite == "account_support_v1"
    with pytest.raises(ValueError, match="is not one of the suites AgentCheck ships"):
        AgentCheckConfig(suite="my_custom_suite")
    with pytest.raises(ValueError, match="suite_path to your own frozen suite file"):
        AgentCheckConfig(suite="my_custom_suite")


def test_an_invalid_suite_in_agentcheck_json_names_the_field_clearly(
    tmp_path: Path,
) -> None:
    (tmp_path / "agent.py").write_text("agent = object()\n", encoding="utf-8")
    (tmp_path / "agentcheck.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.config.v1",
                "entrypoint": "agent.py:agent",
                "suite": "my_custom_suite",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="is not one of the suites AgentCheck ships"):
        load_config(tmp_path)


def test_store_path_rejects_unsafe_locations() -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        AgentCheckConfig(store_path="../escape.sqlite")
    with pytest.raises(ValueError, match="safe relative path"):
        AgentCheckConfig(store_path="/tmp/agentcheck.sqlite")
    with pytest.raises(ValueError, match="must not be empty"):
        AgentCheckConfig(store_path="   ")


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


def test_factory_entrypoint_is_accepted_and_python_executable_is_optional() -> None:
    config = AgentCheckConfig(entrypoint="src/agent.py:create_agent()")
    assert config.entrypoint == "src/agent.py:create_agent()"
    assert config.python_executable is None
    assert AgentCheckConfig().python_executable is None


def test_python_executable_rejects_unsafe_relative_paths() -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        AgentCheckConfig(python_executable="../usr/bin/python")
    with pytest.raises(ValueError, match="must not be empty"):
        AgentCheckConfig(python_executable="   ")
    with pytest.raises(ValidationError):
        AgentCheckConfig.model_validate(
            {
                "schema_version": "agentcheck.config.v1",
                "python_executable": "/usr/bin/python3",
                "unexpected": True,
            }
        )
