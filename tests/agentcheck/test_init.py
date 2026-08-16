from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from agentcheck.cli import main
from agentcheck.config import CONFIG_FILENAME, AgentCheckConfig, load_config
from agentcheck.errors import ConfigurationError
from agentcheck.initialize import (
    DEFAULT_ADAPTER,
    SUPPORTED_ADAPTERS,
    write_initial_config,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE_CONFIG = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent" / CONFIG_FILENAME
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes required")


def _target(tmp_path: Path, *, with_agent: bool = True) -> Path:
    root = tmp_path / "target"
    root.mkdir()
    if with_agent:
        (root / "agent.py").write_text("agent = object()\n", encoding="utf-8")
    return root


def test_supported_adapters_track_the_config_contract() -> None:
    assert SUPPORTED_ADAPTERS == ("openai_agents",)
    assert DEFAULT_ADAPTER in SUPPORTED_ADAPTERS
    assert AgentCheckConfig().adapter == DEFAULT_ADAPTER


def test_written_config_round_trips_through_load_config(tmp_path: Path) -> None:
    root = _target(tmp_path)

    config_path = write_initial_config(root)

    assert config_path == root / CONFIG_FILENAME
    loaded_root, loaded = load_config(root)
    assert loaded_root == root
    assert loaded == AgentCheckConfig()
    assert loaded.schema_version == "agentcheck.config.v1"


def test_written_document_is_the_complete_default_contract(tmp_path: Path) -> None:
    root = _target(tmp_path)

    config_path = write_initial_config(root)

    document = json.loads(config_path.read_text(encoding="utf-8"))
    assert document == {
        "schema_version": "agentcheck.config.v1",
        "adapter": "openai_agents",
        "entrypoint": "agent.py:agent",
        "suite": "account_support_v1",
        "seed": 1729,
        "max_concurrency": 2,
        "environment_allowlist": [],
        "include_instructions_in_report": False,
        "artifacts_directory": ".agentcheck",
    }
    assert document == json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert config_path.read_text(encoding="utf-8").endswith("}\n")


def test_written_config_never_declares_an_environment_allowlist(tmp_path: Path) -> None:
    root = _target(tmp_path)

    config_path = write_initial_config(root)

    text = config_path.read_text(encoding="utf-8")
    assert json.loads(text)["environment_allowlist"] == []
    assert "API_KEY" not in text
    assert "token" not in text.casefold()


def test_custom_adapter_and_entrypoint_are_persisted(tmp_path: Path) -> None:
    root = _target(tmp_path, with_agent=False)
    (root / "src").mkdir()
    (root / "src" / "support.py").write_text("support = object()\n", encoding="utf-8")

    config_path = write_initial_config(
        root,
        adapter="openai_agents",
        entrypoint="src/support.py:support",
    )

    _, loaded = load_config(root)
    assert loaded.entrypoint == "src/support.py:support"
    assert json.loads(config_path.read_text(encoding="utf-8"))["entrypoint"] == (
        "src/support.py:support"
    )


@POSIX_ONLY
def test_written_config_uses_owner_only_permissions(tmp_path: Path) -> None:
    root = _target(tmp_path)

    config_path = write_initial_config(root)

    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_existing_config_is_never_replaced_without_force(tmp_path: Path) -> None:
    root = _target(tmp_path)
    existing = root / CONFIG_FILENAME
    existing.write_text('{"schema_version": "agentcheck.config.v1", "seed": 7}\n', encoding="utf-8")

    with pytest.raises(ConfigurationError, match="already exists"):
        write_initial_config(root)

    assert json.loads(existing.read_text(encoding="utf-8"))["seed"] == 7


def test_force_replaces_an_existing_config(tmp_path: Path) -> None:
    root = _target(tmp_path)
    write_initial_config(root)
    (root / CONFIG_FILENAME).write_text("not json at all", encoding="utf-8")

    config_path = write_initial_config(root, force=True)

    assert load_config(root)[1] == AgentCheckConfig()
    if os.name == "posix":
        assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_repeated_initialization_is_deterministic(tmp_path: Path) -> None:
    first_root = _target(tmp_path)
    second_root = tmp_path / "second"
    second_root.mkdir()
    (second_root / "agent.py").write_text("agent = object()\n", encoding="utf-8")

    first = write_initial_config(first_root).read_bytes()
    rewritten = write_initial_config(first_root, force=True).read_bytes()
    second = write_initial_config(second_root).read_bytes()

    assert first == rewritten == second


def test_forced_rewrite_leaves_no_temporary_file(tmp_path: Path) -> None:
    root = _target(tmp_path)
    write_initial_config(root)

    write_initial_config(root, force=True)

    assert sorted(path.name for path in root.iterdir()) == sorted([CONFIG_FILENAME, "agent.py"])


@pytest.mark.parametrize(
    "entrypoint",
    [
        "../outside.py:agent",
        "nested/../../outside.py:agent",
        "/etc/passwd.py:agent",
        "agent.txt:agent",
        "agent.py",
        "agent.py:2bad",
        "agent.py:agent:extra",
        "",
    ],
)
def test_unsafe_or_malformed_entrypoints_are_rejected(tmp_path: Path, entrypoint: str) -> None:
    root = _target(tmp_path)
    (tmp_path / "outside.py").write_text("agent = object()\n", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        write_initial_config(root, entrypoint=entrypoint)

    assert not (root / CONFIG_FILENAME).exists()


@POSIX_ONLY
def test_entrypoint_symlinked_outside_the_target_is_rejected(tmp_path: Path) -> None:
    root = _target(tmp_path, with_agent=False)
    outside = tmp_path / "outside.py"
    outside.write_text("agent = object()\n", encoding="utf-8")
    (root / "agent.py").symlink_to(outside)

    with pytest.raises(ConfigurationError, match="inside the target"):
        write_initial_config(root)

    assert not (root / CONFIG_FILENAME).exists()


@POSIX_ONLY
def test_symlinked_destination_is_refused_rather_than_followed(tmp_path: Path) -> None:
    root = _target(tmp_path)
    outside = tmp_path / "captured.json"
    outside.write_text("original\n", encoding="utf-8")
    (root / CONFIG_FILENAME).symlink_to(outside)

    with pytest.raises(ConfigurationError, match="already exists"):
        write_initial_config(root)

    assert outside.read_text(encoding="utf-8") == "original\n"


@POSIX_ONLY
def test_forced_write_replaces_a_symlink_without_writing_through_it(tmp_path: Path) -> None:
    root = _target(tmp_path)
    outside = tmp_path / "captured.json"
    outside.write_text("original\n", encoding="utf-8")
    (root / CONFIG_FILENAME).symlink_to(outside)

    config_path = write_initial_config(root, force=True)

    assert outside.read_text(encoding="utf-8") == "original\n"
    assert not config_path.is_symlink()
    assert load_config(root)[1] == AgentCheckConfig()


def test_unknown_adapter_is_rejected(tmp_path: Path) -> None:
    root = _target(tmp_path)

    with pytest.raises(ConfigurationError, match="unsupported adapter"):
        write_initial_config(root, adapter="langchain")

    assert not (root / CONFIG_FILENAME).exists()


def test_missing_target_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="existing directory"):
        write_initial_config(tmp_path / "absent")


def test_file_target_is_rejected_rather_than_redirected_to_its_parent(tmp_path: Path) -> None:
    root = _target(tmp_path)

    with pytest.raises(ConfigurationError, match="existing directory"):
        write_initial_config(root / "agent.py")

    assert not (root / CONFIG_FILENAME).exists()


@POSIX_ONLY
@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses directory permissions")
def test_unwritable_target_reports_an_actionable_error(tmp_path: Path) -> None:
    root = _target(tmp_path)
    root.chmod(0o500)
    try:
        with pytest.raises(ConfigurationError, match="unable to write"):
            write_initial_config(root)
    finally:
        root.chmod(0o700)


def test_initialization_creates_nothing_but_the_configuration(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    write_initial_config(root)

    assert [path.name for path in root.iterdir()] == [CONFIG_FILENAME]


def test_initialization_never_spawns_a_child_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _target(tmp_path)

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("init must not execute or import the target")

    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    monkeypatch.setattr(subprocess, "run", _forbidden)

    assert main(["init", str(root)]) == 0


def test_cli_reports_the_written_configuration(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _target(tmp_path)

    assert main(["init", str(root)]) == 0

    output = capsys.readouterr()
    assert output.err == ""
    assert "AgentCheck configuration written." in output.out
    assert str(root / CONFIG_FILENAME) in output.out
    assert "Adapter:    openai_agents" in output.out
    assert "Entrypoint: agent.py:agent" in output.out
    assert f"- agentcheck inspect {root}" in output.out
    assert f"- agentcheck generate {root}" in output.out
    assert f"- agentcheck test {root}" in output.out
    assert "does not exist yet" not in output.out


def test_cli_notes_a_missing_entrypoint_without_failing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    assert main(["init", str(root)]) == 0

    output = capsys.readouterr()
    assert "the entrypoint source does not exist yet" in output.out
    assert str(root / "agent.py") in output.out


def test_cli_refuses_to_clobber_and_recovers_with_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _target(tmp_path)

    assert main(["init", str(root)]) == 0
    capsys.readouterr()

    assert main(["init", str(root)]) == 2
    refusal = capsys.readouterr()
    assert "already exists" in refusal.err
    assert "--force" in refusal.err

    assert main(["init", str(root), "--force"]) == 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["init", "--entrypoint", "../outside.py:agent"],
        ["init", "--entrypoint", "notes.txt:agent"],
    ],
)
def test_cli_exits_two_on_unsafe_entrypoints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], arguments: list[str]
) -> None:
    root = _target(tmp_path)

    assert main([*arguments, str(root)]) == 2

    assert "AgentCheck error:" in capsys.readouterr().err
    assert not (root / CONFIG_FILENAME).exists()


def test_cli_exits_two_on_a_missing_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["init", str(tmp_path / "absent")]) == 2
    assert "existing directory" in capsys.readouterr().err


def test_cli_rejects_an_unknown_adapter_with_exit_two(tmp_path: Path) -> None:
    root = _target(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["init", str(root), "--adapter", "langchain"])

    assert excinfo.value.code == 2
    assert not (root / CONFIG_FILENAME).exists()


def test_existing_commands_remain_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "init" in help_text
    assert "inspect" in help_text
    assert "generate" in help_text
    assert "test" in help_text
    assert "report" in help_text
    assert "replay" in help_text
    assert "shrink" in help_text
