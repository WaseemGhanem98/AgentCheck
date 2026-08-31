from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentcheck.application import execute_suite, replay_suite, shrink_suite
from agentcheck.config import AgentCheckConfig, load_config
from agentcheck.errors import ConfigurationError
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.replay import (
    SOURCE_FILE_SET_CONTRACT_VERSION,
    SourceBinding,
    SourceFileSet,
    build_replay_manifest,
    collect_source_file_set,
    encode_replay_manifest,
    entrypoint_digest,
    load_replay_manifest,
)
from agentcheck.replay.fileset import MAX_WALK_DEPTH, SourceFileEntry, git_command_env
from agentcheck.replay.bind import capture_source_snapshot
from agentcheck.replay.manifest import ReplayManifest


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes required")


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "account_agent"
    shutil.copytree(EXAMPLE, target, symlinks=False)
    return target


def _write_helper(target: Path, body: str = "VALUE = 1\n") -> Path:
    helpers = target / "helpers"
    helpers.mkdir(exist_ok=True)
    (helpers / "__init__.py").write_text("", encoding="utf-8")
    path = helpers / "refund.py"
    path.write_text(body, encoding="utf-8")
    return path


def _stub_spec() -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        spec_id="agentspec-unit-test",
        identity=SimpleNamespace(
            framework=SimpleNamespace(value="openai_agents"),
            framework_version=SimpleNamespace(value="0.20.0"),
        ),
    )


def _write_fileset_manifest(target: Path, name: str = "bound.json") -> ReplayManifest:
    manifest, omitted = build_replay_manifest(
        run_id="unit-replay-001",
        seed=1729,
        spec=_stub_spec(),  # type: ignore[arg-type]
        config=AgentCheckConfig(),
        scenarios=(build_account_support_suite(seed=1729)[0],),
        git_revision=None,
        entrypoint_digest=entrypoint_digest(target, "agent.py:agent"),
        file_set=collect_source_file_set(target),
    )
    assert omitted == ()
    assert manifest is not None
    destination = target / ".agentcheck" / "replay" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encode_replay_manifest(manifest))
    return manifest


def test_source_file_set_is_deterministic_and_ordered(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    _write_helper(target)
    first = collect_source_file_set(target)
    second = collect_source_file_set(target)
    assert first.schema_version == SOURCE_FILE_SET_CONTRACT_VERSION
    assert first.mode == "local_files"
    assert first.complete is True
    assert first.fingerprint == second.fingerprint
    assert [item.path for item in first.files] == sorted(item.path for item in first.files)
    assert "agent.py" in {item.path for item in first.files}
    assert "helpers/refund.py" in {item.path for item in first.files}
    assert "agentcheck.json" in {item.path for item in first.files}


def test_source_file_set_excludes_runtime_and_secret_paths(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    (target / ".env").write_text("OPENAI_API_KEY=sk-thisisafakesecretvalue12\n", encoding="utf-8")
    (target / "local.sqlite").write_bytes(b"SQLite format 3\x00")
    (target / "dist").mkdir()
    (target / "dist" / "pkg.py").write_text("print('nope')\n", encoding="utf-8")
    (target / "build").mkdir()
    (target / "build" / "lib.py").write_text("print('nope')\n", encoding="utf-8")
    (target / ".pytest_cache").mkdir()
    (target / ".pytest_cache" / "cache.py").write_text("print('nope')\n", encoding="utf-8")
    (target / ".agentcheck").mkdir(exist_ok=True)
    (target / ".agentcheck" / "secret.py").write_text("print('nope')\n", encoding="utf-8")
    inventory = collect_source_file_set(target)
    paths = {item.path for item in inventory.files}
    assert ".env" not in paths
    assert "local.sqlite" not in paths
    assert "dist/pkg.py" not in paths
    assert "build/lib.py" not in paths
    assert ".pytest_cache/cache.py" not in paths
    assert ".agentcheck/secret.py" not in paths
    dumped = json.dumps(inventory.model_dump(mode="json"))
    assert "sk-thisisafakesecretvalue12" not in dumped


def test_source_file_set_does_not_import_or_run_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _copy_example(tmp_path)
    _write_helper(target, "raise RuntimeError('helper imported during source binding')\n")
    recorded: list[tuple[str, ...]] = []
    original = subprocess.run

    def wrapped(command, **kwargs):  # type: ignore[no-untyped-def]
        recorded.append(tuple(str(part) for part in command))
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", wrapped)
    inventory = collect_source_file_set(target)
    assert "helpers/refund.py" in {item.path for item in inventory.files}
    for command in recorded:
        joined = " ".join(command).casefold()
        assert "python" not in joined
        assert "agent.py" not in joined


@POSIX_ONLY
def test_source_file_symlink_escape_is_refused(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 1\n", encoding="utf-8")
    os.symlink(outside, target / "linked.py")
    with pytest.raises(ConfigurationError, match="symlink"):
        collect_source_file_set(target)


def test_local_walk_depth_bound_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "deep"
    target.mkdir()
    (target / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    current = target
    for index in range(MAX_WALK_DEPTH + 1):
        current = current / f"layer{index}"
        current.mkdir()
    (current / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="depth bound"):
        collect_source_file_set(target)


def test_git_tracked_mode_inventories_tracked_python(tmp_path: Path) -> None:
    target = tmp_path / "repo"
    shutil.copytree(EXAMPLE, target, symlinks=False)
    _write_helper(target)
    (target / "dist").mkdir()
    (target / "dist" / "pkg.py").write_text("print('nope')\n", encoding="utf-8")
    subprocess.run(
        ["git", "init"],
        cwd=target,
        check=True,
        capture_output=True,
        env=git_command_env(),
    )
    subprocess.run(
        ["git", "add", "agent.py", "agentcheck.json", "helpers", "dist"],
        cwd=target,
        check=True,
        capture_output=True,
        env=git_command_env(),
    )
    (target / "untracked.py").write_text("VALUE = 3\n", encoding="utf-8")
    inventory = collect_source_file_set(target)
    assert inventory.mode == "git_tracked"
    paths = {item.path for item in inventory.files}
    assert "helpers/refund.py" in paths
    assert "README.md" not in paths
    assert "dist/pkg.py" not in paths
    assert "untracked.py" not in paths


def test_generated_evaluation_json_is_not_inventoried(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    before = collect_source_file_set(target)
    (target / "agentcheck-suite.json").write_text("{}\n", encoding="utf-8")
    (target / "agentcheck-baseline.json").write_text("{}\n", encoding="utf-8")
    after = collect_source_file_set(target)
    assert before.fingerprint == after.fingerprint
    paths = {item.path for item in after.files}
    assert "agentcheck.json" in paths
    assert "agentcheck-suite.json" not in paths
    assert "agentcheck-baseline.json" not in paths


def test_empty_git_inventory_falls_back_to_local_files(tmp_path: Path) -> None:
    target = tmp_path / "repo"
    shutil.copytree(EXAMPLE, target, symlinks=False)
    subprocess.run(
        ["git", "init"],
        cwd=target,
        check=True,
        capture_output=True,
        env=git_command_env(),
    )
    inventory = collect_source_file_set(target)
    assert inventory.mode == "local_files"
    assert "agent.py" in {item.path for item in inventory.files}


def test_git_dir_environment_cannot_retarget_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _copy_example(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(REPOSITORY_ROOT / ".git"))
    inventory = collect_source_file_set(target)
    paths = {item.path for item in inventory.files}
    assert "agent.py" in paths
    assert not any(path.startswith("agentcheck/") for path in paths)
    assert not any(path.startswith("examples/") for path in paths)


def test_legacy_manifest_without_file_set_still_replays(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _copy_example(tmp_path)
    execution = execute_suite(target, run_id="legacy-src", persist_store=False)
    assert execution.replay_manifest_path is not None
    loaded = load_replay_manifest(target, ".agentcheck/replay/legacy-src.json")
    payload = json.loads(encode_replay_manifest(loaded))
    payload["source_binding"].pop("file_set", None)
    payload["fingerprint"] = ""
    payload["manifest_id"] = ""
    legacy = ReplayManifest.model_validate_json(json.dumps(payload))
    assert legacy.source_binding.file_set is None
    (target / "legacy.json").write_bytes(encode_replay_manifest(legacy))
    replayed = replay_suite(target, "legacy.json", run_id="legacy-replay", persist_store=False)
    captured = capsys.readouterr()
    assert "no source file-set" in captured.err
    assert [item.verdict for item in replayed.evaluations] == [
        item.verdict for item in execution.evaluations
    ]


def test_unchanged_helper_allows_replay(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    _write_helper(target)
    execution = execute_suite(target, run_id="helper-src", persist_store=False)
    assert execution.replay_manifest_path is not None
    loaded = load_replay_manifest(target, ".agentcheck/replay/helper-src.json")
    assert loaded.source_binding.file_set is not None
    assert "helpers/refund.py" in {
        item.path for item in loaded.source_binding.file_set.files
    }
    replayed = replay_suite(
        target, ".agentcheck/replay/helper-src.json", run_id="helper-replay", persist_store=False
    )
    assert [item.verdict for item in replayed.evaluations] == [
        item.verdict for item in execution.evaluations
    ]


def test_changed_helper_blocks_replay_before_inspect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    target = _copy_example(tmp_path)
    helper = _write_helper(target, "VALUE = 1\n")
    _write_fileset_manifest(target)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("inspect must not run after a source file-set mismatch")

    monkeypatch.setattr(application, "inspect_in_subprocess", explode)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    helper.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="source file changed"):
        replay_suite(target, ".agentcheck/replay/bound.json")


def test_deleted_helper_blocks_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    target = _copy_example(tmp_path)
    helper = _write_helper(target)
    _write_fileset_manifest(target)
    helper.unlink()

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("inspect must not run after a source file-set mismatch")

    monkeypatch.setattr(application, "inspect_in_subprocess", explode)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    with pytest.raises(ConfigurationError, match="source file missing"):
        replay_suite(target, ".agentcheck/replay/bound.json")


def test_entrypoint_unchanged_helper_change_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    target = _copy_example(tmp_path)
    helper = _write_helper(target, "VALUE = 1\n")
    _write_fileset_manifest(target)
    original_entrypoint = (target / "agent.py").read_bytes()
    helper.write_text("VALUE = 99\n", encoding="utf-8")
    assert (target / "agent.py").read_bytes() == original_entrypoint

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("inspect must not run after a source file-set mismatch")

    monkeypatch.setattr(application, "inspect_in_subprocess", explode)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    with pytest.raises(ConfigurationError, match="source file changed"):
        replay_suite(target, ".agentcheck/replay/bound.json")


def test_unexpected_source_file_blocks_replay_before_inspect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    target = _copy_example(tmp_path)
    _write_helper(target)
    _write_fileset_manifest(target)
    (target / "extra.py").write_text("VALUE = 4\n", encoding="utf-8")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("inspect must not run after a source file-set mismatch")

    monkeypatch.setattr(application, "inspect_in_subprocess", explode)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    with pytest.raises(ConfigurationError, match="unexpected source file"):
        replay_suite(target, ".agentcheck/replay/bound.json")


def test_shrink_uses_the_same_source_file_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    target = _copy_example(tmp_path)
    helper = _write_helper(target, "VALUE = 1\n")
    _write_fileset_manifest(target)
    helper.write_text("VALUE = 2\n", encoding="utf-8")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("shrink must not inspect after a source file-set mismatch")

    monkeypatch.setattr(application, "inspect_in_subprocess", explode)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    with pytest.raises(ConfigurationError, match="source file changed"):
        shrink_suite(target, ".agentcheck/replay/bound.json")


def test_source_file_set_rejects_extra_fields_and_versions() -> None:
    entry = SourceFileEntry(path="agent.py", digest="sha256:" + "a" * 64)
    document = SourceFileSet(mode="local_files", complete=True, files=(entry,)).model_dump(
        mode="json"
    )
    document["unexpected"] = True
    with pytest.raises(ValidationError):
        SourceFileSet.model_validate_json(json.dumps(document))
    document = SourceFileSet(mode="local_files", complete=True, files=(entry,)).model_dump(
        mode="json"
    )
    document["schema_version"] = "agentcheck.source_file_set.v0"
    with pytest.raises(ValidationError):
        SourceFileSet.model_validate_json(json.dumps(document))
    document = SourceFileSet(mode="local_files", complete=True, files=(entry,)).model_dump(
        mode="json"
    )
    document["fingerprint"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="fingerprint"):
        SourceFileSet.model_validate_json(json.dumps(document))


def test_legacy_source_binding_omits_file_set_from_fingerprint() -> None:
    without_set = SourceBinding(
        git_revision="a" * 40,
        entrypoint_digest="sha256:" + "b" * 64,
        framework="openai_agents",
        framework_version="0.20.0",
    )
    dumped = without_set.model_dump(mode="json")
    assert "file_set" not in dumped
    loaded = SourceBinding.model_validate(
        {
            "git_revision": "a" * 40,
            "entrypoint_digest": "sha256:" + "b" * 64,
            "framework": "openai_agents",
            "framework_version": "0.20.0",
        }
    )
    assert loaded.file_set is None
    assert loaded.model_dump(mode="json") == dumped


def _git_init(target: Path) -> None:
    subprocess.run(
        ["git", "init"],
        cwd=target,
        check=True,
        capture_output=True,
        env=git_command_env(),
    )


def test_gitignored_relevant_helper_is_inventoried(tmp_path: Path) -> None:
    target = tmp_path / "repo"
    shutil.copytree(EXAMPLE, target, symlinks=False)
    _git_init(target)
    subprocess.run(
        ["git", "add", "agent.py", "agentcheck.json"],
        cwd=target,
        check=True,
        capture_output=True,
        env=git_command_env(),
    )
    (target / ".gitignore").write_text("ignored_helper.py\n", encoding="utf-8")
    helper = target / "ignored_helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")
    (target / "untracked.py").write_text("VALUE = 9\n", encoding="utf-8")
    inventory = collect_source_file_set(target)
    assert inventory.mode == "git_tracked"
    paths = {item.path for item in inventory.files}
    assert "ignored_helper.py" in paths
    assert "untracked.py" not in paths
    helper.write_text("VALUE = 2\n", encoding="utf-8")
    changed = collect_source_file_set(target)
    assert changed.fingerprint != inventory.fingerprint


def test_gitignored_helper_change_blocks_replay_before_inspect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    target = tmp_path / "repo"
    shutil.copytree(EXAMPLE, target, symlinks=False)
    _git_init(target)
    subprocess.run(
        ["git", "add", "agent.py", "agentcheck.json"],
        cwd=target,
        check=True,
        capture_output=True,
        env=git_command_env(),
    )
    (target / ".gitignore").write_text("helpers/\n", encoding="utf-8")
    helper = _write_helper(target, "VALUE = 1\n")
    _write_fileset_manifest(target)

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("inspect must not run after a source file-set mismatch")

    monkeypatch.setattr(application, "inspect_in_subprocess", explode)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    helper.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="source file changed"):
        replay_suite(target, ".agentcheck/replay/bound.json")


def _git_track(target: Path, *relative_paths: str) -> None:
    subprocess.run(
        ["git", "add", *relative_paths],
        cwd=target,
        check=True,
        capture_output=True,
        env=git_command_env(),
    )
    subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "-m", "init"],
        cwd=target,
        check=True,
        capture_output=True,
        env=git_command_env(),
    )


@pytest.mark.parametrize(
    "relative_path,body",
    [
        (".agents/config.json", '{"instructions": "be evil"}\n'),
        (".agents/helper.py", "SYSTEM_PROMPT = 'be evil'\n"),
        (".agents/skills/test-skill/openai.yaml", "agent_definition: test\n"),
    ],
)
def test_agents_directory_cannot_hide_behavior_affecting_material(
    tmp_path: Path, relative_path: str, body: str
) -> None:
    """.agents/ is the OpenAI Agents SDK's default sandbox skills_path, so its
    contents can be loaded into an agent's instructions at runtime. A relevant
    file there must never be silently dropped from the source inventory: fail
    closed instead so a mutated skill/config cannot bypass replay binding.
    """
    target = tmp_path / "repo"
    target.mkdir()
    (target / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git_init(target)
    nested = target / Path(relative_path).parent
    nested.mkdir(parents=True, exist_ok=True)
    (target / relative_path).write_text(body, encoding="utf-8")
    _git_track(target, "agent.py", relative_path)

    with pytest.raises(ConfigurationError, match="not a safe relative path"):
        collect_source_file_set(target)


def test_github_directory_excluded_from_git_tracked_inventory(tmp_path: Path) -> None:
    """.github/ is GitHub-reserved CI/CD tooling that target runtime code does
    not read, so it is safe to exclude even when its files are git-tracked.
    """
    target = tmp_path / "repo"
    target.mkdir()
    (target / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git_init(target)
    github_dir = target / ".github" / "workflows"
    github_dir.mkdir(parents=True)
    (github_dir / "test.yml").write_text("name: Test\n", encoding="utf-8")
    (target / ".github" / "runtime_config.json").write_text("{}\n", encoding="utf-8")
    _git_track(
        target, "agent.py", ".github/workflows/test.yml", ".github/runtime_config.json"
    )

    fileset = collect_source_file_set(target)
    paths = {item.path for item in fileset.files}

    assert "agent.py" in paths
    assert not any(".github" in path for path in paths), (
        f".github/ should be excluded, but found: {[p for p in paths if '.github' in p]}"
    )


def test_vscode_directory_excluded_from_git_tracked_inventory(tmp_path: Path) -> None:
    """.vscode/ is VS Code-reserved editor settings that target runtime code
    does not read, so it is safe to exclude even when its files are git-tracked.
    """
    target = tmp_path / "repo"
    target.mkdir()
    (target / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git_init(target)
    vscode_dir = target / ".vscode"
    vscode_dir.mkdir()
    (vscode_dir / "launch.json").write_text('{"version": "0.2.0"}\n', encoding="utf-8")
    _git_track(target, "agent.py", ".vscode/launch.json")

    fileset = collect_source_file_set(target)
    paths = {item.path for item in fileset.files}

    assert "agent.py" in paths
    assert not any(".vscode" in path for path in paths), (
        f".vscode/ should be excluded, but found: {[p for p in paths if '.vscode' in p]}"
    )


def test_claude_directory_excluded_from_git_tracked_inventory(tmp_path: Path) -> None:
    """.claude/ is Claude Code's own project settings and subagent definitions
    (for itself, not for the target), not read by target runtime code -- found
    against a real independent repo whose committed .claude/settings.json hit
    the path-safety refusal exactly like an unrecognized virtualenv name would.
    """
    target = tmp_path / "repo"
    target.mkdir()
    (target / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git_init(target)
    claude_dir = target / ".claude" / "agents"
    claude_dir.mkdir(parents=True)
    (target / ".claude" / "settings.json").write_text("{}\n", encoding="utf-8")
    (claude_dir / "reviewer.md").write_text("# reviewer\n", encoding="utf-8")
    _git_track(target, "agent.py", ".claude/settings.json", ".claude/agents/reviewer.md")

    fileset = collect_source_file_set(target)
    paths = {item.path for item in fileset.files}

    assert "agent.py" in paths
    assert not any(".claude" in path for path in paths), (
        f".claude/ should be excluded, but found: {[p for p in paths if '.claude' in p]}"
    )


def test_unsafe_relevant_filename_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "local"
    target.mkdir()
    (target / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    (target / "weird name.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not a safe relative path"):
        collect_source_file_set(target)


def test_hidden_relevant_python_file_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "local"
    target.mkdir()
    (target / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    (target / ".hidden.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not a safe relative path"):
        collect_source_file_set(target)


def _write_virtualenv(root: Path, name: str) -> None:
    """A minimal but faithful virtualenv: the ``pyvenv.cfg`` marker every
    real venv-creation tool (venv, virtualenv, uv venv) writes, plus an
    installed-looking .py file the way a real site-packages tree would have
    one -- exactly the shape reported blocking replay manifest generation."""

    venv = root / name
    (venv / "lib" / "python3.12" / "site-packages" / "somepkg").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("version = 3.12.3\n", encoding="utf-8")
    (venv / "lib" / "python3.12" / "site-packages" / "somepkg" / "__init__.py").write_text(
        "X = 1\n", encoding="utf-8"
    )


@pytest.mark.parametrize("venv_name", [".test-venv", "test-venv", "myenv"])
def test_an_arbitrarily_named_virtualenv_is_excluded_in_local_mode(
    tmp_path: Path, venv_name: str
) -> None:
    target = tmp_path / "local"
    target.mkdir()
    (target / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _write_virtualenv(target, venv_name)

    inventory = collect_source_file_set(target)

    assert inventory.mode == "local_files"
    paths = {item.path for item in inventory.files}
    assert paths == {"agent.py"}


def test_an_arbitrarily_named_gitignored_virtualenv_is_excluded_in_git_mode(
    tmp_path: Path,
) -> None:
    """Reproduces the reported failure exactly: a virtualenv gitignored under
    a name other than .venv/venv (.test-venv here) used to be picked up by
    `git ls-files --others --ignored` as an "ignored-but-present" candidate,
    survive the fixed .venv/venv name-list exclusion, and then hit the
    dot-prefixed path-safety refusal -- turning a harmless local virtualenv
    into a hard replay-manifest failure instead of a silent, correct
    exclusion of what was never the target's own source to begin with."""

    target = tmp_path / "repo"
    target.mkdir()
    (target / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git_init(target)
    (target / ".gitignore").write_text(".test-venv/\n", encoding="utf-8")
    _write_virtualenv(target, ".test-venv")
    # Nothing is committed -- matching the reported scenario, where the
    # virtualenv was gitignored but the rest of the target was never staged.

    inventory = collect_source_file_set(target)

    paths = {item.path for item in inventory.files}
    assert "agent.py" in paths
    assert not any(".test-venv" in path for path in paths)
    root, config = load_config(target)
    snapshot = capture_source_snapshot(root, config)
    assert snapshot.git_revision is None
    assert snapshot.commit_bindable is False
    assert snapshot.file_set == inventory


def test_arbitrarily_named_venv_cannot_reclassify_commit_certified_paths(
    tmp_path: Path,
) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    (target / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git_init(target)
    (target / ".gitignore").write_text(".test-venv/\n", encoding="utf-8")
    _git_track(target, "agent.py", ".gitignore")
    _write_virtualenv(target, ".test-venv")

    with pytest.raises(ConfigurationError, match="not a safe relative path"):
        collect_source_file_set(target)


def test_a_dot_prefixed_directory_that_is_not_a_virtualenv_still_fails_closed(
    tmp_path: Path,
) -> None:
    """The fix is specific to genuine virtualenvs (pyvenv.cfg present) -- an
    unrelated dot-prefixed, gitignored directory with no such marker must
    still refuse rather than be silently excluded, preserving the existing
    source-integrity guarantee for everything that is not a virtualenv."""

    target = tmp_path / "repo"
    target.mkdir()
    (target / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git_init(target)
    (target / ".gitignore").write_text(".mystery/\n", encoding="utf-8")
    nested = target / ".mystery" / "deeply" / "nested"
    nested.mkdir(parents=True)
    (nested / "thing.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="not a safe relative path"):
        collect_source_file_set(target)
