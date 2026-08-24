"""Fail-closed source and configuration bindings for replay."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from agentcheck.config import AgentCheckConfig, resolve_entrypoint
from agentcheck.domain import AgentSpec
from agentcheck.errors import ConfigurationError
from agentcheck.identity import identity_mismatch_hint, spec_identity_matches

from .fileset import collect_source_file_set, describe_file_set_mismatch, git_command_env
from .manifest import ReplayManifest


def git_revision(root: Path) -> str | None:
    """Return HEAD when git is available. Walks to the enclosing repository."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=git_command_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def git_worktree_is_dirty(root: Path) -> bool:
    """True when git reports a change, or when cleanliness cannot be proved."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=git_command_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if completed.returncode != 0:
        return True
    return bool(completed.stdout.strip())


def entrypoint_digest(root: Path, entrypoint: str) -> str:
    """SHA-256 of the entrypoint file bytes. Does not follow a symlink."""

    source, _attribute = resolve_entrypoint(root, entrypoint)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ConfigurationError(
            f"unable to read entrypoint {source} for replay binding: {exc}"
        ) from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
    except OSError as exc:
        raise ConfigurationError(
            f"unable to read entrypoint {source} for replay binding: {exc}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def verify_replay_source_bindings(
    manifest: ReplayManifest,
    *,
    root: Path,
    config: AgentCheckConfig,
) -> None:
    """Refuse replay when source/config identity differs. Does not import the target."""

    binding = manifest.spec_binding
    if config.adapter != binding.adapter:
        raise ConfigurationError(
            f"replay manifest adapter {binding.adapter} does not match "
            f"configured adapter {config.adapter}"
        )
    if config.entrypoint != binding.entrypoint:
        raise ConfigurationError(
            f"replay manifest entrypoint {binding.entrypoint} does not match "
            f"configured entrypoint {config.entrypoint}"
        )

    source = manifest.source_binding
    live_digest = entrypoint_digest(root, config.entrypoint)
    if live_digest != source.entrypoint_digest:
        raise ConfigurationError(
            "entrypoint source digest does not match the replay manifest"
        )
    if source.git_revision is not None:
        live_revision = git_revision(root)
        if live_revision != source.git_revision:
            raise ConfigurationError(
                "git revision does not match the replay manifest"
            )
        if git_worktree_is_dirty(root):
            raise ConfigurationError(
                "target git worktree is dirty; replay refuses a dirty tree"
            )

    required = manifest.environment_requirements.names
    live_allowlist = tuple(sorted(config.environment_allowlist))
    if live_allowlist != required:
        raise ConfigurationError(
            "environment_allowlist does not match the replay manifest"
        )
    missing = [name for name in required if not (os.environ.get(name) or "").strip()]
    if missing:
        raise ConfigurationError(
            "replay requires environment variables that are not set: "
            + ", ".join(missing)
        )

    if source.file_set is None:
        return
    live_files = collect_source_file_set(root)
    reason = describe_file_set_mismatch(source.file_set, live_files)
    if reason:
        raise ConfigurationError(reason)


def verify_replay_spec_bindings(
    manifest: ReplayManifest,
    *,
    spec: AgentSpec,
    policy_pack_ids: tuple[str, ...],
) -> None:
    """Refuse replay when the inspected spec is not the bound surface."""

    binding = manifest.spec_binding
    if not spec_identity_matches(spec, binding.spec_id):
        raise ConfigurationError(
            f"replay manifest was bound to spec {binding.spec_id}, "
            f"but this target inspects as {spec.spec_id}"
            + identity_mismatch_hint(spec, binding.spec_id)
        )
    live_packs = tuple(sorted(set(policy_pack_ids)))
    bound_packs = tuple(sorted(set(binding.policy_pack_ids)))
    if live_packs != bound_packs:
        raise ConfigurationError(
            "declared policy packs do not match the replay manifest"
        )
    source = manifest.source_binding
    if spec.identity.framework.value != source.framework:
        raise ConfigurationError(
            "inspected framework does not match the replay manifest"
        )
    if spec.identity.framework_version.value != source.framework_version:
        raise ConfigurationError(
            "inspected framework version does not match the replay manifest"
        )


def verify_replay_bindings(
    manifest: ReplayManifest,
    *,
    root: Path,
    config: AgentCheckConfig,
    spec: AgentSpec,
    policy_pack_ids: tuple[str, ...],
) -> None:
    """Refuse replay when the live target is not the bound source."""

    verify_replay_source_bindings(manifest, root=root, config=config)
    verify_replay_spec_bindings(
        manifest, spec=spec, policy_pack_ids=policy_pack_ids
    )


__all__ = [
    "entrypoint_digest",
    "git_revision",
    "git_worktree_is_dirty",
    "verify_replay_bindings",
    "verify_replay_source_bindings",
    "verify_replay_spec_bindings",
]
