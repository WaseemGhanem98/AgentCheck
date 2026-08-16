"""Secure replay manifests: emit, load, and bind. Execution lives in application."""

from .bind import (
    entrypoint_digest,
    git_revision,
    git_worktree_is_dirty,
    verify_replay_bindings,
    verify_replay_source_bindings,
    verify_replay_spec_bindings,
)
from .load import load_replay_manifest, load_replay_manifest_path
from .manifest import (
    MAX_MANIFEST_BYTES,
    REPLAY_MANIFEST_CONTRACT_VERSION,
    EnvironmentRequirements,
    OmittedCase,
    ReplayManifest,
    SourceBinding,
    SpecBinding,
    build_replay_manifest,
    encode_replay_manifest,
    replay_manifest_relative_path,
    secret_shaped_reason,
    write_replay_manifest,
)
from .fileset import (
    SOURCE_FILE_SET_CONTRACT_VERSION,
    SourceFileEntry,
    SourceFileSet,
    collect_source_file_set,
)

__all__ = [
    "MAX_MANIFEST_BYTES",
    "REPLAY_MANIFEST_CONTRACT_VERSION",
    "SOURCE_FILE_SET_CONTRACT_VERSION",
    "EnvironmentRequirements",
    "OmittedCase",
    "ReplayManifest",
    "SourceBinding",
    "SourceFileEntry",
    "SourceFileSet",
    "SpecBinding",
    "build_replay_manifest",
    "collect_source_file_set",
    "encode_replay_manifest",
    "entrypoint_digest",
    "git_revision",
    "git_worktree_is_dirty",
    "load_replay_manifest",
    "load_replay_manifest_path",
    "replay_manifest_relative_path",
    "secret_shaped_reason",
    "verify_replay_bindings",
    "verify_replay_source_bindings",
    "verify_replay_spec_bindings",
    "write_replay_manifest",
]
