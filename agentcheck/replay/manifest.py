"""Versioned replay manifests: pre-redaction execution recipes.

A replay manifest is not a disclosure artifact, not a frozen suite, and not
the SQLite index. It is written from in-memory scenarios before
``redact_artifact`` so fixture identity survives. Secret-shaped values are
refused at write time rather than stored in redacted form.
"""

from __future__ import annotations

import hmac
import json
import os
import re
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import Field, model_serializer, model_validator

from agentcheck import __version__
from agentcheck.artifacts import replace_private_file
from agentcheck.config import AgentCheckConfig, contained_path
from agentcheck.domain import (
    AgentSpec,
    ContractModel,
    Scenario,
    canonical_hash,
)
from agentcheck.errors import ConfigurationError
from agentcheck.privacy import redact_artifact, redact_log_text
from agentcheck.replay.fileset import SourceFileSet, omit_absent_file_set


REPLAY_MANIFEST_CONTRACT_VERSION: Literal["agentcheck.replay_manifest.v1"] = (
    "agentcheck.replay_manifest.v1"
)
REPLAY_SUBDIRECTORY = "replay"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$")
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_GIT_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")


class SpecBinding(ContractModel):
    """Inspected target surface that must still match at replay time."""

    spec_id: str = Field(min_length=1, max_length=200)
    adapter: Literal["openai_agents"]
    entrypoint: str = Field(min_length=1, max_length=500)
    policy_pack_ids: tuple[str, ...] = ()


class SourceBinding(ContractModel):
    """Source identity. Git revision is absent for non-git targets.

    ``file_set`` is additive. Slice 1 manifests omit it and keep the weaker
    entrypoint-only digest; new manifests include the bounded source inventory.
    Absence is not equivalent to a file-set match.
    """

    git_revision: str | None = Field(default=None, max_length=64)
    entrypoint_digest: str = Field(min_length=8, max_length=80)
    framework: str = Field(min_length=1, max_length=100)
    framework_version: str | None = Field(default=None, max_length=100)
    file_set: SourceFileSet | None = None

    @model_serializer(mode="wrap")
    def omit_legacy_file_set(self, serializer: Any) -> dict[str, Any]:
        return omit_absent_file_set(serializer(self))

    @model_validator(mode="after")
    def validate_revision_shape(self) -> "SourceBinding":
        if self.git_revision is not None and _GIT_REVISION_RE.fullmatch(
            self.git_revision
        ) is None:
            raise ValueError("git_revision must be a full hex object name")
        if not self.entrypoint_digest.startswith("sha256:"):
            raise ValueError("entrypoint_digest must be a sha256 digest")
        return self


class EnvironmentRequirements(ContractModel):
    """Allowlisted environment *names* only. Values are never serialized."""

    names: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_names(self) -> "EnvironmentRequirements":
        for name in self.names:
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
                raise ValueError(
                    "environment requirement names must be uppercase variable names"
                )
        if tuple(sorted(set(self.names))) != self.names:
            raise ValueError("environment requirement names must be unique and sorted")
        return self


class OmittedCase(ContractModel):
    """A lint-clean executed case that could not be serialized without secrets."""

    scenario_id: str = Field(min_length=1, max_length=200)
    fingerprint: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2_000)


class ReplayManifest(ContractModel):
    """Fingerprinted recipe for re-executing an AgentCheck evaluation."""

    schema_version: Literal["agentcheck.replay_manifest.v1"] = (
        REPLAY_MANIFEST_CONTRACT_VERSION
    )
    manifest_id: str = Field(default="", max_length=200)
    created_from_run_id: str = Field(min_length=1, max_length=120)
    agentcheck_version: str = Field(min_length=1, max_length=100)
    seed: int = Field(ge=0, le=2**63 - 1)
    spec_binding: SpecBinding
    source_binding: SourceBinding
    environment_requirements: EnvironmentRequirements = Field(
        default_factory=EnvironmentRequirements
    )
    cases: tuple[Scenario, ...] = Field(min_length=1)
    omitted: tuple[OmittedCase, ...] = ()
    fingerprint: str = ""

    @model_serializer(mode="wrap")
    def omit_idle_fields(self, serializer: Any) -> dict[str, Any]:
        data = serializer(self)
        if not data.get("omitted"):
            data.pop("omitted", None)
        return data

    def expected_fingerprint(self) -> str:
        return canonical_hash(
            self.model_dump(mode="json", exclude={"fingerprint", "manifest_id"})
        )

    def expected_manifest_id(self) -> str:
        return f"replay-{self.expected_fingerprint().split(':', 1)[1][:24]}"

    @model_validator(mode="after")
    def validate_identity_and_uniqueness(self) -> "ReplayManifest":
        if _SAFE_RUN_ID.fullmatch(self.created_from_run_id) is None:
            raise ValueError(
                "created_from_run_id must contain only letters, digits, "
                "underscores, or hyphens"
            )
        scenario_ids = [case.scenario_id for case in self.cases]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("replay manifest scenario IDs must be unique")
        fingerprints = [case.fingerprint for case in self.cases]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("replay manifest scenarios must be deduplicated")
        for case in self.cases:
            if case.fingerprint != case.expected_fingerprint():
                raise ValueError(
                    f"scenario {case.scenario_id} fingerprint does not match its contents"
                )
        expected = self.expected_fingerprint()
        if self.fingerprint and not _digest_equal(self.fingerprint, expected):
            raise ValueError("replay manifest fingerprint does not match its contents")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)
        expected_id = self.expected_manifest_id()
        if self.manifest_id and not _digest_equal(self.manifest_id, expected_id):
            raise ValueError("replay manifest ID does not match its contents")
        if not self.manifest_id:
            object.__setattr__(self, "manifest_id", expected_id)
        return self


def _digest_equal(left: str, right: str) -> bool:
    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    if len(left_bytes) != len(right_bytes):
        hmac.compare_digest(left_bytes, left_bytes)
        return False
    return hmac.compare_digest(left_bytes, right_bytes)


def secret_shaped_reason(scenario: Scenario) -> str | None:
    """Return a refusal reason if the scenario cannot be serialized safely."""

    dumped = json.loads(scenario.model_dump_json())
    try:
        _assert_no_secret_shaped_values(dumped)
    except ValueError as exc:
        return str(exc)
    redacted = redact_artifact(dumped)
    if redacted != dumped:
        return "scenario contains values that disclosure redaction would rewrite"
    return None


def _assert_no_secret_shaped_values(value: object, *, path: str = "$") -> None:
    if isinstance(value, str):
        if redact_log_text(value) != value:
            raise ValueError(f"credential-shaped content at {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_secret_shaped_values(item, path=f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_no_secret_shaped_values(item, path=f"{path}[{index}]")


def build_replay_manifest(
    *,
    run_id: str,
    seed: int,
    spec: AgentSpec,
    config: AgentCheckConfig,
    scenarios: Sequence[Scenario],
    git_revision: str | None,
    entrypoint_digest: str,
    policy_pack_ids: Sequence[str] = (),
    file_set: SourceFileSet | None = None,
) -> tuple[ReplayManifest | None, tuple[OmittedCase, ...]]:
    """Build a manifest from pre-redaction scenarios. Omits unscreenable cases."""

    kept: list[Scenario] = []
    omitted: list[OmittedCase] = []
    for scenario in scenarios:
        reason = secret_shaped_reason(scenario)
        if reason is not None:
            omitted.append(
                OmittedCase(
                    scenario_id=scenario.scenario_id,
                    fingerprint=scenario.fingerprint,
                    reason=reason,
                )
            )
            continue
        kept.append(scenario)
    if not kept:
        return None, tuple(omitted)
    environment_names = tuple(sorted(config.environment_allowlist))
    manifest = ReplayManifest(
        created_from_run_id=run_id,
        agentcheck_version=__version__,
        seed=seed,
        spec_binding=SpecBinding(
            spec_id=spec.spec_id,
            adapter=config.adapter,
            entrypoint=config.entrypoint,
            policy_pack_ids=tuple(sorted(set(policy_pack_ids))),
        ),
        source_binding=SourceBinding(
            git_revision=git_revision,
            entrypoint_digest=entrypoint_digest,
            framework=spec.identity.framework.value,
            framework_version=spec.identity.framework_version.value,
            file_set=file_set,
        ),
        environment_requirements=EnvironmentRequirements(names=environment_names),
        cases=tuple(kept),
        omitted=tuple(omitted),
    )
    return manifest, tuple(omitted)


def encode_replay_manifest(manifest: ReplayManifest) -> bytes:
    payload = (
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ConfigurationError(
            f"replay manifest exceeds the {MAX_MANIFEST_BYTES} byte size bound"
        )
    return payload


def replay_manifest_relative_path(config: AgentCheckConfig, run_id: str) -> str:
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ConfigurationError(
            "run ID must contain only letters, digits, underscores, or hyphens"
        )
    return f"{config.artifacts_directory}/{REPLAY_SUBDIRECTORY}/{run_id}.json"


def write_replay_manifest(
    root: Path,
    config: AgentCheckConfig,
    manifest: ReplayManifest,
) -> Path:
    relative = replay_manifest_relative_path(config, manifest.created_from_run_id)
    destination = contained_path(root, relative)
    payload = encode_replay_manifest(manifest)
    try:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o700)
        replace_private_file(destination, payload)
    except OSError as exc:
        raise ConfigurationError(f"unable to write replay manifest: {exc}") from exc
    return destination


__all__ = [
    "REPLAY_MANIFEST_CONTRACT_VERSION",
    "REPLAY_SUBDIRECTORY",
    "MAX_MANIFEST_BYTES",
    "EnvironmentRequirements",
    "OmittedCase",
    "ReplayManifest",
    "SourceBinding",
    "SourceFileSet",
    "SpecBinding",
    "build_replay_manifest",
    "encode_replay_manifest",
    "replay_manifest_relative_path",
    "secret_shaped_reason",
    "write_replay_manifest",
]
