"""Fail-closed host-side contracts for the containment research spike.

Only the trusted parent/provider may construct these records. Target output is
bounded diagnostic material and cannot satisfy observation or postconditions.
The frozen catalog and semantic ground truth own denominators and meanings;
callers cannot relabel cases or paths to improve a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from agentcheck.domain import canonical_hash
from agentcheck.replay.fileset import (
    SOURCE_FILE_SET_CONTRACT_VERSION,
    SourceFileSet,
)


MEASUREMENT_SCHEMA_VERSION = "agentcheck.environment_containment.measurement.v1"
CATALOG_SCHEMA_VERSION = "agentcheck.environment_containment.catalog.v1"
GROUND_TRUTH_SCHEMA_VERSION = (
    "agentcheck.environment_containment.semantic_ground_truth.v1"
)
FROZEN_CATALOG_ID = "agentcheck-environment-containment-v1"
FROZEN_CATALOG_VERSION = "1.0.0"
FROZEN_CATALOG_SHA256 = (
    "2ca3e50abd294f63e85b85a843e8107f41bf6fe5b21cab33963bd17fcf8d3067"
)
FROZEN_GROUND_TRUTH_SHA256 = (
    "3f0e26bb7856712c4c3b565ac923c0ece130153bb2d5f2771840540559068572"
)
FROZEN_CATALOG_DEFINITION_FINGERPRINT = (
    "sha256:598edf1c4e56878a48167d8b247c8669ff39e2a20701902bed038c40da5158ab"
)
FROZEN_GROUND_TRUTH_DEFINITION_FINGERPRINT = (
    "sha256:ad2ad68c64482e9d5fc53f3bd1c2decb4f39a6d102fc477fefc63c5211f6ed06"
)
FROZEN_HOSTILE_CASES = 27
FROZEN_HOSTILE_SUBATTEMPTS = 40
FROZEN_CONTROL_CASES = 2
FROZEN_CONTROL_SUBATTEMPTS = 2
FROZEN_BEHAVIORAL_PATHS = 9
_RAW_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCE_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")


class ContractViolation(ValueError):
    """Raised when a record would overstate the evidence supplied to it."""


class CaseType(str, Enum):
    HOSTILE = "hostile"
    CONTROL = "control"


class AttackOutcome(str, Enum):
    UNEXECUTED = "unexecuted"
    BLOCKED = "blocked"
    CONTAINED = "contained"
    CONTROL_SUCCEEDED = "control_succeeded"
    CONTROL_FAILED = "control_failed"
    ESCAPED = "escaped"
    INCONCLUSIVE = "inconclusive"


class SubattemptDisposition(str, Enum):
    BLOCKED = "blocked"
    CONTAINED = "contained"
    PROVIDER_LIMIT_ENFORCED = "provider_limit_enforced"
    CONTROL_SUCCEEDED = "control_succeeded"


class TrustedEvidenceKind(str, Enum):
    """Evidence produced outside target-controlled stdout and state."""

    PARENT_DISPATCH = "parent_dispatch"
    PARENT_SENTINEL = "parent_sentinel"
    PARENT_EFFECT_RECEIPT = "parent_effect_receipt"
    PROVIDER_TRACE = "provider_trace"
    CONTROLLED_ENDPOINT_RECEIPT = "controlled_endpoint_receipt"
    CONTROLLED_ENDPOINT_POSTCONDITION = "controlled_endpoint_postcondition"
    NETWORK_POLICY_POSTCONDITION = "network_policy_postcondition"
    CONTAINER_EXIT_STATUS = "container_exit_status"
    PROCESS_LIFECYCLE_TRACE = "process_lifecycle_trace"
    PROCESS_TEARDOWN_POSTCONDITION = "process_teardown_postcondition"
    RESOURCE_CONTROLLER = "resource_controller"
    SOURCE_BINDING = "source_binding"
    TOOL_GATEWAY_EVENT = "tool_gateway_event"
    CANONICAL_BEHAVIOR_EVIDENCE = "canonical_behavior_evidence"


class DiagnosticChannel(str, Enum):
    STDOUT = "stdout"
    STDERR = "stderr"


class SemanticStatus(str, Enum):
    UNDERSTOOD = "UNDERSTOOD"
    UNKNOWN = "UNKNOWN"


class PolicyResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceScope(str, Enum):
    """This schema records one trial, never a stochastic aggregate."""

    SINGLE_TRIAL = "single_trial"


class InfrastructureStatus(str, Enum):
    """Typed completion state for the trusted execution/observation boundary."""

    NOT_RUN = "not_run"
    COMPLETE = "complete"
    INFRA_ERROR = "infra_error"


class InfrastructureFailureStage(str, Enum):
    """The trusted boundary stage that prevented complete trial evidence."""

    SETUP = "setup"
    TARGET_CONFIGURATION = "target_configuration"
    TARGET_EXECUTION = "target_execution"
    OBSERVER = "observer"
    TEARDOWN = "teardown"


class EvidenceSubjectType(str, Enum):
    """The exact measurement subject one receipt is allowed to support."""

    SUBATTEMPT = "subattempt"
    POSTCONDITION = "postcondition"
    EFFECT = "effect"
    ACTION = "action"
    PATH = "path"
    TRIAL_COMPLETION = "trial_completion"


class ConfirmationState(str, Enum):
    OBSERVED = "observed"
    CONFIRMED_ABSENT = "confirmed_absent"


class EffectKind(str, Enum):
    EXTERNAL_CANARY_CHANGE = "external_canary_change"
    CONTROLLED_MUTATION_REACHED = "controlled_mutation_reached"
    REAL_EXTERNAL_MUTATION = "real_external_mutation"
    HOST_SECRET_LEAKAGE = "host_secret_leakage"
    PERSISTENT_DESCENDANT = "persistent_descendant"
    PROVIDER_LIMIT_ENFORCED = "provider_limit_enforced"


_ATTEMPT_OBSERVERS = frozenset(
    {
        TrustedEvidenceKind.PARENT_EFFECT_RECEIPT,
        TrustedEvidenceKind.PROVIDER_TRACE,
        TrustedEvidenceKind.CONTROLLED_ENDPOINT_RECEIPT,
        TrustedEvidenceKind.PROCESS_LIFECYCLE_TRACE,
        TrustedEvidenceKind.RESOURCE_CONTROLLER,
    }
)
_POSTCONDITION_EVIDENCE = frozenset(
    {
        TrustedEvidenceKind.PARENT_SENTINEL,
        TrustedEvidenceKind.PARENT_EFFECT_RECEIPT,
        TrustedEvidenceKind.CONTROLLED_ENDPOINT_POSTCONDITION,
        TrustedEvidenceKind.NETWORK_POLICY_POSTCONDITION,
        TrustedEvidenceKind.PROCESS_TEARDOWN_POSTCONDITION,
        TrustedEvidenceKind.RESOURCE_CONTROLLER,
        TrustedEvidenceKind.SOURCE_BINDING,
    }
)
_EFFECT_EVIDENCE = _ATTEMPT_OBSERVERS | _POSTCONDITION_EVIDENCE
_ACTION_OBSERVERS = frozenset(
    {
        TrustedEvidenceKind.TOOL_GATEWAY_EVENT,
    }
)
_PATH_OBSERVERS = frozenset(
    {
        TrustedEvidenceKind.CANONICAL_BEHAVIOR_EVIDENCE,
    }
)
_FAVORABLE_HOSTILE_OUTCOMES = frozenset(
    {AttackOutcome.BLOCKED, AttackOutcome.CONTAINED}
)


def _require_nonempty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ContractViolation(f"{field_name} must not be empty")


def _require_raw_digest(value: str, field_name: str) -> None:
    if _RAW_SHA256.fullmatch(value) is None:
        raise ContractViolation(f"{field_name} must be a lowercase SHA-256 digest")


def _parse_string_tuple(
    value: Any, field_name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ContractViolation(f"{field_name} must be a string list")
    result = tuple(value)
    if not allow_empty and not result:
        raise ContractViolation(f"{field_name} must not be empty")
    if len(set(result)) != len(result):
        raise ContractViolation(f"{field_name} must contain unique values")
    return result


def canonical_json(value: Mapping[str, Any]) -> str:
    """Canonicalize synthetic action arguments for exact oracle comparison."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractViolation("action arguments must be JSON-compatible") from exc
    if len(rendered) > 4_096:
        raise ContractViolation("canonical action arguments exceed 4,096 characters")
    return rendered


def _parse_source_file_set(value: Any, field_name: str) -> SourceFileSet:
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ContractViolation(f"{field_name} is malformed")
    normalized = dict(value)
    normalized["files"] = tuple(value["files"])
    try:
        return SourceFileSet.model_validate(normalized)
    except ValueError as exc:
        raise ContractViolation(f"{field_name} is invalid") from exc


_BOUND_EVIDENCE_TOKEN = object()


@dataclass(frozen=True, init=False)
class EvidenceReference:
    kind: TrustedEvidenceKind
    reference: str
    artifact_digest: str
    artifact_size_bytes: int
    provenance_fingerprint: str
    run_id: str
    trial_id: str
    environment_instance_id: str
    subject_type: EvidenceSubjectType
    scope_id: str
    subject_id: str
    sequence_index: int
    receipt_id: str = field(init=False)

    def __init__(
        self,
        *,
        kind: TrustedEvidenceKind,
        reference: str,
        artifact_digest: str,
        artifact_size_bytes: int,
        provenance_fingerprint: str,
        run_id: str,
        trial_id: str,
        environment_instance_id: str,
        subject_type: EvidenceSubjectType,
        scope_id: str,
        subject_id: str,
        sequence_index: int,
        _binding_token: object | None = None,
    ) -> None:
        if _binding_token is not _BOUND_EVIDENCE_TOKEN:
            raise ContractViolation(
                "evidence receipts must be created by hashing an artifact with "
                "bind_evidence_bytes or bind_evidence_file"
            )
        for name, value in (
            ("reference", reference),
            ("run_id", run_id),
            ("trial_id", trial_id),
            ("environment_instance_id", environment_instance_id),
            ("scope_id", scope_id),
            ("subject_id", subject_id),
        ):
            _require_nonempty(value, name)
        if len(reference) > 2_000:
            raise ContractViolation("evidence reference exceeds 2,000 characters")
        reference_path = PurePosixPath(reference)
        if (
            reference_path.is_absolute()
            or ".." in reference_path.parts
            or str(reference_path) != reference
        ):
            raise ContractViolation(
                "evidence reference must be a normalized target-independent relative path"
            )
        _require_raw_digest(artifact_digest, "evidence artifact digest")
        if artifact_size_bytes < 0:
            raise ContractViolation("evidence artifact size cannot be negative")
        if _SOURCE_FINGERPRINT.fullmatch(provenance_fingerprint) is None:
            raise ContractViolation("evidence provenance fingerprint is malformed")
        if sequence_index < 0:
            raise ContractViolation("evidence sequence index cannot be negative")
        values = {
            "kind": kind,
            "reference": reference,
            "artifact_digest": artifact_digest,
            "artifact_size_bytes": artifact_size_bytes,
            "provenance_fingerprint": provenance_fingerprint,
            "run_id": run_id,
            "trial_id": trial_id,
            "environment_instance_id": environment_instance_id,
            "subject_type": subject_type,
            "scope_id": scope_id,
            "subject_id": subject_id,
            "sequence_index": sequence_index,
        }
        for name, attribute_value in values.items():
            object.__setattr__(self, name, attribute_value)
        object.__setattr__(
            self,
            "receipt_id",
            canonical_hash(
                {
                    "kind": kind.value,
                    "reference": reference,
                    "artifact_digest": artifact_digest,
                    "artifact_size_bytes": artifact_size_bytes,
                    "provenance_fingerprint": provenance_fingerprint,
                    "run_id": run_id,
                    "trial_id": trial_id,
                    "environment_instance_id": environment_instance_id,
                    "subject_type": subject_type.value,
                    "scope_id": scope_id,
                    "subject_id": subject_id,
                    "sequence_index": sequence_index,
                }
            ),
        )


@dataclass(frozen=True)
class UntrustedDiagnostic:
    channel: DiagnosticChannel
    text: str

    def __post_init__(self) -> None:
        if len(self.text) > 4_096:
            raise ContractViolation("target diagnostic exceeds 4,096 characters")


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    trial_id: str
    environment_instance_ids: tuple[str, ...]
    repetition_index: int
    target_source_digest: str
    target_source_binding: str
    catalog_digest: str
    semantic_ground_truth_digest: str
    environment_provider: str
    provider_version: str
    containment_profile: str
    behavioral_execution_allowed: bool
    infrastructure_status: InfrastructureStatus = InfrastructureStatus.NOT_RUN
    containment_tier: str | None = None
    containment_status: str | None = None
    evidence_scope: EvidenceScope = EvidenceScope.SINGLE_TRIAL
    schema_version: str = MEASUREMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEASUREMENT_SCHEMA_VERSION:
            raise ContractViolation("unsupported measurement schema version")
        if self.evidence_scope is not EvidenceScope.SINGLE_TRIAL:
            raise ContractViolation(
                "this schema records one trial; aggregate evidence needs a separate record"
            )
        for name, value in (
            ("run_id", self.run_id),
            ("trial_id", self.trial_id),
            ("target_source_binding", self.target_source_binding),
            ("environment_provider", self.environment_provider),
            ("provider_version", self.provider_version),
            ("containment_profile", self.containment_profile),
        ):
            _require_nonempty(value, name)
        if not self.environment_instance_ids or len(set(self.environment_instance_ids)) != len(
            self.environment_instance_ids
        ):
            raise ContractViolation(
                "environment_instance_ids must be non-empty and unique"
            )
        for environment_instance_id in self.environment_instance_ids:
            _require_nonempty(environment_instance_id, "environment_instance_id")
        if self.repetition_index < 1:
            raise ContractViolation("repetition_index must start at one")
        if _SOURCE_FINGERPRINT.fullmatch(self.target_source_digest) is None:
            raise ContractViolation(
                "target_source_digest must be an AgentCheck SourceFileSet fingerprint"
            )
        if self.target_source_binding != SOURCE_FILE_SET_CONTRACT_VERSION:
            raise ContractViolation(
                "target_source_binding must use agentcheck.source_file_set.v1"
            )
        _require_raw_digest(self.catalog_digest, "catalog_digest")
        _require_raw_digest(
            self.semantic_ground_truth_digest,
            "semantic_ground_truth_digest",
        )
        for optional_name, optional_value in (
            ("containment_tier", self.containment_tier),
            ("containment_status", self.containment_status),
        ):
            if optional_value is not None:
                _require_nonempty(optional_value, optional_name)

    @property
    def fingerprint(self) -> str:
        """Bind receipts to every execution/provenance field in this trial."""

        return canonical_hash(
            {
                "schema_version": self.schema_version,
                "evidence_scope": self.evidence_scope.value,
                "run_id": self.run_id,
                "trial_id": self.trial_id,
                "environment_instance_ids": list(self.environment_instance_ids),
                "repetition_index": self.repetition_index,
                "target_source_digest": self.target_source_digest,
                "target_source_binding": self.target_source_binding,
                "catalog_digest": self.catalog_digest,
                "semantic_ground_truth_digest": self.semantic_ground_truth_digest,
                "environment_provider": self.environment_provider,
                "provider_version": self.provider_version,
                "containment_profile": self.containment_profile,
                "behavioral_execution_allowed": self.behavioral_execution_allowed,
                "infrastructure_status": self.infrastructure_status.value,
                "containment_tier": self.containment_tier,
                "containment_status": self.containment_status,
            }
        )


def _bind_evidence_digest(
    *,
    kind: TrustedEvidenceKind,
    reference: str,
    artifact_digest: str,
    artifact_size_bytes: int,
    provenance: RunProvenance,
    environment_instance_id: str,
    subject_type: EvidenceSubjectType,
    scope_id: str,
    subject_id: str,
    sequence_index: int,
) -> EvidenceReference:
    if environment_instance_id not in provenance.environment_instance_ids:
        raise ContractViolation(
            "evidence environment is absent from the run provenance"
        )
    return EvidenceReference(
        kind=kind,
        reference=reference,
        artifact_digest=artifact_digest,
        artifact_size_bytes=artifact_size_bytes,
        provenance_fingerprint=provenance.fingerprint,
        run_id=provenance.run_id,
        trial_id=provenance.trial_id,
        environment_instance_id=environment_instance_id,
        subject_type=subject_type,
        scope_id=scope_id,
        subject_id=subject_id,
        sequence_index=sequence_index,
        _binding_token=_BOUND_EVIDENCE_TOKEN,
    )


def bind_evidence_bytes(
    *,
    kind: TrustedEvidenceKind,
    reference: str,
    artifact: bytes,
    provenance: RunProvenance,
    environment_instance_id: str,
    subject_type: EvidenceSubjectType,
    scope_id: str,
    subject_id: str,
    sequence_index: int = 0,
) -> EvidenceReference:
    """Content-address one host/provider artifact and bind its exact subject.

    This proves artifact integrity and internal run/subject binding. It does not
    authenticate who supplied ``artifact``; producer trust must come from the
    maintained provider boundary, outside target reach.
    """

    if not isinstance(artifact, bytes):
        raise ContractViolation("evidence artifact must be bytes")
    return _bind_evidence_digest(
        kind=kind,
        reference=reference,
        artifact_digest=hashlib.sha256(artifact).hexdigest(),
        artifact_size_bytes=len(artifact),
        provenance=provenance,
        environment_instance_id=environment_instance_id,
        subject_type=subject_type,
        scope_id=scope_id,
        subject_id=subject_id,
        sequence_index=sequence_index,
    )


def _hash_stable_evidence_file(artifact_path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(artifact_path, flags)
    except OSError as exc:
        raise ContractViolation("evidence artifact cannot be opened safely") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or size != after.st_size:
        raise ContractViolation("evidence artifact changed while it was hashed")
    return digest.hexdigest(), size


def bind_evidence_file(
    *,
    kind: TrustedEvidenceKind,
    reference: str,
    artifact_path: Path,
    provenance: RunProvenance,
    environment_instance_id: str,
    subject_type: EvidenceSubjectType,
    scope_id: str,
    subject_id: str,
    sequence_index: int = 0,
) -> EvidenceReference:
    """Hash a stable, non-symlink host artifact before creating its receipt."""

    artifact_digest, artifact_size_bytes = _hash_stable_evidence_file(artifact_path)
    return _bind_evidence_digest(
        kind=kind,
        reference=reference,
        artifact_digest=artifact_digest,
        artifact_size_bytes=artifact_size_bytes,
        provenance=provenance,
        environment_instance_id=environment_instance_id,
        subject_type=subject_type,
        scope_id=scope_id,
        subject_id=subject_id,
        sequence_index=sequence_index,
    )


def verify_evidence_file(
    reference: EvidenceReference,
    artifact_path: Path,
) -> None:
    """Re-attest that persisted artifact bytes still match a bound receipt."""

    artifact_digest, artifact_size_bytes = _hash_stable_evidence_file(artifact_path)
    if (
        artifact_digest != reference.artifact_digest
        or artifact_size_bytes != reference.artifact_size_bytes
    ):
        raise ContractViolation("evidence artifact bytes do not match their receipt")


def _validate_evidence_bindings(
    references: tuple[EvidenceReference, ...],
    *,
    provenance: RunProvenance,
    subject_type: EvidenceSubjectType,
    scope_id: str,
    subject_id: str,
    sequence_index: int,
    environment_instance_ids: tuple[str, ...],
) -> None:
    for reference in references:
        if (
            reference.provenance_fingerprint != provenance.fingerprint
            or reference.run_id != provenance.run_id
            or reference.trial_id != provenance.trial_id
        ):
            raise ContractViolation(
                f"evidence receipt is bound to the wrong run for {scope_id}/{subject_id}"
            )
        if reference.environment_instance_id not in environment_instance_ids:
            raise ContractViolation(
                f"evidence receipt is bound to the wrong environment for "
                f"{scope_id}/{subject_id}"
            )
        if (
            reference.subject_type is not subject_type
            or reference.scope_id != scope_id
            or reference.subject_id != subject_id
            or reference.sequence_index != sequence_index
        ):
            raise ContractViolation(
                f"evidence receipt is bound to the wrong subject for "
                f"{scope_id}/{subject_id}"
            )


def _require_unique_receipt_uses(references: tuple[EvidenceReference, ...]) -> None:
    receipt_ids = tuple(reference.receipt_id for reference in references)
    if len(receipt_ids) != len(set(receipt_ids)):
        raise ContractViolation(
            "one evidence receipt cannot be reused across measurement subjects"
        )


def _container_exit_status_artifact(exit_code: int) -> bytes:
    return canonical_json({"container_exit_code": exit_code}).encode("utf-8")


def bind_container_exit_status(
    *,
    reference: str,
    exit_code: int,
    provenance: RunProvenance,
    environment_instance_id: str,
) -> EvidenceReference:
    """Bind the host-observed process exit to this exact run and trial.

    The canonical artifact binds the typed integer to the receipt digest. As
    with every evidence binder here, producer authentication remains the
    maintained provider's responsibility and is not claimed by this schema.
    """

    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise ContractViolation("container exit code must be an integer")
    return bind_evidence_bytes(
        kind=TrustedEvidenceKind.CONTAINER_EXIT_STATUS,
        reference=reference,
        artifact=_container_exit_status_artifact(exit_code),
        provenance=provenance,
        environment_instance_id=environment_instance_id,
        subject_type=EvidenceSubjectType.TRIAL_COMPLETION,
        scope_id=provenance.run_id,
        subject_id=provenance.trial_id,
        sequence_index=0,
    )


@dataclass(frozen=True)
class TrialCompletion:
    """Typed host-side completion facts for exactly one frozen trial."""

    status: InfrastructureStatus
    container_exit_code: int | None
    evidence_references: tuple[EvidenceReference, ...] = ()
    failure_stage: InfrastructureFailureStage | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, InfrastructureStatus):
            raise ContractViolation("trial completion status must be typed")
        if self.failure_stage is not None and not isinstance(
            self.failure_stage, InfrastructureFailureStage
        ):
            raise ContractViolation("trial failure stage must be typed")
        if self.container_exit_code is not None and (
            isinstance(self.container_exit_code, bool)
            or not isinstance(self.container_exit_code, int)
        ):
            raise ContractViolation("container exit code must be an integer")
        if any(not item.strip() for item in self.limitations):
            raise ContractViolation("trial-completion limitations must be non-empty")
        if any(
            item.kind is not TrustedEvidenceKind.CONTAINER_EXIT_STATUS
            for item in self.evidence_references
        ):
            raise ContractViolation(
                "trial completion accepts only container-exit-status evidence"
            )

        if self.status is InfrastructureStatus.NOT_RUN:
            if (
                self.container_exit_code is not None
                or self.evidence_references
                or self.failure_stage is not None
            ):
                raise ContractViolation(
                    "a not-run trial cannot claim completion or failure evidence"
                )
            return

        if self.status is InfrastructureStatus.COMPLETE:
            if self.container_exit_code != 0:
                raise ContractViolation(
                    "a complete trial requires a successful container exit"
                )
            if self.failure_stage is not None:
                raise ContractViolation("a complete trial cannot name a failure stage")
        else:
            if self.failure_stage is None:
                raise ContractViolation("an infrastructure error needs a typed failure stage")
            if self.failure_stage in {
                InfrastructureFailureStage.TARGET_CONFIGURATION,
                InfrastructureFailureStage.TARGET_EXECUTION,
            } and (self.container_exit_code is None or self.container_exit_code == 0):
                raise ContractViolation(
                    "a target configuration/execution error needs a nonzero exit"
                )

        if self.container_exit_code is None:
            if self.evidence_references:
                raise ContractViolation(
                    "a trial without a process exit cannot carry exit-status evidence"
                )
        elif len(self.evidence_references) != 1:
            raise ContractViolation(
                "a recorded container exit requires exactly one bound exit receipt"
            )


def _validate_trial_completion(
    completion: TrialCompletion,
    *,
    provenance: RunProvenance,
) -> None:
    if completion.status is not provenance.infrastructure_status:
        raise ContractViolation(
            "trial completion status does not match run provenance"
        )
    _validate_evidence_bindings(
        completion.evidence_references,
        provenance=provenance,
        subject_type=EvidenceSubjectType.TRIAL_COMPLETION,
        scope_id=provenance.run_id,
        subject_id=provenance.trial_id,
        sequence_index=0,
        environment_instance_ids=provenance.environment_instance_ids,
    )
    if completion.container_exit_code is None:
        return
    expected = _container_exit_status_artifact(completion.container_exit_code)
    reference = completion.evidence_references[0]
    if (
        reference.artifact_digest != hashlib.sha256(expected).hexdigest()
        or reference.artifact_size_bytes != len(expected)
    ):
        raise ContractViolation(
            "container exit receipt does not match the typed exit status"
        )


@dataclass(frozen=True)
class StaticDiscovery:
    findings_discovered_at_least: int
    exhaustive: bool = False

    def __post_init__(self) -> None:
        if self.findings_discovered_at_least < 0:
            raise ContractViolation("static discovery count cannot be negative")
        if self.exhaustive:
            raise ContractViolation("static discovery is never exhaustive")

    @property
    def claim(self) -> str:
        return (
            f"at least {self.findings_discovered_at_least} findings "
            "discovered statically"
        )


@dataclass(frozen=True)
class CatalogEffectDefinition:
    kind: EffectKind
    expected_value: bool
    required_evidence_kinds: frozenset[TrustedEvidenceKind]

    def __post_init__(self) -> None:
        if not self.required_evidence_kinds:
            raise ContractViolation("effect checks need required trusted evidence")
        if not self.required_evidence_kinds <= _EFFECT_EVIDENCE:
            raise ContractViolation("effect check uses a non-effect evidence kind")


@dataclass(frozen=True)
class CatalogCaseDefinition:
    attack_id: str
    case_type: CaseType
    category: str
    technique: str
    security_property_under_test: str
    expected_outcome: str
    observability_requirement: str
    subattempt_ids: tuple[str, ...]
    allowed_subattempt_dispositions: tuple[frozenset[SubattemptDisposition], ...]
    attempt_evidence_kinds: frozenset[TrustedEvidenceKind]
    mutation_sentinel: str
    required_evidence_kinds: frozenset[TrustedEvidenceKind]
    effect_checks: tuple[CatalogEffectDefinition, ...]
    limitations: tuple[str, ...]
    required_environment_instances: int = 1

    def __post_init__(self) -> None:
        _require_nonempty(self.attack_id, "attack_id")
        _require_nonempty(self.category, "category")
        _require_nonempty(self.technique, "technique")
        _require_nonempty(
            self.security_property_under_test, "security_property_under_test"
        )
        _require_nonempty(self.expected_outcome, "expected_outcome")
        _require_nonempty(
            self.observability_requirement, "observability_requirement"
        )
        _require_nonempty(self.mutation_sentinel, "mutation_sentinel")
        if not self.subattempt_ids or len(set(self.subattempt_ids)) != len(
            self.subattempt_ids
        ):
            raise ContractViolation("catalog subattempt IDs must be non-empty and unique")
        if len(self.allowed_subattempt_dispositions) != len(self.subattempt_ids):
            raise ContractViolation(
                "catalog subattempt dispositions must align with subattempt IDs"
            )
        if any(not item for item in self.allowed_subattempt_dispositions):
            raise ContractViolation(
                "catalog subattempt dispositions must be non-empty"
            )
        if not self.attempt_evidence_kinds:
            raise ContractViolation("catalog cases need trusted attempt evidence kinds")
        if not self.attempt_evidence_kinds <= _ATTEMPT_OBSERVERS:
            raise ContractViolation("catalog attempt evidence kind is not an observer")
        if not self.required_evidence_kinds:
            raise ContractViolation("catalog cases need required trusted evidence kinds")
        if self.required_evidence_kinds.isdisjoint(_POSTCONDITION_EVIDENCE):
            raise ContractViolation("catalog cases need trusted postcondition evidence")
        non_postcondition = self.required_evidence_kinds - _POSTCONDITION_EVIDENCE
        if not non_postcondition <= self.attempt_evidence_kinds:
            raise ContractViolation(
                "non-postcondition required evidence must be required per subattempt"
            )
        effect_kinds = tuple(item.kind for item in self.effect_checks)
        if not effect_kinds or len(effect_kinds) != len(set(effect_kinds)):
            raise ContractViolation("catalog effect checks must be non-empty and unique")
        if self.required_environment_instances < 1:
            raise ContractViolation("required_environment_instances must be positive")
        if not self.limitations or not all(item.strip() for item in self.limitations):
            raise ContractViolation("catalog cases need non-empty limitations")


def _catalog_definition_fingerprint(
    *,
    source_digest: str,
    catalog_id: str,
    catalog_version: str,
    target_file_set: SourceFileSet,
    cases: tuple[CatalogCaseDefinition, ...],
) -> str:
    """Bind every parsed security-relevant catalog field to frozen bytes."""

    return canonical_hash(
        {
            "source_digest": source_digest,
            "catalog_id": catalog_id,
            "catalog_version": catalog_version,
            "target_file_set": target_file_set.model_dump(mode="json"),
            "cases": [
                {
                    "attack_id": item.attack_id,
                    "case_type": item.case_type.value,
                    "category": item.category,
                    "technique": item.technique,
                    "security_property_under_test": (
                        item.security_property_under_test
                    ),
                    "expected_outcome": item.expected_outcome,
                    "observability_requirement": item.observability_requirement,
                    "subattempt_ids": list(item.subattempt_ids),
                    "allowed_subattempt_dispositions": [
                        sorted(disposition.value for disposition in dispositions)
                        for dispositions in item.allowed_subattempt_dispositions
                    ],
                    "attempt_evidence_kinds": sorted(
                        kind.value for kind in item.attempt_evidence_kinds
                    ),
                    "mutation_sentinel": item.mutation_sentinel,
                    "required_evidence_kinds": sorted(
                        kind.value for kind in item.required_evidence_kinds
                    ),
                    "effect_checks": [
                        {
                            "kind": effect.kind.value,
                            "expected_value": effect.expected_value,
                            "required_evidence_kinds": sorted(
                                kind.value
                                for kind in effect.required_evidence_kinds
                            ),
                        }
                        for effect in item.effect_checks
                    ],
                    "limitations": list(item.limitations),
                    "required_environment_instances": (
                        item.required_environment_instances
                    ),
                }
                for item in cases
            ],
        }
    )


@dataclass(frozen=True)
class FrozenCatalog:
    source_digest: str
    catalog_id: str
    catalog_version: str
    target_file_set: SourceFileSet
    cases: tuple[CatalogCaseDefinition, ...]

    def __post_init__(self) -> None:
        _require_raw_digest(self.source_digest, "catalog source digest")
        if self.catalog_id != FROZEN_CATALOG_ID:
            raise ContractViolation("unexpected frozen catalog ID")
        if self.catalog_version != FROZEN_CATALOG_VERSION:
            raise ContractViolation("unexpected frozen catalog version")
        if self.target_file_set.mode != "local_files" or not self.target_file_set.complete:
            raise ContractViolation(
                "frozen hostile target binding must be a complete standalone fileset"
            )
        ids = tuple(item.attack_id for item in self.cases)
        if len(ids) != len(set(ids)):
            raise ContractViolation("frozen catalog case IDs must be unique")
        hostile = sum(item.case_type is CaseType.HOSTILE for item in self.cases)
        controls = sum(item.case_type is CaseType.CONTROL for item in self.cases)
        if (hostile, controls) != (FROZEN_HOSTILE_CASES, FROZEN_CONTROL_CASES):
            raise ContractViolation("frozen catalog denominator is not 27 hostile + 2 controls")
        hostile_subattempts = sum(
            len(item.subattempt_ids)
            for item in self.cases
            if item.case_type is CaseType.HOSTILE
        )
        control_subattempts = sum(
            len(item.subattempt_ids)
            for item in self.cases
            if item.case_type is CaseType.CONTROL
        )
        if (hostile_subattempts, control_subattempts) != (
            FROZEN_HOSTILE_SUBATTEMPTS,
            FROZEN_CONTROL_SUBATTEMPTS,
        ):
            raise ContractViolation(
                "frozen operation denominator is not 40 hostile + 2 controls"
            )
        all_subattempt_ids = tuple(
            subattempt_id
            for item in self.cases
            for subattempt_id in item.subattempt_ids
        )
        if len(all_subattempt_ids) != len(set(all_subattempt_ids)):
            raise ContractViolation(
                "frozen subattempt IDs must be globally unique"
            )
        definition_fingerprint = _catalog_definition_fingerprint(
            source_digest=self.source_digest,
            catalog_id=self.catalog_id,
            catalog_version=self.catalog_version,
            target_file_set=self.target_file_set,
            cases=self.cases,
        )
        if definition_fingerprint != FROZEN_CATALOG_DEFINITION_FINGERPRINT:
            raise ContractViolation(
                "parsed catalog definitions do not match the frozen contract "
                f"({definition_fingerprint})"
            )


@dataclass(frozen=True)
class SubattemptMeasurement:
    subattempt_id: str
    executed: bool = False
    observed: bool = False
    disposition: SubattemptDisposition | None = None
    evidence_references: tuple[EvidenceReference, ...] = ()
    untrusted_diagnostics: tuple[UntrustedDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.subattempt_id, "subattempt_id")
        if not self.executed:
            if self.observed or self.disposition is not None or self.evidence_references:
                raise ContractViolation(
                    "an unexecuted subattempt cannot carry trusted execution evidence"
                )
            return
        evidence_kinds = {item.kind for item in self.evidence_references}
        if self.observed:
            if evidence_kinds.isdisjoint(_ATTEMPT_OBSERVERS):
                raise ContractViolation(
                    "observed subattempts require an independent parent/provider observer"
                )
            if self.disposition is None:
                raise ContractViolation(
                    "observed subattempts require a trusted security disposition"
                )
        elif self.disposition is not None:
            raise ContractViolation(
                "an unobserved subattempt cannot claim a security disposition"
            )


@dataclass(frozen=True)
class TrustedPostcondition:
    sentinel_id: str
    satisfied: bool | None = None
    evidence_references: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.sentinel_id, "sentinel_id")
        if self.satisfied is None:
            if self.evidence_references:
                raise ContractViolation(
                    "an unchecked postcondition cannot carry trusted evidence"
                )
            return
        kinds = {item.kind for item in self.evidence_references}
        if not self.evidence_references or kinds.isdisjoint(_POSTCONDITION_EVIDENCE):
            raise ContractViolation(
                "a checked postcondition needs independent sentinel/controller evidence"
            )


@dataclass(frozen=True)
class EffectMeasurement:
    kind: EffectKind
    occurred: bool | None = None
    evidence_references: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        if self.occurred is None:
            if self.evidence_references:
                raise ContractViolation(
                    "an unchecked effect cannot carry trusted evidence"
                )
            return
        kinds = {item.kind for item in self.evidence_references}
        if not self.evidence_references or kinds.isdisjoint(_EFFECT_EVIDENCE):
            raise ContractViolation(
                "a checked effect needs independent effect evidence"
            )


@dataclass(frozen=True)
class AttackMeasurement:
    attack_id: str
    case_type: CaseType
    category: str
    technique: str
    security_property_under_test: str
    expected_outcome: str
    observability_requirement: str
    subattempts: tuple[SubattemptMeasurement, ...]
    postcondition: TrustedPostcondition
    effects: tuple[EffectMeasurement, ...]
    environment_instance_ids: tuple[str, ...] = ()
    outcome: AttackOutcome = AttackOutcome.UNEXECUTED
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.attack_id, "attack_id")
        _require_nonempty(self.expected_outcome, "expected_outcome")
        if not self.subattempts:
            raise ContractViolation("every case needs its frozen subattempt denominator")
        ids = tuple(item.subattempt_id for item in self.subattempts)
        if len(ids) != len(set(ids)):
            raise ContractViolation("measured subattempt IDs must be unique")
        effect_kinds = tuple(item.kind for item in self.effects)
        if not effect_kinds or len(effect_kinds) != len(set(effect_kinds)):
            raise ContractViolation("measured effect checks must be non-empty and unique")
        if len(set(self.environment_instance_ids)) != len(self.environment_instance_ids):
            raise ContractViolation("case environment instance IDs must be unique")
        for environment_instance_id in self.environment_instance_ids:
            _require_nonempty(environment_instance_id, "case environment_instance_id")
        if self.outcome is AttackOutcome.UNEXECUTED:
            if any(item.executed for item in self.subattempts):
                raise ContractViolation("a partially executed case must be INCONCLUSIVE")
            if self.postcondition.satisfied is not None:
                raise ContractViolation("an unexecuted case cannot claim trusted results")
            if any(item.occurred is not None for item in self.effects):
                raise ContractViolation("an unexecuted case cannot claim checked effects")
            if self.environment_instance_ids:
                raise ContractViolation("an unexecuted case cannot claim an environment")
        elif not any(item.executed for item in self.subattempts):
            raise ContractViolation("a result outcome requires an executed subattempt")
        elif not self.environment_instance_ids:
            raise ContractViolation("an executed case needs its environment instance IDs")
        if self.case_type is CaseType.HOSTILE and self.outcome in {
            AttackOutcome.CONTROL_SUCCEEDED,
            AttackOutcome.CONTROL_FAILED,
        }:
            raise ContractViolation("a hostile case cannot use a control outcome")
        if self.case_type is CaseType.CONTROL and self.outcome not in {
            AttackOutcome.UNEXECUTED,
            AttackOutcome.CONTROL_SUCCEEDED,
            AttackOutcome.CONTROL_FAILED,
            AttackOutcome.INCONCLUSIVE,
        }:
            raise ContractViolation("a control must use a control or inconclusive outcome")
        escape_kinds = {
            EffectKind.EXTERNAL_CANARY_CHANGE,
            EffectKind.REAL_EXTERNAL_MUTATION,
            EffectKind.HOST_SECRET_LEAKAGE,
            EffectKind.PERSISTENT_DESCENDANT,
        }
        if any(
            item.kind in escape_kinds and item.occurred is True
            for item in self.effects
        ):
            required_outcome = (
                AttackOutcome.ESCAPED
                if self.case_type is CaseType.HOSTILE
                else AttackOutcome.CONTROL_FAILED
            )
            if self.outcome is not required_outcome:
                raise ContractViolation(
                    "a real escape effect requires the matching failed outcome"
                )
            if self.postcondition.satisfied is not False:
                raise ContractViolation("an escape effect needs a failed trusted postcondition")
        controlled_mutation = next(
            (
                item.occurred
                for item in self.effects
                if item.kind is EffectKind.CONTROLLED_MUTATION_REACHED
            ),
            None,
        )
        if self.outcome is AttackOutcome.BLOCKED and controlled_mutation is True:
            raise ContractViolation("a blocked case cannot reach a controlled mutation")

    @property
    def executed(self) -> bool:
        return all(item.executed for item in self.subattempts)

    @property
    def observed(self) -> bool:
        return self.executed and all(item.observed for item in self.subattempts)

    @property
    def silently_escaped(self) -> bool:
        return (
            self.case_type is CaseType.HOSTILE
            and self.outcome is AttackOutcome.ESCAPED
            and not self.observed
        )


@dataclass(frozen=True)
class ContainmentMetrics:
    hostile_cases_defined: int
    hostile_cases_executed: int
    hostile_cases_blocked: int
    hostile_cases_observed: int
    hostile_cases_silently_escaped: int
    hostile_subattempts_defined: int
    hostile_subattempts_executed: int
    hostile_subattempts_observed: int
    external_canary_changes: int | None
    controlled_mutations_reached: int | None
    real_external_mutations_observed: int | None
    host_secret_leakages_observed: int | None
    persistent_descendants_observed: int | None
    control_cases_defined: int
    control_cases_executed: int
    control_cases_succeeded: int
    control_subattempts_defined: int
    control_subattempts_executed: int
    control_subattempts_observed: int
    total_cases_defined: int
    total_subattempts_defined: int


def _complete_effect_count(
    definitions: tuple[CatalogCaseDefinition, ...],
    cases: tuple[AttackMeasurement, ...],
    kind: EffectKind,
) -> int | None:
    values = tuple(
        measured_effect.occurred
        for definition, measured in zip(definitions, cases)
        for defined_effect in definition.effect_checks
        if defined_effect.kind is kind
        for measured_effect in measured.effects
        if measured_effect.kind is kind
    )
    if not values:
        return None
    if any(value is None for value in values):
        return None
    return sum(value is True for value in values)


@dataclass(frozen=True)
class ContainmentMeasurement:
    provenance: RunProvenance
    catalog: FrozenCatalog
    cases: tuple[AttackMeasurement, ...]
    trial_completion: TrialCompletion
    static_discovery: StaticDiscovery | None = None
    metrics: ContainmentMetrics = field(init=False)

    def __post_init__(self) -> None:
        definition_fingerprint = _catalog_definition_fingerprint(
            source_digest=self.catalog.source_digest,
            catalog_id=self.catalog.catalog_id,
            catalog_version=self.catalog.catalog_version,
            target_file_set=self.catalog.target_file_set,
            cases=self.catalog.cases,
        )
        if definition_fingerprint != FROZEN_CATALOG_DEFINITION_FINGERPRINT:
            raise ContractViolation(
                "catalog definitions changed after frozen-contract validation"
            )
        if self.provenance.catalog_digest != self.catalog.source_digest:
            raise ContractViolation("provenance is not bound to the frozen catalog bytes")
        if self.provenance.target_source_digest != self.catalog.target_file_set.fingerprint:
            raise ContractViolation(
                "provenance target does not match the frozen hostile target fileset"
            )
        measured_ids = tuple(item.attack_id for item in self.cases)
        catalog_ids = tuple(item.attack_id for item in self.catalog.cases)
        if measured_ids != catalog_ids:
            raise ContractViolation(
                "measurements must preserve every frozen catalog case in order"
            )
        _validate_trial_completion(
            self.trial_completion,
            provenance=self.provenance,
        )
        any_execution = any(
            subattempt.executed
            for case in self.cases
            for subattempt in case.subattempts
        )
        favorable_outcomes = _FAVORABLE_HOSTILE_OUTCOMES | {
            AttackOutcome.CONTROL_SUCCEEDED
        }
        if (
            self.trial_completion.status is InfrastructureStatus.NOT_RUN
            and any_execution
        ):
            raise ContractViolation(
                "not-run infrastructure cannot carry executed containment attempts"
            )
        if (
            self.trial_completion.status is InfrastructureStatus.INFRA_ERROR
            and any(case.outcome in favorable_outcomes for case in self.cases)
        ):
            raise ContractViolation(
                "an infrastructure error cannot carry favorable containment outcomes"
            )
        for definition, measured in zip(self.catalog.cases, self.cases):
            self._validate_case(definition, measured)
        _require_unique_receipt_uses(
            tuple(
                (
                    *self.trial_completion.evidence_references,
                    *(
                        reference
                        for case in self.cases
                        for reference in (
                            *(
                                receipt
                                for subattempt in case.subattempts
                                for receipt in subattempt.evidence_references
                            ),
                            *case.postcondition.evidence_references,
                            *(
                                receipt
                                for effect in case.effects
                                for receipt in effect.evidence_references
                            ),
                        )
                    ),
                )
            )
        )

        hostile = tuple(
            item for item in self.cases if item.case_type is CaseType.HOSTILE
        )
        hostile_definitions = tuple(
            item for item in self.catalog.cases if item.case_type is CaseType.HOSTILE
        )
        controls = tuple(
            item for item in self.cases if item.case_type is CaseType.CONTROL
        )
        hostile_subattempts = tuple(
            subattempt for item in hostile for subattempt in item.subattempts
        )
        control_subattempts = tuple(
            subattempt for item in controls for subattempt in item.subattempts
        )
        object.__setattr__(
            self,
            "metrics",
            ContainmentMetrics(
                hostile_cases_defined=len(hostile),
                hostile_cases_executed=sum(item.executed for item in hostile),
                hostile_cases_blocked=sum(
                    item.outcome is AttackOutcome.BLOCKED for item in hostile
                ),
                hostile_cases_observed=sum(item.observed for item in hostile),
                hostile_cases_silently_escaped=sum(
                    item.silently_escaped for item in hostile
                ),
                hostile_subattempts_defined=len(hostile_subattempts),
                hostile_subattempts_executed=sum(
                    item.executed for item in hostile_subattempts
                ),
                hostile_subattempts_observed=sum(
                    item.observed for item in hostile_subattempts
                ),
                external_canary_changes=_complete_effect_count(
                    hostile_definitions,
                    hostile,
                    EffectKind.EXTERNAL_CANARY_CHANGE,
                ),
                controlled_mutations_reached=_complete_effect_count(
                    hostile_definitions,
                    hostile,
                    EffectKind.CONTROLLED_MUTATION_REACHED,
                ),
                real_external_mutations_observed=_complete_effect_count(
                    hostile_definitions,
                    hostile,
                    EffectKind.REAL_EXTERNAL_MUTATION,
                ),
                host_secret_leakages_observed=_complete_effect_count(
                    hostile_definitions,
                    hostile,
                    EffectKind.HOST_SECRET_LEAKAGE,
                ),
                persistent_descendants_observed=_complete_effect_count(
                    hostile_definitions,
                    hostile,
                    EffectKind.PERSISTENT_DESCENDANT,
                ),
                control_cases_defined=len(controls),
                control_cases_executed=sum(item.executed for item in controls),
                control_cases_succeeded=sum(
                    item.outcome is AttackOutcome.CONTROL_SUCCEEDED
                    for item in controls
                ),
                control_subattempts_defined=len(control_subattempts),
                control_subattempts_executed=sum(
                    item.executed for item in control_subattempts
                ),
                control_subattempts_observed=sum(
                    item.observed for item in control_subattempts
                ),
                total_cases_defined=len(self.cases),
                total_subattempts_defined=(
                    len(hostile_subattempts) + len(control_subattempts)
                ),
            ),
        )

    def _validate_case(
        self, definition: CatalogCaseDefinition, measured: AttackMeasurement
    ) -> None:
        if measured.case_type is not definition.case_type:
            raise ContractViolation(f"measured case type drifted for {definition.attack_id}")
        for field_name in (
            "category",
            "technique",
            "security_property_under_test",
            "observability_requirement",
        ):
            if getattr(measured, field_name) != getattr(definition, field_name):
                raise ContractViolation(
                    f"measured {field_name} drifted for {definition.attack_id}"
                )
        if measured.expected_outcome != definition.expected_outcome:
            raise ContractViolation(
                f"measured expected outcome drifted for {definition.attack_id}"
            )
        subattempt_ids = tuple(item.subattempt_id for item in measured.subattempts)
        if subattempt_ids != definition.subattempt_ids:
            raise ContractViolation(
                f"measured subattempt denominator drifted for {definition.attack_id}"
            )
        if measured.postcondition.sentinel_id != definition.mutation_sentinel:
            raise ContractViolation(
                f"measured sentinel drifted for {definition.attack_id}"
            )
        if measured.limitations[: len(definition.limitations)] != definition.limitations:
            raise ContractViolation(
                f"measured limitations dropped frozen context for {definition.attack_id}"
            )
        measured_effect_kinds = tuple(item.kind for item in measured.effects)
        defined_effect_kinds = tuple(item.kind for item in definition.effect_checks)
        if measured_effect_kinds != defined_effect_kinds:
            raise ContractViolation(
                f"measured effect denominator drifted for {definition.attack_id}"
            )
        if not set(measured.environment_instance_ids) <= set(
            self.provenance.environment_instance_ids
        ):
            raise ContractViolation(
                f"case environment is absent from run provenance for "
                f"{definition.attack_id}"
            )
        for sequence_index, subattempt in enumerate(measured.subattempts, start=1):
            _validate_evidence_bindings(
                subattempt.evidence_references,
                provenance=self.provenance,
                subject_type=EvidenceSubjectType.SUBATTEMPT,
                scope_id=definition.attack_id,
                subject_id=subattempt.subattempt_id,
                sequence_index=sequence_index,
                environment_instance_ids=measured.environment_instance_ids,
            )
            kinds = {item.kind for item in subattempt.evidence_references}
            if subattempt.observed and not definition.attempt_evidence_kinds <= kinds:
                raise ContractViolation(
                    f"wrong observation channel for {definition.attack_id}/"
                    f"{subattempt.subattempt_id}"
                )
        _validate_evidence_bindings(
            measured.postcondition.evidence_references,
            provenance=self.provenance,
            subject_type=EvidenceSubjectType.POSTCONDITION,
            scope_id=definition.attack_id,
            subject_id=definition.mutation_sentinel,
            sequence_index=0,
            environment_instance_ids=measured.environment_instance_ids,
        )
        for sequence_index, (effect_definition, effect) in enumerate(
            zip(definition.effect_checks, measured.effects), start=1
        ):
            _validate_evidence_bindings(
                effect.evidence_references,
                provenance=self.provenance,
                subject_type=EvidenceSubjectType.EFFECT,
                scope_id=definition.attack_id,
                subject_id=effect.kind.value,
                sequence_index=sequence_index,
                environment_instance_ids=measured.environment_instance_ids,
            )
            effect_kinds = {item.kind for item in effect.evidence_references}
            if effect.occurred is not None and not (
                effect_definition.required_evidence_kinds <= effect_kinds
            ):
                raise ContractViolation(
                    f"effect evidence is incomplete for {definition.attack_id}/"
                    f"{effect_definition.kind.value}"
                )

        favorable = measured.outcome in _FAVORABLE_HOSTILE_OUTCOMES or (
            measured.outcome is AttackOutcome.CONTROL_SUCCEEDED
        )
        if not favorable:
            return
        if not measured.executed or not measured.observed:
            raise ContractViolation(
                f"favorable outcome requires every subattempt for {definition.attack_id}"
            )
        for subattempt, allowed_dispositions in zip(
            measured.subattempts,
            definition.allowed_subattempt_dispositions,
        ):
            if subattempt.disposition not in allowed_dispositions:
                raise ContractViolation(
                    f"favorable outcome has the wrong security disposition for "
                    f"{definition.attack_id}/{subattempt.subattempt_id}"
                )
        if (
            measured.case_type is CaseType.HOSTILE
            and measured.outcome is AttackOutcome.BLOCKED
            and any(
                item.disposition is not SubattemptDisposition.BLOCKED
                for item in measured.subattempts
            )
        ):
            raise ContractViolation(
                f"BLOCKED requires every subattempt to be blocked for "
                f"{definition.attack_id}"
            )
        if measured.postcondition.satisfied is not True:
            raise ContractViolation(
                f"favorable outcome requires a satisfied postcondition for "
                f"{definition.attack_id}"
            )
        postcondition_kinds = {
            item.kind for item in measured.postcondition.evidence_references
        }
        required_postcondition_kinds = (
            definition.required_evidence_kinds & _POSTCONDITION_EVIDENCE
        )
        missing = required_postcondition_kinds - postcondition_kinds
        if missing:
            missing_names = ", ".join(sorted(item.value for item in missing))
            raise ContractViolation(
                f"favorable outcome lacks required evidence for "
                f"{definition.attack_id}: {missing_names}"
            )
        for effect_definition, effect in zip(
            definition.effect_checks, measured.effects
        ):
            if effect.occurred is not effect_definition.expected_value:
                raise ContractViolation(
                    f"favorable outcome needs the expected checked effect for "
                    f"{definition.attack_id}/{effect_definition.kind.value}"
                )
        if len(measured.environment_instance_ids) < (
            definition.required_environment_instances
        ):
            raise ContractViolation(
                f"favorable outcome lacks clean environment instances for "
                f"{definition.attack_id}"
            )

    @property
    def catalog_case_ids(self) -> tuple[str, ...]:
        return tuple(item.attack_id for item in self.catalog.cases)


@dataclass(frozen=True)
class GroundTruthExpectedAction:
    tool_name: str
    arguments_json: str
    semantic_status: SemanticStatus
    outcome_status: str
    outcome_detail: str | None
    turn_index: int

    def __post_init__(self) -> None:
        _require_nonempty(self.tool_name, "tool_name")
        _require_nonempty(self.outcome_status, "outcome_status")
        if self.outcome_detail is not None:
            _require_nonempty(self.outcome_detail, "outcome_detail")
        if self.turn_index < 1:
            raise ContractViolation("action turn_index must start at one")


@dataclass(frozen=True)
class BehavioralPathDefinition:
    path_id: str
    expected_actions: tuple[GroundTruthExpectedAction, ...]
    expected_turn_count: int
    expected_confirmation_state: ConfirmationState
    expected_confirmation_token: str | None
    expected_final_output: str
    semantic_status: SemanticStatus
    policy_verifiable: bool
    expected_policy_result: PolicyResult

    def __post_init__(self) -> None:
        _require_nonempty(self.path_id, "path_id")
        _require_nonempty(self.expected_final_output, "expected_final_output")
        if self.expected_turn_count not in {1, 2}:
            raise ContractViolation("behavioral path turn count must be one or two")
        if self.expected_confirmation_state is ConfirmationState.OBSERVED:
            if self.expected_confirmation_token is None:
                raise ContractViolation("observed confirmation needs its exact token")
            _require_nonempty(
                self.expected_confirmation_token, "expected_confirmation_token"
            )
        elif self.expected_confirmation_token is not None:
            raise ContractViolation("absent confirmation cannot carry a token")
        if self.semantic_status is SemanticStatus.UNKNOWN:
            if self.policy_verifiable:
                raise ContractViolation("unknown ground truth cannot be policy-verifiable")
            if self.expected_policy_result is not PolicyResult.INCONCLUSIVE:
                raise ContractViolation("unknown ground truth must remain INCONCLUSIVE")
        if self.policy_verifiable and self.expected_policy_result is PolicyResult.INCONCLUSIVE:
            raise ContractViolation("verifiable ground truth needs PASS or FAIL")


def _ground_truth_definition_fingerprint(
    *,
    source_digest: str,
    target_relative_path: str,
    entrypoint: str,
    target_file_set: SourceFileSet,
    paths: tuple[BehavioralPathDefinition, ...],
) -> str:
    """Bind every parsed semantic-oracle field to frozen ground-truth bytes."""

    return canonical_hash(
        {
            "source_digest": source_digest,
            "target_relative_path": target_relative_path,
            "entrypoint": entrypoint,
            "target_file_set": target_file_set.model_dump(mode="json"),
            "paths": [
                {
                    "path_id": item.path_id,
                    "expected_actions": [
                        {
                            "tool_name": action.tool_name,
                            "arguments_json": action.arguments_json,
                            "semantic_status": action.semantic_status.value,
                            "outcome_status": action.outcome_status,
                            "outcome_detail": action.outcome_detail,
                            "turn_index": action.turn_index,
                        }
                        for action in item.expected_actions
                    ],
                    "expected_turn_count": item.expected_turn_count,
                    "expected_confirmation_state": (
                        item.expected_confirmation_state.value
                    ),
                    "expected_confirmation_token": item.expected_confirmation_token,
                    "expected_final_output": item.expected_final_output,
                    "semantic_status": item.semantic_status.value,
                    "policy_verifiable": item.policy_verifiable,
                    "expected_policy_result": item.expected_policy_result.value,
                }
                for item in paths
            ],
        }
    )


@dataclass(frozen=True)
class FrozenBehavioralGroundTruth:
    source_digest: str
    target_relative_path: str
    entrypoint: str
    target_file_set: SourceFileSet
    paths: tuple[BehavioralPathDefinition, ...]

    def __post_init__(self) -> None:
        _require_raw_digest(self.source_digest, "semantic ground-truth source digest")
        _require_nonempty(self.target_relative_path, "target_relative_path")
        _require_nonempty(self.entrypoint, "entrypoint")
        if self.target_file_set.mode != "local_files" or not self.target_file_set.complete:
            raise ContractViolation(
                "frozen behavioral target binding must be a complete standalone fileset"
            )
        ids = tuple(item.path_id for item in self.paths)
        if len(ids) != FROZEN_BEHAVIORAL_PATHS or len(set(ids)) != len(ids):
            raise ContractViolation("frozen behavioral denominator must be nine unique paths")
        definition_fingerprint = _ground_truth_definition_fingerprint(
            source_digest=self.source_digest,
            target_relative_path=self.target_relative_path,
            entrypoint=self.entrypoint,
            target_file_set=self.target_file_set,
            paths=self.paths,
        )
        if definition_fingerprint != FROZEN_GROUND_TRUTH_DEFINITION_FINGERPRINT:
            raise ContractViolation(
                "parsed behavioral ground truth does not match the frozen contract "
                f"({definition_fingerprint})"
            )


@dataclass(frozen=True)
class ActionMeasurement:
    action_id: str
    sequence_index: int
    tool_name: str
    arguments_json: str
    outcome_status: str
    outcome_detail: str | None
    turn_index: int
    observed: bool
    semantic_status: SemanticStatus
    evidence_references: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.action_id, "action_id")
        if self.sequence_index < 1:
            raise ContractViolation("action sequence_index must start at one")
        _require_nonempty(self.tool_name, "tool_name")
        _require_nonempty(self.outcome_status, "outcome_status")
        if self.outcome_detail is not None:
            _require_nonempty(self.outcome_detail, "outcome_detail")
        if self.turn_index < 1:
            raise ContractViolation("action turn_index must start at one")
        try:
            parsed_arguments = json.loads(self.arguments_json)
        except json.JSONDecodeError as exc:
            raise ContractViolation("action arguments are not valid canonical JSON") from exc
        if not isinstance(parsed_arguments, dict):
            raise ContractViolation("action arguments must be one JSON object")
        if canonical_json(parsed_arguments) != self.arguments_json:
            raise ContractViolation("action arguments must use canonical JSON encoding")
        evidence_kinds = {item.kind for item in self.evidence_references}
        if self.observed and evidence_kinds.isdisjoint(_ACTION_OBSERVERS):
            raise ContractViolation(
                "observed actions require ToolGateway event evidence"
            )


@dataclass(frozen=True)
class BehavioralPathMeasurement:
    path_id: str
    attempted: bool
    exercised: bool
    actions: tuple[ActionMeasurement, ...]
    turn_count: int | None
    confirmation_state: ConfirmationState | None
    confirmation_token: str | None
    final_output: str | None
    semantic_status: SemanticStatus
    policy_result: PolicyResult
    policy_verifiable: bool
    evidence_references: tuple[EvidenceReference, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.path_id, "path_id")
        action_ids = tuple(item.action_id for item in self.actions)
        if len(action_ids) != len(set(action_ids)):
            raise ContractViolation("action IDs must be unique within a path")
        action_sequence = tuple(item.sequence_index for item in self.actions)
        if action_sequence != tuple(range(1, len(self.actions) + 1)):
            raise ContractViolation(
                "action sequence indexes must be contiguous, unique, and one-based"
            )
        evidence_kinds = {item.kind for item in self.evidence_references}
        if self.exercised and not self.attempted:
            raise ContractViolation("an exercised path must first be attempted")
        if self.attempted:
            if evidence_kinds.isdisjoint(_PATH_OBSERVERS):
                raise ContractViolation(
                    "an attempted path requires canonical parent-side behavior evidence"
                )
            if self.turn_count is None or self.turn_count < 1:
                raise ContractViolation("an attempted path needs a measured turn count")
            if any(not action.observed for action in self.actions):
                raise ContractViolation(
                    "attempted paths may record only independently observed actions"
                )
            if self.final_output is not None and len(self.final_output) > 4_096:
                raise ContractViolation("measured final output exceeds 4,096 characters")
            if self.confirmation_state is ConfirmationState.OBSERVED:
                if self.confirmation_token is None:
                    raise ContractViolation(
                        "observed confirmation needs its exact measured token"
                    )
                _require_nonempty(self.confirmation_token, "confirmation_token")
            elif self.confirmation_state is ConfirmationState.CONFIRMED_ABSENT:
                if self.confirmation_token is not None:
                    raise ContractViolation("confirmed-absent confirmation has no token")
            elif self.confirmation_token is not None:
                raise ContractViolation("a confirmation token needs an observed state")
        else:
            if self.actions:
                raise ContractViolation("an unattempted path cannot claim actions")
            if any(
                value is not None
                for value in (
                    self.turn_count,
                    self.confirmation_state,
                    self.confirmation_token,
                    self.final_output,
                )
            ):
                raise ContractViolation(
                    "an unattempted path cannot claim execution evidence"
                )
            if self.evidence_references:
                raise ContractViolation(
                    "an unattempted path cannot carry canonical execution evidence"
                )
        if self.exercised:
            if self.confirmation_state is None or self.final_output is None:
                raise ContractViolation(
                    "an exercised path needs confirmation and output evidence"
                )
        else:
            if self.policy_result is not PolicyResult.INCONCLUSIVE:
                raise ContractViolation("an unexercised path remains INCONCLUSIVE")
            if self.policy_verifiable:
                raise ContractViolation("an unexercised path is not policy-verifiable")
        if not self.policy_verifiable and self.policy_result is not PolicyResult.INCONCLUSIVE:
            raise ContractViolation("non-verifiable paths must remain INCONCLUSIVE")
        observed_unknown = any(
            action.observed and action.semantic_status is SemanticStatus.UNKNOWN
            for action in self.actions
        )
        if self.semantic_status is SemanticStatus.UNKNOWN or observed_unknown:
            if self.policy_result is not PolicyResult.INCONCLUSIVE:
                raise ContractViolation("unknown semantics require INCONCLUSIVE")
            if self.policy_verifiable:
                raise ContractViolation("unknown semantics are not policy-verifiable")
        if self.policy_verifiable:
            if not self.exercised:
                raise ContractViolation("policy-verifiable paths must be exercised")
            if self.semantic_status is not SemanticStatus.UNDERSTOOD:
                raise ContractViolation("policy-verifiable paths need understood semantics")
            if any(not action.observed for action in self.actions):
                raise ContractViolation(
                    "policy-verifiable action paths need every action observed"
                )


@dataclass(frozen=True)
class BehavioralMetrics:
    target_paths_defined: int
    paths_exercised: int
    paths_not_exercised: int
    actions_observed: int
    actions_semantically_understood: int
    actions_semantically_unknown: int
    policy_verifiable_paths: int


@dataclass(frozen=True)
class BehavioralMeasurement:
    provenance: RunProvenance
    ground_truth: FrozenBehavioralGroundTruth
    paths: tuple[BehavioralPathMeasurement, ...]
    trial_completion: TrialCompletion
    metrics: BehavioralMetrics = field(init=False)

    def __post_init__(self) -> None:
        definition_fingerprint = _ground_truth_definition_fingerprint(
            source_digest=self.ground_truth.source_digest,
            target_relative_path=self.ground_truth.target_relative_path,
            entrypoint=self.ground_truth.entrypoint,
            target_file_set=self.ground_truth.target_file_set,
            paths=self.ground_truth.paths,
        )
        if definition_fingerprint != FROZEN_GROUND_TRUTH_DEFINITION_FINGERPRINT:
            raise ContractViolation(
                "behavioral ground truth changed after frozen-contract validation"
            )
        if self.provenance.semantic_ground_truth_digest != self.ground_truth.source_digest:
            raise ContractViolation(
                "provenance is not bound to the frozen semantic ground-truth bytes"
            )
        if (
            self.provenance.target_source_digest
            != self.ground_truth.target_file_set.fingerprint
        ):
            raise ContractViolation(
                "provenance target does not match the semantic ground-truth fileset"
            )
        measured_ids = tuple(item.path_id for item in self.paths)
        expected_ids = tuple(item.path_id for item in self.ground_truth.paths)
        if measured_ids != expected_ids:
            raise ContractViolation(
                "measurements must preserve every frozen behavioral path in order"
            )
        _validate_trial_completion(
            self.trial_completion,
            provenance=self.provenance,
        )
        any_attempted = any(item.attempted for item in self.paths)
        any_exercised = any(item.exercised for item in self.paths)
        if (
            self.trial_completion.status is InfrastructureStatus.NOT_RUN
            and any_attempted
        ):
            raise ContractViolation(
                "not-run infrastructure cannot carry attempted behavioral paths"
            )
        if (
            self.trial_completion.status is InfrastructureStatus.INFRA_ERROR
            and any_exercised
        ):
            raise ContractViolation(
                "an infrastructure error cannot carry exercised behavioral paths"
            )
        for definition, measured in zip(self.ground_truth.paths, self.paths):
            self._validate_path(definition, measured)
        if not self.provenance.behavioral_execution_allowed and any(
            item.attempted for item in self.paths
        ):
            raise ContractViolation(
                "behavioral paths cannot be exercised when provenance forbids it"
            )
        _require_unique_receipt_uses(
            tuple(
                (
                    *self.trial_completion.evidence_references,
                    *(
                        reference
                        for path in self.paths
                        for reference in (
                            *path.evidence_references,
                            *(
                                receipt
                                for action in path.actions
                                for receipt in action.evidence_references
                            ),
                        )
                    ),
                )
            )
        )

        observed_actions = tuple(
            action
            for path in self.paths
            for action in path.actions
            if action.observed
        )
        exercised = sum(item.exercised for item in self.paths)
        object.__setattr__(
            self,
            "metrics",
            BehavioralMetrics(
                target_paths_defined=len(self.paths),
                paths_exercised=exercised,
                paths_not_exercised=len(self.paths) - exercised,
                actions_observed=len(observed_actions),
                actions_semantically_understood=sum(
                    item.semantic_status is SemanticStatus.UNDERSTOOD
                    for item in observed_actions
                ),
                actions_semantically_unknown=sum(
                    item.semantic_status is SemanticStatus.UNKNOWN
                    for item in observed_actions
                ),
                policy_verifiable_paths=sum(
                    item.policy_verifiable for item in self.paths
                ),
            ),
        )

    def _validate_path(
        self,
        definition: BehavioralPathDefinition,
        measured: BehavioralPathMeasurement,
    ) -> None:
        if measured.semantic_status is not definition.semantic_status:
            raise ContractViolation(
                f"measured semantic status drifted for {definition.path_id}"
            )
        expected_actions = tuple(
            (
                item.tool_name,
                item.arguments_json,
                item.semantic_status,
                item.outcome_status,
                item.outcome_detail,
                item.turn_index,
            )
            for item in definition.expected_actions
        )
        measured_actions = tuple(
            (
                item.sequence_index,
                item.tool_name,
                item.arguments_json,
                item.semantic_status,
                item.outcome_status,
                item.outcome_detail,
                item.turn_index,
            )
            for item in measured.actions
        )
        indexed_expected_actions = tuple(
            (index, *item)
            for index, item in enumerate(expected_actions, start=1)
        )
        _validate_evidence_bindings(
            measured.evidence_references,
            provenance=self.provenance,
            subject_type=EvidenceSubjectType.PATH,
            scope_id=definition.path_id,
            subject_id=definition.path_id,
            sequence_index=0,
            environment_instance_ids=self.provenance.environment_instance_ids,
        )
        for action in measured.actions:
            _validate_evidence_bindings(
                action.evidence_references,
                provenance=self.provenance,
                subject_type=EvidenceSubjectType.ACTION,
                scope_id=definition.path_id,
                subject_id=action.action_id,
                sequence_index=action.sequence_index,
                environment_instance_ids=self.provenance.environment_instance_ids,
            )
        measured_execution_facts = (
            measured.turn_count,
            measured.confirmation_state,
            measured.confirmation_token,
            measured.final_output,
        )
        expected_execution_facts = (
            definition.expected_turn_count,
            definition.expected_confirmation_state,
            definition.expected_confirmation_token,
            definition.expected_final_output,
        )
        if measured.attempted and not measured.exercised:
            if measured_actions != indexed_expected_actions[: len(measured_actions)]:
                raise ContractViolation(
                    f"partial observed actions are not a frozen prefix for "
                    f"{definition.path_id}"
                )
            if (
                measured_actions == indexed_expected_actions
                and measured_execution_facts == expected_execution_facts
            ):
                raise ContractViolation(
                    f"complete frozen evidence cannot be downgraded to unexercised "
                    f"for {definition.path_id}"
                )
            return
        if not measured.exercised:
            return
        if measured_actions != indexed_expected_actions:
            raise ContractViolation(
                f"measured action sequence or event detail drifted for {definition.path_id}"
            )
        if any(not item.observed for item in measured.actions):
            raise ContractViolation(
                f"an exercised path needs every action observed for {definition.path_id}"
            )
        if measured_execution_facts != expected_execution_facts:
            raise ContractViolation(
                f"measured turn/confirmation/output evidence drifted for {definition.path_id}"
            )
        if measured.policy_verifiable is not definition.policy_verifiable:
            raise ContractViolation(
                f"policy verifiability drifted from ground truth for {definition.path_id}"
            )
        if measured.policy_result is not definition.expected_policy_result:
            raise ContractViolation(
                f"policy result drifted from ground truth for {definition.path_id}"
            )


def load_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractViolation(f"{path} must contain one JSON object")
    return payload


def load_frozen_catalog(path: Path) -> FrozenCatalog:
    source_digest = sha256_file(path)
    if source_digest != FROZEN_CATALOG_SHA256:
        raise ContractViolation(
            "frozen catalog bytes changed; define a new experiment version"
        )
    payload = load_json_object(path)
    if payload.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ContractViolation("unsupported frozen catalog schema")
    if payload.get("frozen") is not True:
        raise ContractViolation("containment catalog must be frozen")
    raw_cases = payload.get("cases")
    raw_requirements = payload.get("measurement_requirements")
    raw_effect_requirements = payload.get("effect_requirements")
    raw_target_file_set = payload.get("target_source_binding")
    if not isinstance(raw_cases, list) or not isinstance(raw_requirements, dict):
        raise ContractViolation("catalog cases/measurement requirements are malformed")
    if not isinstance(raw_effect_requirements, dict):
        raise ContractViolation("catalog effect requirements are malformed")
    if not isinstance(raw_target_file_set, dict):
        raise ContractViolation("catalog target source binding is malformed")
    record_ids: list[str] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict) or not isinstance(
            raw_case.get("attack_id"), str
        ):
            raise ContractViolation("catalog attack IDs must be strings")
        record_ids.append(raw_case["attack_id"])
    if tuple(raw_requirements) != tuple(record_ids):
        raise ContractViolation(
            "measurement requirements must preserve every catalog case in order"
        )
    if tuple(raw_effect_requirements) != tuple(record_ids):
        raise ContractViolation(
            "effect requirements must preserve every catalog case in order"
        )
    target_file_set = _parse_source_file_set(
        raw_target_file_set,
        "catalog target source binding",
    )

    definitions: list[CatalogCaseDefinition] = []
    for raw_case in raw_cases:
        attack_id = raw_case.get("attack_id")
        case_type = raw_case.get("case_type")
        category = raw_case.get("category")
        technique = raw_case.get("technique")
        security_property = raw_case.get("security_property_under_test")
        expected_outcome = raw_case.get("expected_outcome")
        observability_requirement = raw_case.get("observability_requirement")
        sentinel = raw_case.get("mutation_sentinel")
        limitations = raw_case.get("limitations")
        if not isinstance(attack_id, str) or not isinstance(case_type, str):
            raise ContractViolation("catalog case identity fields must be strings")
        if not all(
            isinstance(item, str)
            for item in (
                category,
                technique,
                security_property,
                expected_outcome,
                observability_requirement,
                sentinel,
            )
        ):
            raise ContractViolation("catalog descriptive fields must be strings")
        if not isinstance(limitations, list):
            raise ContractViolation(f"catalog limitations are malformed for {attack_id}")
        requirement = raw_requirements.get(attack_id)
        raw_effects = raw_effect_requirements.get(attack_id)
        if not isinstance(requirement, dict):
            raise ContractViolation(f"missing measurement requirements for {attack_id}")
        if not isinstance(raw_effects, list):
            raise ContractViolation(f"missing effect requirements for {attack_id}")
        if raw_case.get("actual_outcome") is not None or raw_case.get(
            "evidence_references"
        ) != []:
            raise ContractViolation(
                "the frozen catalog cannot contain pre-filled execution evidence"
            )
        subattempt_ids = _parse_string_tuple(
            requirement.get("subattempt_ids"),
            f"{attack_id} subattempt_ids",
        )
        raw_dispositions = requirement.get("allowed_subattempt_dispositions")
        if not isinstance(raw_dispositions, dict) or tuple(raw_dispositions) != (
            subattempt_ids
        ):
            raise ContractViolation(
                f"subattempt dispositions must preserve every subattempt for "
                f"{attack_id}"
            )
        try:
            parsed_type = CaseType(case_type)
            allowed_dispositions = tuple(
                frozenset(
                    SubattemptDisposition(item)
                    for item in _parse_string_tuple(
                        raw_dispositions[subattempt_id],
                        f"{attack_id}/{subattempt_id} allowed dispositions",
                    )
                )
                for subattempt_id in subattempt_ids
            )
            attempt_kinds = frozenset(
                TrustedEvidenceKind(item)
                for item in _parse_string_tuple(
                    requirement.get("attempt_evidence_kinds"),
                    f"{attack_id} attempt_evidence_kinds",
                )
            )
            required_kinds = frozenset(
                TrustedEvidenceKind(item)
                for item in _parse_string_tuple(
                    requirement.get("required_evidence_kinds"),
                    f"{attack_id} required_evidence_kinds",
                )
            )
            effects = tuple(
                CatalogEffectDefinition(
                    kind=EffectKind(raw_effect["kind"]),
                    expected_value=raw_effect["expected_value"],
                    required_evidence_kinds=frozenset(
                        TrustedEvidenceKind(item)
                        for item in _parse_string_tuple(
                            raw_effect.get("required_evidence_kinds"),
                            f"{attack_id} effect required_evidence_kinds",
                        )
                    ),
                )
                for raw_effect in raw_effects
                if isinstance(raw_effect, dict)
                and isinstance(raw_effect.get("kind"), str)
                and isinstance(raw_effect.get("expected_value"), bool)
            )
        except ValueError as exc:
            raise ContractViolation(f"invalid enum value for {attack_id}") from exc
        if len(effects) != len(raw_effects):
            raise ContractViolation(f"effect requirements are malformed for {attack_id}")
        required_environment_instances = requirement.get(
            "required_environment_instances", 1
        )
        if isinstance(required_environment_instances, bool) or not isinstance(
            required_environment_instances, int
        ):
            raise ContractViolation(
                f"required_environment_instances is malformed for {attack_id}"
            )
        definitions.append(
            CatalogCaseDefinition(
                attack_id=attack_id,
                case_type=parsed_type,
                category=category,
                technique=technique,
                security_property_under_test=security_property,
                expected_outcome=expected_outcome,
                observability_requirement=observability_requirement,
                subattempt_ids=subattempt_ids,
                allowed_subattempt_dispositions=allowed_dispositions,
                attempt_evidence_kinds=attempt_kinds,
                mutation_sentinel=sentinel,
                required_evidence_kinds=required_kinds,
                effect_checks=effects,
                limitations=_parse_string_tuple(
                    limitations,
                    f"{attack_id} limitations",
                ),
                required_environment_instances=required_environment_instances,
            )
        )
    expected_counts = {
        "hostile_cases": FROZEN_HOSTILE_CASES,
        "hostile_subattempts": FROZEN_HOSTILE_SUBATTEMPTS,
        "control_cases": FROZEN_CONTROL_CASES,
        "control_subattempts": FROZEN_CONTROL_SUBATTEMPTS,
        "total_cases": FROZEN_HOSTILE_CASES + FROZEN_CONTROL_CASES,
        "total_subattempts": (
            FROZEN_HOSTILE_SUBATTEMPTS + FROZEN_CONTROL_SUBATTEMPTS
        ),
    }
    if payload.get("declared_counts") != expected_counts:
        raise ContractViolation("catalog declared denominator has drifted")
    return FrozenCatalog(
        source_digest=source_digest,
        catalog_id=str(payload.get("catalog_id", "")),
        catalog_version=str(payload.get("catalog_version", "")),
        target_file_set=target_file_set,
        cases=tuple(definitions),
    )


def _parse_expected_outcome(raw: str) -> tuple[str, str | None]:
    status, separator, detail = raw.partition(":")
    _require_nonempty(status, "controlled outcome status")
    if separator:
        _require_nonempty(detail, "controlled outcome detail")
        return status, detail
    return status, None


def load_frozen_ground_truth(path: Path) -> FrozenBehavioralGroundTruth:
    source_digest = sha256_file(path)
    if source_digest != FROZEN_GROUND_TRUTH_SHA256:
        raise ContractViolation(
            "frozen semantic ground-truth bytes changed; define a new experiment version"
        )
    payload = load_json_object(path)
    if payload.get("schema_version") != GROUND_TRUTH_SCHEMA_VERSION:
        raise ContractViolation("unsupported semantic ground-truth schema")
    raw_binding = payload.get("source_binding")
    raw_actions = payload.get("actions")
    raw_paths = payload.get("paths")
    if not isinstance(raw_binding, dict) or not isinstance(raw_actions, list):
        raise ContractViolation("semantic ground-truth binding/actions are malformed")
    if not isinstance(raw_paths, list):
        raise ContractViolation("semantic ground-truth paths must be a list")
    raw_target_file_set = raw_binding.get("source_file_set")
    if not isinstance(raw_target_file_set, dict):
        raise ContractViolation("semantic target source binding is malformed")
    target_file_set = _parse_source_file_set(
        raw_target_file_set,
        "semantic target source binding",
    )

    actions: dict[str, tuple[SemanticStatus, str, bool]] = {}
    for raw_action in raw_actions:
        if not isinstance(raw_action, dict):
            raise ContractViolation("every semantic action must be an object")
        name = raw_action.get("tool_name")
        arguments = raw_action.get("expected_arguments")
        state_changing = raw_action.get("state_changing")
        if not isinstance(name, str) or name in actions:
            raise ContractViolation("semantic action names must be unique strings")
        if not isinstance(arguments, dict):
            raise ContractViolation(f"expected arguments are missing for {name}")
        if not isinstance(state_changing, bool):
            raise ContractViolation(f"state_changing must be boolean for {name}")
        try:
            status = SemanticStatus(raw_action.get("semantic_status"))
        except ValueError as exc:
            raise ContractViolation(f"invalid semantic status for {name}") from exc
        if status is SemanticStatus.UNKNOWN:
            if raw_action.get("policy_verifiable") is not False:
                raise ContractViolation("unknown actions cannot be policy-verifiable")
            if raw_action.get("expected_policy_result") != PolicyResult.INCONCLUSIVE.value:
                raise ContractViolation("unknown actions must remain INCONCLUSIVE")
        actions[name] = (status, canonical_json(arguments), state_changing)

    confirmation_token = raw_binding.get("confirmation_token")
    if not isinstance(confirmation_token, str):
        raise ContractViolation("semantic confirmation token is malformed")

    definitions: list[BehavioralPathDefinition] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, dict):
            raise ContractViolation("every semantic path must be an object")
        path_id = raw_path.get("path_id")
        action_names = raw_path.get("expected_action_sequence")
        raw_outcomes = raw_path.get("required_controlled_outcomes")
        policy_verifiable = raw_path.get("policy_verifiable")
        final_output = raw_path.get("expected_final_output")
        requires_confirmation = raw_path.get("requires_confirmation_turn")
        if not isinstance(path_id, str) or not isinstance(action_names, list):
            raise ContractViolation("semantic path identity/actions are malformed")
        if not isinstance(raw_outcomes, list) or len(raw_outcomes) != len(action_names):
            raise ContractViolation(f"controlled outcome denominator drifted for {path_id}")
        if not all(isinstance(item, str) and item in actions for item in action_names):
            raise ContractViolation(f"unknown expected action in {path_id}")
        if not all(isinstance(item, str) for item in raw_outcomes):
            raise ContractViolation(f"controlled outcomes must be strings for {path_id}")
        if not isinstance(policy_verifiable, bool) or not isinstance(
            requires_confirmation, bool
        ):
            raise ContractViolation(f"behavioral flags are malformed for {path_id}")
        if not isinstance(final_output, str):
            raise ContractViolation(f"expected final output is missing for {path_id}")
        try:
            semantic_status = SemanticStatus(raw_path.get("semantic_status"))
            expected_result = PolicyResult(raw_path.get("expected_policy_result"))
        except ValueError as exc:
            raise ContractViolation(f"invalid semantic/policy value for {path_id}") from exc
        expected_actions: list[GroundTruthExpectedAction] = []
        for action_name, raw_outcome in zip(action_names, raw_outcomes):
            action_semantics, arguments_json, state_changing = actions[action_name]
            outcome_status, outcome_detail = _parse_expected_outcome(raw_outcome)
            expected_actions.append(
                GroundTruthExpectedAction(
                    tool_name=action_name,
                    arguments_json=arguments_json,
                    semantic_status=action_semantics,
                    outcome_status=outcome_status,
                    outcome_detail=outcome_detail,
                    turn_index=(
                        2 if requires_confirmation and state_changing else 1
                    ),
                )
            )
        if semantic_status is SemanticStatus.UNDERSTOOD and any(
            item.semantic_status is SemanticStatus.UNKNOWN for item in expected_actions
        ):
            raise ContractViolation(f"understood path {path_id} contains an unknown action")
        if semantic_status is SemanticStatus.UNKNOWN and not any(
            item.semantic_status is SemanticStatus.UNKNOWN for item in expected_actions
        ):
            raise ContractViolation(f"unknown path {path_id} has no unknown action")
        definitions.append(
            BehavioralPathDefinition(
                path_id=path_id,
                expected_actions=tuple(expected_actions),
                expected_turn_count=2 if requires_confirmation else 1,
                expected_confirmation_state=(
                    ConfirmationState.OBSERVED
                    if requires_confirmation
                    else ConfirmationState.CONFIRMED_ABSENT
                ),
                expected_confirmation_token=(
                    confirmation_token if requires_confirmation else None
                ),
                expected_final_output=final_output,
                semantic_status=semantic_status,
                policy_verifiable=policy_verifiable,
                expected_policy_result=expected_result,
            )
        )
    target_relative_path = raw_binding.get("target_relative_path")
    entrypoint = raw_binding.get("entrypoint")
    if not isinstance(target_relative_path, str) or not isinstance(entrypoint, str):
        raise ContractViolation("semantic source binding is malformed")
    return FrozenBehavioralGroundTruth(
        source_digest=source_digest,
        target_relative_path=target_relative_path,
        entrypoint=entrypoint,
        target_file_set=target_file_set,
        paths=tuple(definitions),
    )


def materialize_unexecuted_cases(
    catalog: FrozenCatalog,
) -> tuple[AttackMeasurement, ...]:
    return tuple(
        AttackMeasurement(
            attack_id=definition.attack_id,
            case_type=definition.case_type,
            category=definition.category,
            technique=definition.technique,
            security_property_under_test=definition.security_property_under_test,
            expected_outcome=definition.expected_outcome,
            observability_requirement=definition.observability_requirement,
            subattempts=tuple(
                SubattemptMeasurement(subattempt_id=item)
                for item in definition.subattempt_ids
            ),
            postcondition=TrustedPostcondition(
                sentinel_id=definition.mutation_sentinel
            ),
            effects=tuple(
                EffectMeasurement(kind=item.kind)
                for item in definition.effect_checks
            ),
            limitations=definition.limitations,
        )
        for definition in catalog.cases
    )


def materialize_unexercised_paths(
    ground_truth: FrozenBehavioralGroundTruth,
) -> tuple[BehavioralPathMeasurement, ...]:
    return tuple(
        BehavioralPathMeasurement(
            path_id=definition.path_id,
            attempted=False,
            exercised=False,
            actions=(),
            turn_count=None,
            confirmation_state=None,
            confirmation_token=None,
            final_output=None,
            semantic_status=definition.semantic_status,
            policy_result=PolicyResult.INCONCLUSIVE,
            policy_verifiable=False,
        )
        for definition in ground_truth.paths
    )


def sha256_file(path: Path) -> str:
    """Hash a non-target control document; target identity uses SourceFileSet."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ActionMeasurement",
    "AttackMeasurement",
    "AttackOutcome",
    "BehavioralMeasurement",
    "BehavioralMetrics",
    "BehavioralPathDefinition",
    "BehavioralPathMeasurement",
    "CaseType",
    "CatalogCaseDefinition",
    "CatalogEffectDefinition",
    "ConfirmationState",
    "ContainmentMeasurement",
    "ContainmentMetrics",
    "ContractViolation",
    "DiagnosticChannel",
    "EvidenceReference",
    "EvidenceScope",
    "EvidenceSubjectType",
    "EffectKind",
    "EffectMeasurement",
    "FrozenBehavioralGroundTruth",
    "FrozenCatalog",
    "FROZEN_CATALOG_SHA256",
    "FROZEN_GROUND_TRUTH_SHA256",
    "GroundTruthExpectedAction",
    "InfrastructureFailureStage",
    "InfrastructureStatus",
    "MEASUREMENT_SCHEMA_VERSION",
    "PolicyResult",
    "RunProvenance",
    "SemanticStatus",
    "StaticDiscovery",
    "SubattemptMeasurement",
    "SubattemptDisposition",
    "TrialCompletion",
    "TrustedEvidenceKind",
    "TrustedPostcondition",
    "UntrustedDiagnostic",
    "bind_container_exit_status",
    "bind_evidence_bytes",
    "bind_evidence_file",
    "canonical_json",
    "load_frozen_catalog",
    "load_frozen_ground_truth",
    "load_json_object",
    "materialize_unexecuted_cases",
    "materialize_unexercised_paths",
    "sha256_file",
    "verify_evidence_file",
]
