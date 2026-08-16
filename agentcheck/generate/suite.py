"""Frozen suite documents: a reviewable, integrity-checked suite on disk.

A frozen suite is *input* to evaluation, not a disclosure artifact, so it is
integrity-bearing: the document carries a fingerprint over its own contents and
every embedded scenario re-verifies its own fingerprint on load.  For that reason
the document is never rewritten by redaction on the way out -- a transform would
invalidate the very fingerprints the contract exists to protect, which is the
same correctness problem that keeps replay out of Phase 2.  Instead every value
is screened before the suite is built, so nothing redaction would touch can enter
it.

Nothing in the document is executable.  Loading performs no import, resolves no
schema reference, and accepts no field the contract does not declare.
"""

from __future__ import annotations

import hmac
import json
import os
from enum import Enum
from pathlib import Path
from typing import Literal, Sequence

from pydantic import Field, model_validator

from agentcheck import __version__
from agentcheck.artifacts import create_private_file, replace_private_file
from agentcheck.config import AgentCheckConfig, contained_path
from agentcheck.domain import AgentSpec, ContractModel, Scenario, canonical_hash
from agentcheck.errors import ConfigurationError, ScenarioValidationError
from agentcheck.privacy import redact_log_text

from .boundaries import build_boundary_cases, unsupported_boundary_reasons
from .lint import ScenarioLintIssue, lint_suite
from .templates import build_account_support_suite


FROZEN_SUITE_CONTRACT_VERSION: Literal["agentcheck.frozen_suite.v1"] = (
    "agentcheck.frozen_suite.v1"
)
DEFAULT_SUITE_FILENAME = "agentcheck-suite.json"
GENERATOR_NAME = "agentcheck.generate.suite"

MAX_UNSUPPORTED_FEATURES = 100
_MAX_SUITE_BYTES = 8 * 1024 * 1024


class CaseOrigin(str, Enum):
    BUILT_IN = "built_in"
    SCHEMA_BOUNDARY = "schema_boundary"


class CaseLineage(ContractModel):
    """Where a case came from; lineage lives here so ``Scenario`` stays at v1."""

    origin: CaseOrigin
    tool_name: str | None = Field(default=None, max_length=200)
    boundary_kind: str | None = Field(default=None, max_length=100)
    schema_pointer: str | None = Field(default=None, max_length=1_000)


class FrozenCase(ContractModel):
    scenario: Scenario
    lineage: CaseLineage


class LintIssueRecord(ContractModel):
    code: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=4_000)
    severity: str = Field(min_length=1, max_length=50)


class RejectedCase(ContractModel):
    """A candidate excluded from scoring, kept with the reason it was excluded."""

    scenario: Scenario
    lineage: CaseLineage
    issues: tuple[LintIssueRecord, ...] = Field(min_length=1)


class GeneratorProvenance(ContractModel):
    generator: str = Field(min_length=1, max_length=200)
    generator_version: str = Field(min_length=1, max_length=100)
    sources: tuple[str, ...] = Field(min_length=1)


class SuiteCoverage(ContractModel):
    tools: tuple[str, ...] = ()
    tools_without_boundary_cases: tuple[str, ...] = ()
    boundary_kinds: tuple[str, ...] = ()
    unsupported_schema_features: tuple[str, ...] = ()


class FrozenSuite(ContractModel):
    """A deterministic, fingerprinted suite document."""

    schema_version: Literal["agentcheck.frozen_suite.v1"] = (
        FROZEN_SUITE_CONTRACT_VERSION
    )
    suite_id: str = Field(default="", max_length=200)
    spec_id: str = Field(min_length=1, max_length=200)
    seed: int = Field(ge=0, le=2**63 - 1)
    provenance: GeneratorProvenance
    coverage: SuiteCoverage = Field(default_factory=SuiteCoverage)
    cases: tuple[FrozenCase, ...] = Field(min_length=1)
    rejected: tuple[RejectedCase, ...] = ()
    fingerprint: str = ""

    @property
    def scenarios(self) -> tuple[Scenario, ...]:
        return tuple(case.scenario for case in self.cases)

    def expected_fingerprint(self) -> str:
        # Identity covers every behavioral field.  There is deliberately no
        # timestamp, host path, or random identifier, so the same target, config,
        # and seed always freeze byte-identical documents.
        return canonical_hash(
            self.model_dump(mode="json", exclude={"fingerprint", "suite_id"})
        )

    def expected_suite_id(self) -> str:
        return f"frozensuite-{self.expected_fingerprint().split(':', 1)[1][:24]}"

    @model_validator(mode="after")
    def validate_identity_and_uniqueness(self) -> "FrozenSuite":
        scenario_ids = [case.scenario.scenario_id for case in self.cases]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("frozen suite scenario IDs must be unique")
        fingerprints = [case.scenario.fingerprint for case in self.cases]
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("frozen suite scenarios must be deduplicated")

        expected = self.expected_fingerprint()
        if self.fingerprint and not _digest_equal(self.fingerprint, expected):
            raise ValueError("frozen suite fingerprint does not match its contents")
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", expected)
        expected_id = self.expected_suite_id()
        if self.suite_id and not _digest_equal(self.suite_id, expected_id):
            raise ValueError("frozen suite ID does not match its contents")
        if not self.suite_id:
            object.__setattr__(self, "suite_id", expected_id)
        return self


def _digest_equal(left: str, right: str) -> bool:
    """Constant-time compare that still returns False on unequal lengths."""

    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    if len(left_bytes) != len(right_bytes):
        hmac.compare_digest(left_bytes, left_bytes)
        return False
    return hmac.compare_digest(left_bytes, right_bytes)


def built_in_suite(config: AgentCheckConfig, seed: int) -> tuple[Scenario, ...]:
    """Return the configured built-in suite for a seed."""

    if config.suite == "account_support_v1":
        return build_account_support_suite(seed=seed)
    raise ValueError(f"unsupported suite: {config.suite}")


def _assert_no_secret_shaped_values(value: object, *, path: str = "$") -> None:
    """Refuse to freeze any value that credential redaction would rewrite.

    The document cannot be redacted on the way out without invalidating its
    fingerprints, so anything redaction would touch is rejected at build time
    instead.  Keys are exempt: a tool may legitimately declare a parameter named
    ``token`` without that parameter's synthetic value being a secret.
    """

    if isinstance(value, str):
        if redact_log_text(value) != value:
            raise ScenarioValidationError(
                f"refusing to freeze credential-shaped content at {path}"
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_secret_shaped_values(item, path=f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_no_secret_shaped_values(item, path=f"{path}[{index}]")


def _issue_records(issues: Sequence[ScenarioLintIssue]) -> tuple[LintIssueRecord, ...]:
    return tuple(
        LintIssueRecord(code=issue.code, message=issue.message, severity=issue.severity)
        for issue in issues
    )


def build_frozen_suite(
    spec: AgentSpec, config: AgentCheckConfig, *, seed: int
) -> FrozenSuite:
    """Derive, deduplicate, lint, and freeze every supported case for a target."""

    candidates: list[tuple[Scenario, CaseLineage]] = [
        (scenario, CaseLineage(origin=CaseOrigin.BUILT_IN))
        for scenario in built_in_suite(config, seed)
    ]
    boundary_tools: set[str] = set()
    for boundary, scenario in build_boundary_cases(spec, seed=seed):
        boundary_tools.add(boundary.tool_name)
        candidates.append(
            (
                scenario,
                CaseLineage(
                    origin=CaseOrigin.SCHEMA_BOUNDARY,
                    tool_name=boundary.tool_name,
                    boundary_kind=boundary.kind.value,
                    schema_pointer=boundary.pointer,
                ),
            )
        )

    unique: list[tuple[Scenario, CaseLineage]] = []
    rejected: list[RejectedCase] = []
    seen: set[str] = set()
    for scenario, lineage in candidates:
        if scenario.fingerprint in seen:
            rejected.append(
                RejectedCase(
                    scenario=scenario,
                    lineage=lineage,
                    issues=(
                        LintIssueRecord(
                            code="duplicate_scenario_fingerprint",
                            message=(
                                "a structurally identical case was already frozen"
                            ),
                            severity="error",
                        ),
                    ),
                )
            )
            continue
        seen.add(scenario.fingerprint)
        unique.append((scenario, lineage))

    lineage_by_id = {scenario.scenario_id: lineage for scenario, lineage in unique}
    cases: list[FrozenCase] = []
    for scenario, issues in lint_suite([item for item, _ in unique], spec):
        lineage = lineage_by_id[scenario.scenario_id]
        if issues:
            rejected.append(
                RejectedCase(
                    scenario=scenario, lineage=lineage, issues=_issue_records(issues)
                )
            )
        else:
            cases.append(FrozenCase(scenario=scenario, lineage=lineage))

    if not cases:
        raise ScenarioValidationError(
            "No valid scenarios remain after linting; refusing to freeze a suite "
            "that cannot produce a verdict."
        )

    tools = tuple(item.value.name for item in spec.tools.items)
    unsupported: list[str] = []
    for item in spec.tools.items:
        unsupported.extend(unsupported_boundary_reasons(item.value))
    coverage = SuiteCoverage(
        tools=tools,
        tools_without_boundary_cases=tuple(
            name for name in tools if name not in boundary_tools
        ),
        boundary_kinds=tuple(
            sorted(
                {
                    case.lineage.boundary_kind
                    for case in cases
                    if case.lineage.boundary_kind is not None
                }
            )
        ),
        unsupported_schema_features=tuple(
            dict.fromkeys(unsupported)
        )[:MAX_UNSUPPORTED_FEATURES],
    )
    suite = FrozenSuite(
        spec_id=spec.spec_id,
        seed=seed,
        provenance=GeneratorProvenance(
            generator=GENERATOR_NAME,
            generator_version=__version__,
            sources=("built_in", "schema_boundary"),
        ),
        coverage=coverage,
        cases=tuple(cases),
        rejected=tuple(rejected),
    )
    _assert_no_secret_shaped_values(suite.model_dump(mode="json"))
    return suite


def encode_frozen_suite(suite: FrozenSuite) -> bytes:
    return (
        json.dumps(
            suite.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def default_suite_path(root: Path, config: AgentCheckConfig) -> Path:
    """Resolve the configured suite location, contained inside the target."""

    return contained_path(root, config.suite_path or DEFAULT_SUITE_FILENAME)


def resolve_suite_destination(
    root: Path, config: AgentCheckConfig, out: str | None
) -> Path:
    if out is None:
        return default_suite_path(root, config)
    return contained_path(root, out)


def write_frozen_suite(destination: Path, suite: FrozenSuite, *, force: bool) -> Path:
    payload = encode_frozen_suite(suite)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if force:
            replace_private_file(destination, payload)
        else:
            create_private_file(destination, payload)
    except FileExistsError as exc:
        raise ConfigurationError(
            f"{destination.name} already exists at {destination}; "
            "re-run with --force to replace it"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(f"unable to write {destination}: {exc}") from exc
    return destination


def load_frozen_suite(path: Path) -> FrozenSuite:
    """Load a frozen suite as untrusted input, failing closed on anything unexpected."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfigurationError(f"unable to read {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read(_MAX_SUITE_BYTES + 1)
    except OSError as exc:
        raise ConfigurationError(f"unable to read {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > _MAX_SUITE_BYTES:
        raise ConfigurationError(
            f"{path.name} exceeds the {_MAX_SUITE_BYTES} byte frozen-suite limit"
        )
    try:
        text = raw.decode("utf-8")
        document = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid frozen suite {path.name}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError(
            f"invalid frozen suite {path.name}: the document must be a JSON object"
        )
    declared = document.get("schema_version")
    if declared != FROZEN_SUITE_CONTRACT_VERSION:
        raise ConfigurationError(
            f"unsupported frozen suite contract {declared!r}; this build reads "
            f"{FROZEN_SUITE_CONTRACT_VERSION}"
        )
    try:
        return FrozenSuite.model_validate_json(text)
    except ValueError as exc:
        raise ConfigurationError(f"invalid frozen suite {path.name}: {exc}") from exc


def configured_frozen_suite(
    root: Path, config: AgentCheckConfig
) -> tuple[Path, FrozenSuite] | None:
    """Return the frozen suite a run should use, if one is configured or present.

    An explicitly configured ``suite_path`` must exist; the default filename is
    used only when it happens to be there, so Phase 1 targets keep their built-in
    behavior.
    """

    path = default_suite_path(root, config)
    if not path.is_file():
        if config.suite_path is not None:
            raise ConfigurationError(f"configured suite_path does not exist: {path}")
        return None
    return path, load_frozen_suite(path)


__all__ = [
    "DEFAULT_SUITE_FILENAME",
    "FROZEN_SUITE_CONTRACT_VERSION",
    "GENERATOR_NAME",
    "CaseLineage",
    "CaseOrigin",
    "FrozenCase",
    "FrozenSuite",
    "GeneratorProvenance",
    "LintIssueRecord",
    "RejectedCase",
    "SuiteCoverage",
    "build_frozen_suite",
    "built_in_suite",
    "configured_frozen_suite",
    "default_suite_path",
    "encode_frozen_suite",
    "load_frozen_suite",
    "resolve_suite_destination",
    "write_frozen_suite",
]
