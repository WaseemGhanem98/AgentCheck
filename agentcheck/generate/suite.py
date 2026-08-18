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
from typing import Any, Literal, Sequence

from pydantic import Field, model_serializer, model_validator

from agentcheck import __version__
from agentcheck.artifacts import create_private_file, replace_private_file
from agentcheck.config import AgentCheckConfig, contained_path
from agentcheck.domain import (
    AgentSpec,
    ContractModel,
    JsonObject,
    Scenario,
    canonical_hash,
)
from agentcheck.errors import ConfigurationError, ScenarioValidationError
from agentcheck.policies import PolicyPack, apply_policy_packs
from agentcheck.privacy import redact_log_text

from .boundaries import (
    build_boundary_cases,
    build_output_schema_cases,
    build_zero_input_cases,
    unsupported_boundary_reasons,
)
from .lint import ScenarioLintIssue, lint_scenario, lint_suite
from .mutations import (
    DEFAULT_MAX_MUTATIONS,
    MAX_MUTATIONS_PER_SUITE,
    MutationKind,
    build_workflow_mutations,
)
from .realization import (
    RealizationRecord,
    realize_scenarios,
    realization_settings,
)
from .selection import (
    SelectionPlan,
    lineage_coverage_tags,
    select_scenarios,
)
from .templates import (
    build_account_support_suite,
    declared_tool_names,
    empty_generation_message,
    spec_matches_built_in_suite,
)


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
    WORKFLOW_MUTATION = "workflow_mutation"
    ZERO_INPUT_INVOCATION = "zero_input_invocation"
    OUTPUT_SCHEMA = "output_schema"


_MUTATION_LINEAGE_FIELDS = (
    "parent_scenario_id",
    "parent_fingerprint",
    "mutation_kind",
    "mutation_parameters",
    "mutation_rationale",
)


class CaseLineage(ContractModel):
    """Where a case came from; lineage lives here so ``Scenario`` stays at v1."""

    origin: CaseOrigin
    tool_name: str | None = Field(default=None, max_length=200)
    boundary_kind: str | None = Field(default=None, max_length=100)
    schema_pointer: str | None = Field(default=None, max_length=1_000)
    parent_scenario_id: str | None = Field(default=None, max_length=200)
    parent_fingerprint: str | None = Field(default=None, max_length=200)
    mutation_kind: str | None = Field(default=None, max_length=100)
    mutation_parameters: JsonObject = Field(default_factory=dict)
    mutation_rationale: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def validate_mutation_lineage(self) -> "CaseLineage":
        if self.origin is CaseOrigin.WORKFLOW_MUTATION:
            if not (
                self.parent_scenario_id
                and self.parent_fingerprint
                and self.mutation_kind
                and self.mutation_rationale
            ):
                raise ValueError(
                    "workflow mutation lineage requires parent identity, kind, and rationale"
                )
            try:
                MutationKind(self.mutation_kind)
            except ValueError as exc:
                raise ValueError(
                    f"unsupported mutation kind {self.mutation_kind!r}"
                ) from exc
            return self
        if (
            self.parent_scenario_id
            or self.parent_fingerprint
            or self.mutation_kind
            or self.mutation_rationale
            or self.mutation_parameters
        ):
            raise ValueError(
                "mutation lineage fields are only valid for workflow_mutation origin"
            )
        if self.origin is CaseOrigin.ZERO_INPUT_INVOCATION and not self.tool_name:
            raise ValueError("zero_input_invocation lineage requires tool_name")
        return self

    @model_serializer(mode="wrap")
    def drop_idle_mutation_fields(self, serializer: Any) -> dict[str, Any]:
        data = serializer(self)
        origin = data.get("origin")
        if origin not in {
            CaseOrigin.WORKFLOW_MUTATION,
            CaseOrigin.WORKFLOW_MUTATION.value,
        }:
            for key in _MUTATION_LINEAGE_FIELDS:
                data.pop(key, None)
        return data


class FrozenCase(ContractModel):
    scenario: Scenario
    lineage: CaseLineage
    realization: RealizationRecord | None = None

    @model_serializer(mode="wrap")
    def omit_idle_realization(self, serializer: Any) -> dict[str, Any]:
        data = serializer(self)
        if not data.get("realization"):
            data.pop("realization", None)
        return data


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
    policy_packs: tuple[str, ...] = ()

    @model_serializer(mode="wrap")
    def omit_idle_policy_packs(self, serializer: Any) -> dict[str, Any]:
        data = serializer(self)
        if not data.get("policy_packs"):
            data.pop("policy_packs", None)
        return data


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
    selection: SelectionPlan | None = None
    fingerprint: str = ""

    @model_serializer(mode="wrap")
    def omit_idle_selection(self, serializer: Any) -> dict[str, Any]:
        data = serializer(self)
        if not data.get("selection"):
            data.pop("selection", None)
        return data

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


def _mutation_limit(max_mutations: int | None) -> int:
    limit = DEFAULT_MAX_MUTATIONS if max_mutations is None else max_mutations
    if limit < 1 or limit > MAX_MUTATIONS_PER_SUITE:
        raise ValueError(
            f"max_mutations must be between 1 and {MAX_MUTATIONS_PER_SUITE}"
        )
    return limit


def _select_frozen_cases(
    cases: list[FrozenCase],
    spec: AgentSpec,
    *,
    max_cases: int,
) -> tuple[list[FrozenCase], SelectionPlan]:
    extra_tags = {
        case.scenario.scenario_id: lineage_coverage_tags(
            origin=case.lineage.origin.value,
            tool_name=case.lineage.tool_name,
            boundary_kind=case.lineage.boundary_kind,
            mutation_kind=case.lineage.mutation_kind,
        )
        for case in cases
    }
    selected, plan = select_scenarios(
        [case.scenario for case in cases],
        max_cases=max_cases,
        spec=spec,
        extra_tags_by_id=extra_tags,
    )
    by_id = {case.scenario.scenario_id: case for case in cases}
    return [by_id[scenario.scenario_id] for scenario in selected], plan


def _realize_frozen_cases(
    cases: list[FrozenCase],
    config: AgentCheckConfig,
    realizer: Any,
) -> list[FrozenCase]:
    model, max_calls, _max_retries = realization_settings(config)
    realized = realize_scenarios(
        [case.scenario for case in cases],
        realizer,
        provider="openai",
        model=model,
        max_calls=max_calls,
    )
    updated: list[FrozenCase] = []
    for case, (scenario, record) in zip(cases, realized, strict=True):
        updated.append(
            FrozenCase(scenario=scenario, lineage=case.lineage, realization=record)
        )
    return updated


def _append_unique(
    scenario: Scenario,
    lineage: CaseLineage,
    *,
    seen: set[str],
    unique: list[tuple[Scenario, CaseLineage]],
    rejected: list[RejectedCase],
) -> None:
    if scenario.fingerprint in seen:
        rejected.append(
            RejectedCase(
                scenario=scenario,
                lineage=lineage,
                issues=(
                    LintIssueRecord(
                        code="duplicate_scenario_fingerprint",
                        message="a structurally identical case was already frozen",
                        severity="error",
                    ),
                ),
            )
        )
        return
    seen.add(scenario.fingerprint)
    unique.append((scenario, lineage))


def build_frozen_suite(
    spec: AgentSpec,
    config: AgentCheckConfig,
    *,
    seed: int,
    include_mutations: bool = False,
    max_mutations: int | None = None,
    policy_packs: Sequence[PolicyPack] = (),
    max_cases: int | None = None,
    realizer: Any | None = None,
) -> FrozenSuite:
    """Derive, deduplicate, lint, and freeze every supported case for a target."""

    if max_mutations is not None and not include_mutations:
        raise ValueError("max_mutations requires include_mutations=True")
    budget = config.max_cases if max_cases is None else max_cases

    candidates: list[tuple[Scenario, CaseLineage]] = []
    if spec_matches_built_in_suite(spec, config.suite):
        candidates.extend(
            (scenario, CaseLineage(origin=CaseOrigin.BUILT_IN))
            for scenario in built_in_suite(config, seed)
        )
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
    output_schema_cases = build_output_schema_cases(spec, seed=seed)
    for scenario in output_schema_cases:
        candidates.append((scenario, CaseLineage(origin=CaseOrigin.OUTPUT_SCHEMA)))
    zero_input_tools: set[str] = set()
    for tool_name, scenario in build_zero_input_cases(spec, seed=seed):
        zero_input_tools.add(tool_name)
        candidates.append(
            (
                scenario,
                CaseLineage(
                    origin=CaseOrigin.ZERO_INPUT_INVOCATION,
                    tool_name=tool_name,
                ),
            )
        )

    unique: list[tuple[Scenario, CaseLineage]] = []
    rejected: list[RejectedCase] = []
    seen: set[str] = set()
    for scenario, lineage in candidates:
        _append_unique(
            scenario, lineage, seen=seen, unique=unique, rejected=rejected
        )

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

    if include_mutations:
        parents = tuple(
            case.scenario
            for case in cases
            if case.lineage.origin is CaseOrigin.BUILT_IN
        )
        for generated in build_workflow_mutations(
            parents, seed=seed, max_mutations=_mutation_limit(max_mutations)
        ):
            lineage = CaseLineage(
                origin=CaseOrigin.WORKFLOW_MUTATION,
                parent_scenario_id=generated.mutation.parent_scenario_id,
                parent_fingerprint=generated.mutation.parent_fingerprint,
                mutation_kind=generated.mutation.kind.value,
                mutation_parameters=generated.mutation.parameters,
                mutation_rationale=generated.mutation.rationale,
            )
            if generated.scenario.fingerprint in seen:
                rejected.append(
                    RejectedCase(
                        scenario=generated.scenario,
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
            issues = lint_scenario(generated.scenario, spec)
            if issues:
                rejected.append(
                    RejectedCase(
                        scenario=generated.scenario,
                        lineage=lineage,
                        issues=_issue_records(issues),
                    )
                )
                continue
            seen.add(generated.scenario.fingerprint)
            cases.append(FrozenCase(scenario=generated.scenario, lineage=lineage))

    if policy_packs:
        applied_cases: list[FrozenCase] = []
        applied_seen: set[str] = set()
        for case in cases:
            scenario = apply_policy_packs(
                case.scenario, policy_packs, declared=True
            )
            if scenario.fingerprint in applied_seen:
                rejected.append(
                    RejectedCase(
                        scenario=scenario,
                        lineage=case.lineage,
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
            if scenario.fingerprint != case.scenario.fingerprint:
                issues = lint_scenario(scenario, spec)
                if issues:
                    rejected.append(
                        RejectedCase(
                            scenario=scenario,
                            lineage=case.lineage,
                            issues=_issue_records(issues),
                        )
                    )
                    continue
            applied_seen.add(scenario.fingerprint)
            applied_cases.append(FrozenCase(scenario=scenario, lineage=case.lineage))
        cases = applied_cases

    if not cases:
        if not declared_tool_names(spec) and not spec_matches_built_in_suite(
            spec, config.suite
        ):
            raise ScenarioValidationError(empty_generation_message(spec, config.suite))
        raise ScenarioValidationError(
            "No valid scenarios remain after linting; refusing to freeze a suite "
            "that cannot produce a verdict."
        )

    selection: SelectionPlan | None = None
    if budget is not None:
        cases, selection = _select_frozen_cases(cases, spec, max_cases=budget)

    if realizer is not None:
        cases = _realize_frozen_cases(cases, config, realizer)

    tools = tuple(item.value.name for item in spec.tools.items)
    unsupported: list[str] = []
    for item in spec.tools.items:
        unsupported.extend(unsupported_boundary_reasons(item.value))
    sources: tuple[str, ...] = ("built_in", "schema_boundary")
    if any(
        case.lineage.origin is CaseOrigin.ZERO_INPUT_INVOCATION for case in cases
    ):
        sources = (*sources, "zero_input_invocation")
    if any(case.lineage.origin is CaseOrigin.OUTPUT_SCHEMA for case in cases):
        sources = (*sources, "output_schema")
    if include_mutations:
        sources = (*sources, "workflow_mutation")
    if realizer is not None and any(case.realization is not None for case in cases):
        sources = (*sources, "llm_realization")
    coverage = SuiteCoverage(
        tools=tools,
        tools_without_boundary_cases=tuple(
            name
            for name in tools
            if name not in boundary_tools and name not in zero_input_tools
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
            sources=sources,
            policy_packs=tuple(pack.pack_id for pack in policy_packs),
        ),
        coverage=coverage,
        cases=tuple(cases),
        rejected=tuple(rejected),
        selection=selection,
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
    "SelectionPlan",
    "SuiteCoverage",
    "build_frozen_suite",
    "built_in_suite",
    "configured_frozen_suite",
    "default_suite_path",
    "encode_frozen_suite",
    "load_frozen_suite",
    "resolve_suite_destination",
    "spec_matches_built_in_suite",
    "write_frozen_suite",
]
