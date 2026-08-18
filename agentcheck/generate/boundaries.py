"""Deterministic schema-boundary cases derived from declared tool contracts.

A tool's JSON Schema is an authoritative contract, so a call that violates it is
a contract breach rather than an inference, and the resulting scenarios may
support hard failures.  Every synthetic value is *proved* out of contract by the
offline validator before a case is emitted, so a boundary can never assert
something the schema actually permits.

Generation reads the normalized argument surface produced by capability
extraction rather than parsing schemas a second time.  It imports nothing,
executes nothing, retrieves no reference outside the tool's own document, and
uses no randomness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping

from pydantic import Field

from agentcheck.domain import (
    AgentSpec,
    ContractModel,
    ConversationRole,
    ConversationTurn,
    JsonObject,
    JsonValue,
    OracleProvenance,
    OracleStrength,
    OutputCriterion,
    OutputCriterionKind,
    ResourceBudgets,
    Scenario,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolBehaviorConstraint,
    ToolDefinition,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
    ToolFixture,
    canonical_hash,
)
from agentcheck.inspect.capabilities import (
    CapabilityParameter,
    ExtractedCapability,
    JsonSchemaType,
    extract_capabilities,
)
from agentcheck.schema_safety import offline_validator


BOUNDARY_CONTRACT_VERSION: Literal["agentcheck.schema_boundary.v1"] = (
    "agentcheck.schema_boundary.v1"
)

MAX_BOUNDARIES_PER_PARAMETER = 4
MAX_BOUNDARIES_PER_TOOL = 12
MAX_BOUNDARY_SCENARIOS_PER_SPEC = 48

# Obviously synthetic: a generated case must never carry a plausible credential,
# address, or account identifier into a fixture or a transcript.
_SYNTHETIC_STRING = "agentcheck-boundary-value"
_SYNTHETIC_FILLER = "x"
_OUT_OF_ENUM_STRING = "agentcheck-out-of-enum"
_UNEXPECTED_PROPERTY = "agentcheck_unexpected_property"

_MAX_SCENARIO_ID = 150
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Keywords the extractor reports but this slice deliberately does not invert.
_UNSUPPORTED_KEYWORDS = (
    ("pattern", "pattern"),
    ("string_format", "format"),
    ("multiple_of", "multipleOf"),
    ("min_items", "minItems"),
    ("max_items", "maxItems"),
    ("unique_items", "uniqueItems"),
)


class BoundaryKind(str, Enum):
    """Ordered so truncation is deterministic and reviewable."""

    MISSING_REQUIRED_PROPERTY = "missing_required_property"
    WRONG_TYPE = "wrong_type"
    OUT_OF_ENUM = "out_of_enum"
    BELOW_MINIMUM = "below_minimum"
    ABOVE_MAXIMUM = "above_maximum"
    BELOW_MIN_LENGTH = "below_min_length"
    ABOVE_MAX_LENGTH = "above_max_length"
    ADDITIONAL_PROPERTY = "additional_property"


class SchemaBoundary(ContractModel):
    """One proved-invalid argument object traceable to a single schema constraint."""

    schema_version: Literal["agentcheck.schema_boundary.v1"] = BOUNDARY_CONTRACT_VERSION
    tool_name: str = Field(min_length=1, max_length=200)
    parameter: str = Field(min_length=1, max_length=200)
    pointer: str = Field(min_length=1, max_length=1_000)
    kind: BoundaryKind
    invalid_value: JsonValue = None
    value_omitted: bool = False
    arguments: JsonObject
    baseline_arguments: JsonObject
    rationale: str = Field(min_length=1, max_length=2_000)
    evidence_ids: tuple[str, ...] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class _Analysis:
    boundaries: tuple[SchemaBoundary, ...]
    reasons: tuple[str, ...]
    # The in-contract argument object the boundaries were derived from. It is
    # already proved valid by the offline validator, so positive-path
    # generation reuses it rather than deriving schema values a second time.
    baseline: JsonObject | None = None


def _slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.casefold()).strip("-") or "value"


def _scenario_id(tool_name: str, parameter: str, kind: BoundaryKind) -> str:
    candidate = f"boundary-{_slug(tool_name)}-{_slug(parameter)}-{_slug(kind.value)}"
    if len(candidate) <= _MAX_SCENARIO_ID:
        return candidate
    digest = canonical_hash(candidate).split(":", 1)[1][:16]
    return f"{candidate[: _MAX_SCENARIO_ID - 17]}-{digest}"


def _preferred_type(parameter: CapabilityParameter) -> JsonSchemaType | None:
    for candidate in (
        JsonSchemaType.STRING,
        JsonSchemaType.INTEGER,
        JsonSchemaType.NUMBER,
        JsonSchemaType.BOOLEAN,
        JsonSchemaType.ARRAY,
        JsonSchemaType.OBJECT,
        JsonSchemaType.NULL,
    ):
        if candidate in parameter.types:
            return candidate
    return None


def _bounded_string(parameter: CapabilityParameter) -> str | None:
    minimum = parameter.constraints.min_length or 0
    maximum = parameter.constraints.max_length
    value = _SYNTHETIC_STRING
    if maximum is not None:
        if maximum < minimum:
            return None
        value = value[:maximum]
    if len(value) < minimum:
        value = (value + _SYNTHETIC_FILLER * minimum)[:minimum]
    if maximum is not None and len(value) > maximum:
        return None
    return value


def _numeric_bounds(parameter: CapabilityParameter) -> tuple[float | None, float | None]:
    constraints = parameter.constraints
    lower = constraints.minimum
    if lower is None and constraints.exclusive_minimum is not None:
        lower = constraints.exclusive_minimum + 1
    upper = constraints.maximum
    if upper is None and constraints.exclusive_maximum is not None:
        upper = constraints.exclusive_maximum - 1
    return lower, upper


def _baseline_number(parameter: CapabilityParameter, *, integer: bool) -> JsonValue:
    lower, upper = _numeric_bounds(parameter)
    value = lower if lower is not None else (upper if upper is not None else 1.0)
    if lower is not None and value < lower:
        value = lower
    if upper is not None and value > upper:
        value = upper
    multiple = parameter.constraints.multiple_of
    if multiple is not None and multiple > 0:
        value = multiple
    if lower is not None and value < lower:
        return None
    if upper is not None and value > upper:
        return None
    return int(value) if integer else float(value)


def _baseline_value(parameter: CapabilityParameter) -> tuple[bool, JsonValue]:
    """Build a value the declared schema should accept for one parameter."""

    enum_values = parameter.constraints.enum_values
    if enum_values:
        return True, enum_values[0]
    if not parameter.types_known:
        return False, None
    declared = _preferred_type(parameter)
    if declared is JsonSchemaType.STRING:
        value = _bounded_string(parameter)
        return (value is not None), value
    if declared in {JsonSchemaType.INTEGER, JsonSchemaType.NUMBER}:
        number = _baseline_number(
            parameter, integer=declared is JsonSchemaType.INTEGER
        )
        return (number is not None), number
    if declared is JsonSchemaType.BOOLEAN:
        return True, False
    if declared is JsonSchemaType.ARRAY:
        minimum = parameter.constraints.min_items or 0
        if minimum == 0:
            return True, []
        item_types = parameter.constraints.item_types
        if not item_types or parameter.constraints.unique_items:
            return False, None
        filler: JsonValue = (
            _SYNTHETIC_STRING if JsonSchemaType.STRING in item_types else 1
        )
        return True, [filler] * minimum
    if declared is JsonSchemaType.OBJECT:
        return True, {}
    if declared is JsonSchemaType.NULL:
        return True, None
    return False, None


def _wrong_type_candidates() -> tuple[tuple[JsonSchemaType, JsonValue], ...]:
    return (
        (JsonSchemaType.STRING, _SYNTHETIC_STRING),
        (JsonSchemaType.INTEGER, 12_345),
        (JsonSchemaType.BOOLEAN, True),
        (JsonSchemaType.ARRAY, []),
        (JsonSchemaType.OBJECT, {}),
    )


def _wrong_type_value(parameter: CapabilityParameter) -> JsonValue | None:
    declared = set(parameter.types)
    for candidate_type, value in _wrong_type_candidates():
        if candidate_type in declared:
            continue
        if candidate_type is JsonSchemaType.INTEGER and JsonSchemaType.NUMBER in declared:
            continue
        if candidate_type is JsonSchemaType.BOOLEAN and (
            declared & {JsonSchemaType.INTEGER, JsonSchemaType.NUMBER}
        ):
            # A JSON boolean is not a number, but several validators and many
            # schemas treat the pair loosely; the proof below is authoritative.
            continue
        return value
    return None


def _out_of_enum_value(parameter: CapabilityParameter) -> JsonValue:
    values = parameter.constraints.enum_values or ()
    numeric = [
        float(item)
        for item in values
        if isinstance(item, int | float) and not isinstance(item, bool)
    ]
    if numeric and len(numeric) == len(values):
        return int(max(numeric)) + 1
    return _OUT_OF_ENUM_STRING


def _candidate_values(
    parameter: CapabilityParameter,
) -> tuple[tuple[BoundaryKind, JsonValue], ...]:
    """Enumerate one perturbation per supported keyword, in enum order."""

    constraints = parameter.constraints
    candidates: list[tuple[BoundaryKind, JsonValue]] = []
    if parameter.types_known:
        wrong_type = _wrong_type_value(parameter)
        if wrong_type is not None:
            candidates.append((BoundaryKind.WRONG_TYPE, wrong_type))
    if constraints.enum_values:
        candidates.append((BoundaryKind.OUT_OF_ENUM, _out_of_enum_value(parameter)))
    integer = JsonSchemaType.INTEGER in parameter.types
    lower, upper = _numeric_bounds(parameter)
    if lower is not None:
        below = lower - 1
        candidates.append(
            (BoundaryKind.BELOW_MINIMUM, int(below) if integer else below)
        )
    if upper is not None:
        above = upper + 1
        candidates.append(
            (BoundaryKind.ABOVE_MAXIMUM, int(above) if integer else above)
        )
    if constraints.min_length is not None and constraints.min_length > 0:
        candidates.append(
            (
                BoundaryKind.BELOW_MIN_LENGTH,
                _SYNTHETIC_FILLER * (constraints.min_length - 1),
            )
        )
    if constraints.max_length is not None:
        candidates.append(
            (
                BoundaryKind.ABOVE_MAX_LENGTH,
                _SYNTHETIC_FILLER * (constraints.max_length + 1),
            )
        )
    return tuple(candidates)


def _rationale(parameter_name: str, kind: BoundaryKind, value: JsonValue) -> str:
    rendered = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    if kind is BoundaryKind.MISSING_REQUIRED_PROPERTY:
        return (
            f"The declared schema requires {parameter_name!r}, so an invocation that "
            "omits it breaks the tool contract."
        )
    if kind is BoundaryKind.ADDITIONAL_PROPERTY:
        return (
            f"The declared schema rejects additional properties, so sending "
            f"{parameter_name!r} breaks the tool contract."
        )
    reason = {
        BoundaryKind.WRONG_TYPE: "declares a different type for",
        BoundaryKind.OUT_OF_ENUM: "enumerates the permitted values of",
        BoundaryKind.BELOW_MINIMUM: "declares a minimum for",
        BoundaryKind.ABOVE_MAXIMUM: "declares a maximum for",
        BoundaryKind.BELOW_MIN_LENGTH: "declares a minimum length for",
        BoundaryKind.ABOVE_MAX_LENGTH: "declares a maximum length for",
    }[kind]
    return (
        f"The declared schema {reason} {parameter_name!r}, so the value {rendered} "
        "breaks the tool contract."
    )


def _evidence_lookup(extracted: ExtractedCapability) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for item in extracted.evidence:
        if item.locator is not None:
            lookup.setdefault(item.locator, item.evidence_id)
    return lookup


def _evidence_ids(
    lookup: Mapping[str, str], tool_name: str, pointer: str | None
) -> tuple[str, ...]:
    schema_locator = f"tool:{tool_name}.input_schema"
    keys = [] if pointer is None else [f"{schema_locator}#{pointer}"]
    keys.extend((schema_locator, f"tool:{tool_name}"))
    found = tuple(
        dict.fromkeys(lookup[key] for key in keys if key in lookup)
    )
    return found or (f"tool-schema:{tool_name}",)


def _unsupported_parameter_reasons(parameter: CapabilityParameter) -> list[str]:
    reasons: list[str] = []
    if parameter.unsupported_constructs:
        reasons.append(
            f"parameter {parameter.name!r} uses unresolved construct(s) "
            + ", ".join(parameter.unsupported_constructs)
        )
    if not parameter.types_known:
        reasons.append(f"parameter {parameter.name!r} has no established type")
    for attribute, keyword in _UNSUPPORTED_KEYWORDS:
        if getattr(parameter.constraints, attribute) is not None:
            reasons.append(
                f"parameter {parameter.name!r} declares {keyword}, which this slice "
                "does not invert"
            )
    if parameter.constraints.nested_property_names:
        reasons.append(
            f"parameter {parameter.name!r} declares nested properties, which are not "
            "traversed"
        )
    return reasons


def _analyze(tool: ToolDefinition) -> _Analysis:
    # Raises UnsafeSchemaReference for a non-local reference and SchemaError for an
    # invalid schema: an unverifiable contract must be refused, never skipped.
    validator = offline_validator(tool.input_schema)

    (extracted,) = extract_capabilities([tool])
    surface = extracted.arguments
    reasons: list[str] = []
    if not surface.schema_known:
        return _Analysis((), ("the declared input schema could not be read",))
    if surface.truncated:
        reasons.append("the declared parameter list was truncated during extraction")
    if not surface.parameters:
        return _Analysis((), (*reasons, "the declared schema exposes no parameters"))

    baseline: dict[str, Any] = {}
    for parameter in surface.parameters:
        if not parameter.required:
            continue
        usable, value = _baseline_value(parameter)
        if not usable:
            return _Analysis(
                (),
                (
                    *reasons,
                    f"no in-contract value could be constructed for required "
                    f"parameter {parameter.name!r}",
                ),
            )
        baseline[parameter.name] = value
    if not validator.is_valid(baseline):
        return _Analysis(
            (),
            (*reasons, "no valid baseline argument object could be constructed"),
        )
    valid_baseline: JsonObject = dict(baseline)

    lookup = _evidence_lookup(extracted)
    boundaries: list[SchemaBoundary] = []
    # The one tool-level constraint leads so per-parameter volume cannot push it
    # past the cap.
    if surface.additional_properties_allowed is False:
        undeclared = {**baseline, _UNEXPECTED_PROPERTY: _SYNTHETIC_STRING}
        if validator.is_valid(undeclared):
            reasons.append(
                "the declared schema accepted an undeclared property despite "
                "additionalProperties being false"
            )
        else:
            boundaries.append(
                SchemaBoundary(
                    tool_name=tool.name,
                    parameter=_UNEXPECTED_PROPERTY,
                    pointer="/additionalProperties",
                    kind=BoundaryKind.ADDITIONAL_PROPERTY,
                    invalid_value=_SYNTHETIC_STRING,
                    arguments=undeclared,
                    baseline_arguments=dict(baseline),
                    rationale=_rationale(
                        _UNEXPECTED_PROPERTY,
                        BoundaryKind.ADDITIONAL_PROPERTY,
                        _SYNTHETIC_STRING,
                    ),
                    evidence_ids=_evidence_ids(lookup, tool.name, None),
                )
            )

    for parameter in surface.parameters:
        reasons.extend(_unsupported_parameter_reasons(parameter))
        accepted = 0
        pending: list[tuple[BoundaryKind, JsonValue, bool]] = []
        if parameter.required:
            pending.append((BoundaryKind.MISSING_REQUIRED_PROPERTY, None, True))
        pending.extend(
            (kind, value, False) for kind, value in _candidate_values(parameter)
        )
        for kind, value, omitted in pending:
            if accepted >= MAX_BOUNDARIES_PER_PARAMETER:
                reasons.append(
                    f"parameter {parameter.name!r} reached the per-parameter cap of "
                    f"{MAX_BOUNDARIES_PER_PARAMETER} case(s)"
                )
                break
            arguments = dict(baseline)
            if omitted:
                arguments.pop(parameter.name, None)
            else:
                arguments[parameter.name] = value
            if validator.is_valid(arguments):
                reasons.append(
                    f"parameter {parameter.name!r} produced no schema violation for "
                    f"{kind.value}"
                )
                continue
            boundaries.append(
                SchemaBoundary(
                    tool_name=tool.name,
                    parameter=parameter.name,
                    pointer=parameter.pointer,
                    kind=kind,
                    invalid_value=None if omitted else value,
                    value_omitted=omitted,
                    arguments=arguments,
                    baseline_arguments=dict(baseline),
                    rationale=_rationale(parameter.name, kind, value),
                    evidence_ids=_evidence_ids(lookup, tool.name, parameter.pointer),
                )
            )
            accepted += 1

    if len(boundaries) > MAX_BOUNDARIES_PER_TOOL:
        reasons.append(
            f"generation stopped at the per-tool cap of {MAX_BOUNDARIES_PER_TOOL} case(s)"
        )
        boundaries = boundaries[:MAX_BOUNDARIES_PER_TOOL]
    return _Analysis(tuple(boundaries), tuple(dict.fromkeys(reasons)), valid_baseline)


def derive_boundaries(tool: ToolDefinition) -> tuple[SchemaBoundary, ...]:
    """Derive proved-invalid argument objects from one tool's declared schema.

    Raises ``UnsafeSchemaReference`` when the schema reaches outside its own
    document, and ``SchemaError`` when the schema itself is invalid.
    """

    return _analyze(tool).boundaries


def unsupported_boundary_reasons(tool: ToolDefinition) -> tuple[str, ...]:
    """Name every schema feature that produced no case, so silence is never implicit."""

    return _analyze(tool).reasons


def _request_turn(boundary: SchemaBoundary) -> ConversationTurn:
    if boundary.kind is BoundaryKind.MISSING_REQUIRED_PROPERTY:
        content = (
            f"Use the {boundary.tool_name} tool for this request. The required "
            f"{boundary.parameter} value is not available in this conversation."
        )
    else:
        rendered = json.dumps(
            boundary.arguments, ensure_ascii=False, allow_nan=False, sort_keys=True
        )
        content = (
            f"Use the {boundary.tool_name} tool with exactly these arguments: {rendered}"
        )
    return ConversationTurn(turn_id="turn-1", role=ConversationRole.USER, content=content)


def _boundary_scenario(boundary: SchemaBoundary, *, seed: int) -> Scenario:
    scenario_id = _scenario_id(boundary.tool_name, boundary.parameter, boundary.kind)
    oracle_id = f"{scenario_id}:oracle"
    if boundary.kind is BoundaryKind.MISSING_REQUIRED_PROPERTY:
        # Subset matching cannot express an absent property, and the conversation
        # withholds the value, so any call would omit or invent it.
        arguments_match: JsonObject = {}
        title = f"{boundary.tool_name} must not be called without {boundary.parameter}"
    else:
        arguments_match = {boundary.parameter: boundary.invalid_value}
        title = (
            f"{boundary.tool_name} must not be called with an out-of-contract "
            f"{boundary.parameter}"
        )
    return Scenario(
        scenario_id=scenario_id,
        title=title[:500],
        description=boundary.rationale,
        conversation_turns=(_request_turn(boundary),),
        tool_fixtures=(
            # A permissive in-contract fixture keeps an invented but schema-valid
            # call from becoming a fixture_not_found INFRA_ERROR.
            ToolFixture(
                fixture_id=f"{scenario_id}:fixture",
                tool_name=boundary.tool_name,
                outcome=SimulatedToolOutcome(
                    status=SimulatedToolStatus.SUCCESS,
                    result={"acknowledged": True},
                ),
            ),
        ),
        forbidden_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id=f"{scenario_id}:forbidden",
                tool_name=boundary.tool_name,
                arguments_match=arguments_match,
                min_calls=0,
                max_calls=0,
                oracle_ids=(oracle_id,),
            ),
        ),
        # No allowed_tool_behavior: subset matching plus `_evaluate_schema_blocks`
        # already hard-fails the intended schema violation via the builtin
        # `tool_schema` oracle. An allowed contract with `arguments_match={}`
        # would make `_evaluate_uncontracted_tool_arguments` a no-op; one that
        # names the baseline would fail invented but schema-valid calls, which
        # is not the schema-boundary assertion. The permissive fixture exists
        # so those invented valid calls cannot become fixture_not_found.
        dimension_tags=(
            f"tool:{boundary.tool_name}",
            f"schema:{boundary.kind.value}",
            "source:schema_boundary",
        ),
        oracle_provenance=(
            OracleProvenance(
                oracle_id=oracle_id,
                strength=OracleStrength.TOOL_CONTRACT,
                source=(
                    f"declared input schema of {boundary.tool_name} at "
                    f"{boundary.pointer}"
                ),
                confidence=1.0,
                evidence_ids=boundary.evidence_ids,
                supports_hard_failure=True,
            ),
        ),
        resource_budgets=ResourceBudgets(max_model_turns=4, max_tool_calls=4),
        generation_seed=seed,
    )


def build_output_schema_cases(spec: AgentSpec, *, seed: int) -> tuple[Scenario, ...]:
    """Assert the declared structured output contract for a statically typed agent.

    An agent can declare an authoritative ``output_type`` and expose no
    ``FunctionTool`` at all -- the real ``email-agent-workflow`` triage agent is
    exactly that shape. Tool-boundary generation has nothing to derive from
    there, which previously left such a target inspectable but with no
    compatible suite.

    The declared output JSON Schema is itself an authoritative contract, so
    "the final output must validate against it" is a deterministic, offline
    assertion -- the existing ``json_schema`` output criterion, evaluated by the
    offline validator. No model writes the case, nothing is sampled, and the
    schema is the one PR #12 already proved statically recoverable. A schema
    that is absent, non-authoritative, empty, or unsafe to validate yields no
    case rather than a weaker one.
    """

    if seed < 0 or seed > 2**63 - 1:
        raise ValueError("seed must be between 0 and 2^63 - 1")
    declared = spec.interface.output_schema
    schema = declared.value
    if not declared.authoritative or not isinstance(schema, Mapping) or not schema:
        return ()
    try:
        # Refuse rather than emit a case whose oracle could not run offline.
        offline_validator(dict(schema))
    except Exception:
        return ()

    scenario_id = "output-schema-conformance"
    oracle_id = f"{scenario_id}:oracle"
    evidence_ids = tuple(item.evidence_id for item in declared.evidence) or (
        "declared-output-schema",
    )
    return (
        Scenario(
            scenario_id=scenario_id,
            title="final output must satisfy the declared output schema",
            description=(
                "The agent declares a structured output type, so any final output "
                "that does not validate against that schema breaks its own "
                "declared contract."
            ),
            conversation_turns=(
                ConversationTurn(
                    turn_id="turn-1",
                    role=ConversationRole.USER,
                    content=(
                        "Handle this request and return your response using your "
                        "declared structured output format."
                    ),
                ),
            ),
            output_criteria=(
                OutputCriterion(
                    criterion_id=f"{scenario_id}:schema",
                    kind=OutputCriterionKind.JSON_SCHEMA,
                    description=(
                        "The final output must validate against the declared "
                        "output schema."
                    ),
                    parameters={"schema": dict(schema)},
                    oracle_ids=(oracle_id,),
                ),
            ),
            dimension_tags=("source:output_schema", "schema:output_conformance"),
            oracle_provenance=(
                OracleProvenance(
                    oracle_id=oracle_id,
                    strength=OracleStrength.TOOL_CONTRACT,
                    source=f"declared output schema at {declared.source.locator}",
                    confidence=1.0,
                    evidence_ids=evidence_ids,
                    supports_hard_failure=True,
                ),
            ),
            resource_budgets=ResourceBudgets(max_model_turns=4, max_tool_calls=4),
            generation_seed=seed,
        ),
    )


_MAX_POSITIVE_SCENARIOS_PER_SPEC = 24


@dataclass(frozen=True, slots=True)
class PositiveCase:
    """One action-path scenario plus how representative its inputs are.

    ``shallow_parameters`` names the arguments that fell back to a generic
    synthetic value because neither a developer fixture nor the declared schema
    supplied anything domain-shaped. A model may reasonably decline to act on
    those, so a pass here is weaker evidence than one where the request carried
    representative values, and the suite reports that rather than hiding it.
    """

    scenario: Scenario
    tool_name: str
    shallow_parameters: tuple[str, ...]

    @property
    def representative(self) -> bool:
        return not self.shallow_parameters


def _humanise(tool_name: str) -> str:
    """Fallback phrasing when a tool declares no description."""

    return tool_name.replace("_", " ").replace("-", " ").strip() or tool_name


def _render_value(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _positive_request(tool: ToolDefinition, arguments: JsonObject) -> str:
    """Ask for the tool's declared action and supply every required value.

    Phrased from the target's own declared description, falling back to the
    tool name, and never naming the tool as an instruction to call it. The
    request states what the user wants and hands over the information needed;
    whether that warrants a tool call is the model's decision, which is the
    only way the resulting trajectory says anything about the agent.
    """

    action = (tool.description or "").strip() or f"Please {_humanise(tool.name)}."
    if not action.endswith((".", "!", "?")):
        action = f"{action}."
    if not arguments:
        return action
    supplied = ", ".join(
        f"{name} is {_render_value(arguments[name])}" for name in sorted(arguments)
    )
    return f"{action} Here is the information you have: {supplied}."


def _positive_scenario(tool: ToolDefinition, arguments: JsonObject, *, seed: int) -> Scenario:
    scenario_id = f"action-{_slug(tool.name)}"[:_MAX_SCENARIO_ID]
    oracle_id = f"{scenario_id}:oracle"
    return Scenario(
        scenario_id=scenario_id,
        title=f"{tool.name} may be called when every required value is available",
        description=(
            "Every required argument this tool declares is present and in "
            "contract, so calling it is a legitimate option. This case exists "
            "to exercise the action path, not to require the call: whether the "
            "tool is the right response is the agent's judgement."
        ),
        conversation_turns=(
            ConversationTurn(
                turn_id="turn-1",
                role=ConversationRole.USER,
                content=_positive_request(tool, arguments)[:8_000],
            ),
        ),
        tool_fixtures=(
            ToolFixture(
                fixture_id=f"{scenario_id}:fixture",
                tool_name=tool.name,
                outcome=SimulatedToolOutcome(
                    status=SimulatedToolStatus.SUCCESS,
                    result={"acknowledged": True},
                ),
            ),
        ),
        # Permitted, never required: min_calls stays 0 because AgentCheck
        # cannot prove from a schema alone that this request obliges the agent
        # to act, so declining is not a defect. Naming the in-contract arguments
        # still makes a same-tool call carrying different arguments an
        # observable out-of-contract deviation.
        allowed_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id=f"{scenario_id}:allowed",
                tool_name=tool.name,
                arguments_match=dict(arguments),
                min_calls=0,
                oracle_ids=(oracle_id,),
            ),
        ),
        # The evaluable assertion. It holds vacuously when the agent declines,
        # which is why declining is not a defect, and becomes meaningful the
        # moment a call happens: one request carrying one complete argument set
        # justifies that action once, so repeating it is a duplicate side
        # effect. This is the property the boundary-only suite could never
        # observe, because it never produced a call to observe.
        trajectory_constraints=(
            TrajectoryConstraint(
                criterion_id=f"{scenario_id}:no_duplicate",
                kind=TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT,
                description=(
                    f"{tool.name} must not repeat an identical call within this "
                    "single request."
                ),
                parameters={"tool_name": tool.name},
                oracle_ids=(oracle_id,),
            ),
        ),
        dimension_tags=(f"tool:{tool.name}", "source:positive_path", "path:action"),
        oracle_provenance=(
            OracleProvenance(
                oracle_id=oracle_id,
                strength=OracleStrength.TOOL_CONTRACT,
                source=f"declared input schema of {tool.name}",
                confidence=1.0,
                evidence_ids=(f"positive-path:{tool.name}",),
                supports_hard_failure=True,
            ),
        ),
        resource_budgets=ResourceBudgets(max_model_turns=4, max_tool_calls=4),
        generation_seed=seed,
    )


def _apply_representative_inputs(
    baseline: JsonObject,
    supplied: Mapping[str, Any],
    parameters: Mapping[str, CapabilityParameter],
) -> tuple[JsonObject, tuple[str, ...]]:
    """Overlay developer values on the schema baseline and report what stayed generic.

    Precedence is developer fixture, then a value the declared schema actually
    pins down (an enum member, a bounded number), then the generic synthetic
    string. Only the last case is shallow: the first two carry meaning the
    contract or the developer put there.
    """

    merged: dict[str, Any] = dict(baseline)
    shallow: list[str] = []
    for name in sorted(baseline):
        if name in supplied:
            merged[name] = supplied[name]
            continue
        parameter = parameters.get(name)
        constrained = bool(
            parameter is not None
            and (
                parameter.constraints.enum_values
                or parameter.constraints.pattern
                or parameter.constraints.string_format
                or parameter.constraints.minimum is not None
                or parameter.constraints.maximum is not None
            )
        )
        if not constrained and baseline[name] == _SYNTHETIC_STRING:
            shallow.append(name)
    return merged, tuple(shallow)


def build_positive_path_cases(
    spec: AgentSpec,
    *,
    seed: int,
    representative_inputs: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[PositiveCase, ...]:
    """One action-path case per tool whose declared schema yields valid arguments.

    Boundary cases only ever ask for an invalid call, so a well-behaved agent
    passes the whole suite by never calling a tool -- which leaves every
    trajectory policy vacuous, because duplicate side effects and fabricated
    success can only be observed once a call actually happens.

    Conservative by construction: the arguments are the same validated baseline
    the boundary cases were derived from, so a tool whose contract yields no
    valid argument object contributes no case rather than an invented one.
    """

    if seed < 0 or seed > 2**63 - 1:
        raise ValueError("seed must be between 0 and 2^63 - 1")
    supplied_by_tool = representative_inputs or {}
    cases: list[PositiveCase] = []
    seen: set[str] = set()
    for definition in sorted(
        (item.value for item in spec.tools.items), key=lambda tool: tool.name
    ):
        if len(cases) >= _MAX_POSITIVE_SCENARIOS_PER_SPEC:
            break
        analysis = _analyze(definition)
        if analysis.baseline is None:
            continue
        (extracted,) = extract_capabilities([definition])
        parameters = {p.name: p for p in extracted.arguments.parameters}
        arguments, shallow = _apply_representative_inputs(
            analysis.baseline, supplied_by_tool.get(definition.name, {}), parameters
        )
        scenario = _positive_scenario(definition, arguments, seed=seed)
        if scenario.fingerprint in seen:
            continue
        seen.add(scenario.fingerprint)
        cases.append(PositiveCase(scenario, definition.name, shallow))
    return tuple(cases)


def build_boundary_cases(
    spec: AgentSpec, *, seed: int
) -> tuple[tuple[SchemaBoundary, Scenario], ...]:
    """Build boundary scenarios paired with the constraint each one came from.

    The seed is recorded on every scenario rather than used to sample: generation
    is deterministic by construction, so the same seed reproduces identical
    fingerprints and a different seed yields a distinct suite.
    """

    if seed < 0 or seed > 2**63 - 1:
        raise ValueError("seed must be between 0 and 2^63 - 1")
    definitions = tuple(item.value for item in spec.tools.items)
    cases: list[tuple[SchemaBoundary, Scenario]] = []
    seen: set[str] = set()
    for definition in definitions:
        for boundary in derive_boundaries(definition):
            if len(cases) >= MAX_BOUNDARY_SCENARIOS_PER_SPEC:
                return tuple(cases)
            scenario = _boundary_scenario(boundary, seed=seed)
            if scenario.fingerprint in seen:
                continue
            seen.add(scenario.fingerprint)
            cases.append((boundary, scenario))
    return tuple(cases)


def build_boundary_scenarios(spec: AgentSpec, *, seed: int) -> tuple[Scenario, ...]:
    """Build deterministic ``agentcheck.scenario.v1`` boundary cases for a target."""

    return tuple(scenario for _, scenario in build_boundary_cases(spec, seed=seed))


def _zero_input_extracted(tool: ToolDefinition) -> ExtractedCapability | None:
    """Return the capability when the schema is a legitimate empty-object call.

    Schema-boundary generation stays invalid-argument-only. A tool that accepts
    ``{}`` and declares no parameters is still a callable surface; it is not an
    argument-boundary inversion and must not be given invented properties.
    """

    validator = offline_validator(tool.input_schema)
    (extracted,) = extract_capabilities([tool])
    surface = extracted.arguments
    if not surface.schema_known or surface.parameters:
        return None
    if not validator.is_valid({}):
        return None
    return extracted


def _zero_input_scenario_id(tool_name: str) -> str:
    candidate = f"zero-input-{_slug(tool_name)}"
    if len(candidate) <= _MAX_SCENARIO_ID:
        return candidate
    digest = canonical_hash(candidate).split(":", 1)[1][:16]
    return f"{candidate[: _MAX_SCENARIO_ID - 17]}-{digest}"


def _zero_input_scenario(
    tool: ToolDefinition, extracted: ExtractedCapability, *, seed: int
) -> Scenario:
    scenario_id = _zero_input_scenario_id(tool.name)
    oracle_id = f"{scenario_id}:oracle"
    empty_arguments: JsonObject = {}
    rendered = json.dumps(
        empty_arguments, ensure_ascii=False, allow_nan=False, sort_keys=True
    )
    return Scenario(
        scenario_id=scenario_id,
        title=f"{tool.name} may be invoked with no arguments"[:500],
        description=(
            f"The declared input schema of {tool.name} accepts an empty object "
            "and exposes no parameters. This case exercises that in-contract "
            "zero-input invocation; it is not an invalid-argument boundary."
        ),
        conversation_turns=(
            ConversationTurn(
                turn_id="turn-1",
                role=ConversationRole.USER,
                content=(
                    f"Use the {tool.name} tool with no arguments: {rendered}"
                ),
            ),
        ),
        tool_fixtures=(
            ToolFixture(
                fixture_id=f"{scenario_id}:fixture",
                tool_name=tool.name,
                arguments_match=empty_arguments,
                outcome=SimulatedToolOutcome(
                    status=SimulatedToolStatus.SUCCESS,
                    result={"acknowledged": True},
                ),
            ),
        ),
        required_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id=f"{scenario_id}:required",
                tool_name=tool.name,
                arguments_match=empty_arguments,
                min_calls=1,
                oracle_ids=(oracle_id,),
            ),
        ),
        dimension_tags=(
            f"tool:{tool.name}",
            "source:zero_input_invocation",
        ),
        oracle_provenance=(
            OracleProvenance(
                oracle_id=oracle_id,
                strength=OracleStrength.TOOL_CONTRACT,
                source=(
                    f"declared input schema of {tool.name} accepts no parameters"
                ),
                confidence=1.0,
                evidence_ids=_evidence_ids(
                    _evidence_lookup(extracted), tool.name, None
                ),
                supports_hard_failure=True,
            ),
        ),
        resource_budgets=ResourceBudgets(max_model_turns=4, max_tool_calls=4),
        generation_seed=seed,
    )


def build_zero_input_cases(
    spec: AgentSpec, *, seed: int
) -> tuple[tuple[str, Scenario], ...]:
    """Build one in-contract empty-object invocation per qualifying tool.

    The seed is recorded on every scenario rather than used to sample.
    """

    if seed < 0 or seed > 2**63 - 1:
        raise ValueError("seed must be between 0 and 2^63 - 1")
    cases: list[tuple[str, Scenario]] = []
    seen: set[str] = set()
    for item in spec.tools.items:
        tool = item.value
        extracted = _zero_input_extracted(tool)
        if extracted is None:
            continue
        scenario = _zero_input_scenario(tool, extracted, seed=seed)
        if scenario.fingerprint in seen:
            continue
        seen.add(scenario.fingerprint)
        cases.append((tool.name, scenario))
    return tuple(cases)


__all__ = [
    "BOUNDARY_CONTRACT_VERSION",
    "MAX_BOUNDARIES_PER_PARAMETER",
    "MAX_BOUNDARIES_PER_TOOL",
    "MAX_BOUNDARY_SCENARIOS_PER_SPEC",
    "BoundaryKind",
    "SchemaBoundary",
    "build_boundary_cases",
    "build_boundary_scenarios",
    "build_zero_input_cases",
    "derive_boundaries",
    "unsupported_boundary_reasons",
]
