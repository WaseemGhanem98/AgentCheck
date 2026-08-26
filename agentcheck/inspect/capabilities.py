"""Systematic, provider-neutral capability extraction from declared tool contracts.

Extraction is static analysis over data that inspection already collected: it
imports nothing, executes nothing, and reaches no network or filesystem
resource.  The argument surface is read directly from each tool's JSON Schema
and is therefore authoritative.  Action kind and side-effect risk have no
authoritative source in any framework AgentCheck currently supports, so they
stay inferred, non-authoritative, and below the confidence threshold that would
let them authorize a hard failure.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Mapping, NamedTuple, Protocol, Sequence

from pydantic import Field, model_validator

from agentcheck.domain import (
    ActionKind,
    Capability,
    ContractModel,
    JsonValue,
    SourceKind,
    SourceReference,
    SpecEvidence,
    ToolDefinition,
    UnknownProperty,
    canonical_hash,
)
from agentcheck.schema_safety import UnsafeSchemaReference, ensure_local_schema_references


CAPABILITY_EXTRACTION_CONTRACT_VERSION: Literal["agentcheck.extracted_capability.v1"] = (
    "agentcheck.extracted_capability.v1"
)

# Mirrors AgentProperty.protect_hard_rules_from_weak_inference: below this an
# extracted value can never be promoted to an authoritative one.
AUTHORITY_CONFIDENCE_THRESHOLD = 0.8

MAX_PARAMETERS_PER_TOOL = 50
MAX_ENUM_VALUES = 20
MAX_NESTED_PROPERTY_NAMES = 20
MAX_EVIDENCE_PER_CAPABILITY = 24

_COMPOSITION_KEYWORDS = (
    "$dynamicRef",
    "$ref",
    "allOf",
    "anyOf",
    "dependentSchemas",
    "if",
    "not",
    "oneOf",
    "patternProperties",
    "propertyNames",
    "unevaluatedProperties",
)

# Reproduces the Phase 1 name-token vocabulary exactly.  Order is significant:
# the first matching rule wins, so a tool named ``delete_and_update`` stays a
# deletion.  These are weak lexical hints, never authoritative classifications.
_NAME_RULES: tuple[tuple[frozenset[str], ActionKind, bool, bool], ...] = (
    (frozenset({"delete", "remove", "destroy", "erase", "purge"}), ActionKind.DELETE, True, True),
    (frozenset({"cancel", "terminate", "close"}), ActionKind.MODIFY, True, True),
    (frozenset({"update", "modify", "change", "set", "edit"}), ActionKind.MODIFY, True, False),
    (frozenset({"create", "add", "open", "register"}), ActionKind.CREATE, True, False),
    (frozenset({"send", "email", "notify", "publish"}), ActionKind.SEND, True, False),
    (frozenset({"schedule", "book", "reserve"}), ActionKind.SCHEDULE, True, False),
    (frozenset({"lookup", "find", "search", "get", "read"}), ActionKind.LOOKUP, False, False),
    (frozenset({"retrieve", "fetch"}), ActionKind.RETRIEVE, False, False),
    (frozenset({"summarize", "summary"}), ActionKind.SUMMARIZE, False, False),
)

_NAME_TOKEN_CONFIDENCE = 0.6
_CORROBORATED_CONFIDENCE = 0.7
_NO_EVIDENCE_CONFIDENCE = 0.3


class JsonSchemaType(str, Enum):
    ARRAY = "array"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NULL = "null"
    NUMBER = "number"
    OBJECT = "object"
    STRING = "string"


class CapabilitySignalKind(str, Enum):
    """Why an extracted value holds the value it does."""

    NAME_TOKEN = "name_token"
    DESCRIPTION_TOKEN = "description_token"
    NO_CLASSIFICATION_EVIDENCE = "no_classification_evidence"
    SCHEMA_PARAMETER = "schema_parameter"
    SCHEMA_CONSTRAINT = "schema_constraint"
    SCHEMA_UNREADABLE = "schema_unreadable"


_CLASSIFICATION_SIGNAL_KINDS = frozenset(
    {
        CapabilitySignalKind.NAME_TOKEN,
        CapabilitySignalKind.DESCRIPTION_TOKEN,
        CapabilitySignalKind.NO_CLASSIFICATION_EVIDENCE,
    }
)


class CapabilitySignal(ContractModel):
    """One recorded observation, with the authority it does or does not carry."""

    kind: CapabilitySignalKind
    detail: str = Field(min_length=1, max_length=2_000)
    locator: str = Field(min_length=1, max_length=1_000)
    source: SourceReference
    confidence: float = Field(ge=0.0, le=1.0)
    authoritative: bool = False

    @model_validator(mode="after")
    def protect_authority(self) -> "CapabilitySignal":
        if self.authoritative and self.source.kind in {
            SourceKind.LLM_INFERENCE,
            SourceKind.UNKNOWN,
        }:
            raise ValueError("LLM-inferred or unknown signals cannot be authoritative")
        if self.authoritative and self.confidence < AUTHORITY_CONFIDENCE_THRESHOLD:
            raise ValueError("low-confidence signals cannot be authoritative")
        if self.authoritative and self.kind in _CLASSIFICATION_SIGNAL_KINDS:
            raise ValueError("lexical classification signals are never authoritative")
        return self


class ValueConstraints(ContractModel):
    """Schema-declared bounds on one parameter; every field is unknown when null."""

    enum_values: tuple[JsonValue, ...] | None = None
    enum_truncated: bool = False
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: float | None = None
    exclusive_maximum: float | None = None
    multiple_of: float | None = None
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    pattern: str | None = Field(default=None, max_length=1_000)
    string_format: str | None = Field(default=None, max_length=200)
    min_items: int | None = Field(default=None, ge=0)
    max_items: int | None = Field(default=None, ge=0)
    unique_items: bool | None = None
    item_types: tuple[JsonSchemaType, ...] = ()
    nested_property_names: tuple[str, ...] = ()

    @property
    def declared(self) -> bool:
        return self.model_dump(exclude_defaults=True) != {}


class CapabilityParameter(ContractModel):
    """One argument a capability accepts, as declared by the tool schema."""

    name: str = Field(min_length=1, max_length=200)
    pointer: str = Field(min_length=1, max_length=1_000)
    required: bool
    types: tuple[JsonSchemaType, ...] = ()
    types_known: bool = False
    description: str | None = Field(default=None, max_length=2_000)
    constraints: ValueConstraints = Field(default_factory=ValueConstraints)
    unsupported_constructs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def keep_unknown_types_empty(self) -> "CapabilityParameter":
        if not self.types_known and self.types:
            raise ValueError("unknown parameter types must not carry a type list")
        if self.types_known and not self.types:
            raise ValueError("known parameter types require at least one type")
        return self


class CapabilityArgumentSurface(ContractModel):
    """The complete argument contract, including what could not be established."""

    schema_known: bool
    parameters: tuple[CapabilityParameter, ...] = ()
    additional_properties_allowed: bool | None = None
    unsupported_constructs: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def required_parameters(self) -> tuple[CapabilityParameter, ...]:
        return tuple(item for item in self.parameters if item.required)

    @property
    def optional_parameters(self) -> tuple[CapabilityParameter, ...]:
        return tuple(item for item in self.parameters if not item.required)


class ExtractedCapability(ContractModel):
    """A capability plus the evidence, provenance, and confidence behind it."""

    schema_version: Literal["agentcheck.extracted_capability.v1"] = (
        CAPABILITY_EXTRACTION_CONTRACT_VERSION
    )
    tool_name: str = Field(min_length=1, max_length=200)
    capability: Capability
    confidence: float = Field(ge=0.0, le=1.0)
    classification_inferred: bool = True
    evidence: tuple[SpecEvidence, ...] = Field(min_length=1)
    signals: tuple[CapabilitySignal, ...] = Field(min_length=1)
    arguments: CapabilityArgumentSurface
    unknowns: tuple[UnknownProperty, ...] = ()

    @model_validator(mode="after")
    def keep_classification_below_authority(self) -> "ExtractedCapability":
        authoritative_classification = any(
            signal.authoritative and signal.kind in _CLASSIFICATION_SIGNAL_KINDS
            for signal in self.signals
        )
        if (
            self.confidence >= AUTHORITY_CONFIDENCE_THRESHOLD
            and not authoritative_classification
        ):
            raise ValueError(
                "an inferred classification cannot reach the authoritative "
                "confidence threshold without an authoritative signal"
            )
        if not self.classification_inferred and not authoritative_classification:
            raise ValueError(
                "a non-inferred classification requires an authoritative signal"
            )
        return self


class CapabilityExtractor(Protocol):
    """Derives capabilities from declared tool contracts without executing them."""

    def extract(
        self, tools: Sequence[ToolDefinition]
    ) -> tuple[ExtractedCapability, ...]: ...


def _tokens(value: str) -> set[str]:
    normalized = "".join(
        character if character.isalnum() else " " for character in value.casefold()
    )
    return set(normalized.split())


def _matched_rule(
    tokens: set[str],
) -> tuple[frozenset[str], ActionKind, bool, bool] | None:
    for rule in _NAME_RULES:
        if tokens & rule[0]:
            return rule
    return None


def _evidence(locator: str, summary: str, *, discriminator: str) -> SpecEvidence:
    digest = canonical_hash([locator, discriminator]).split(":", 1)[1][:20]
    return SpecEvidence(
        evidence_id=f"evidence-{digest}",
        summary=summary,
        locator=locator,
    )


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _bounded_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    # Contract models reject non-finite numbers outright.
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _bounded_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _bounded_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:limit]


def _declared_types(schema: Mapping[str, Any]) -> tuple[JsonSchemaType, ...] | None:
    """Read an explicit ``type`` keyword, returning None when it is unusable."""

    raw = schema.get("type")
    if isinstance(raw, str):
        names: list[str] = [raw]
    elif isinstance(raw, list) and raw and all(isinstance(item, str) for item in raw):
        names = list(raw)
    else:
        return None
    resolved: set[JsonSchemaType] = set()
    for name in names:
        try:
            resolved.add(JsonSchemaType(name))
        except ValueError:
            return None
    return tuple(sorted(resolved, key=lambda item: item.value))


def _union_branch_types(schema: Mapping[str, Any]) -> tuple[JsonSchemaType, ...] | None:
    """Resolve the common ``anyOf``/``oneOf`` union of plain typed branches.

    Optional arguments are usually declared this way.  Only branches that carry
    nothing but a ``type`` are resolved; anything richer stays unknown.
    """

    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if not isinstance(branches, list) or not branches:
            continue
        resolved: set[JsonSchemaType] = set()
        for branch in branches:
            mapping = _mapping(branch)
            if mapping is None or set(mapping) != {"type"}:
                return None
            branch_types = _declared_types(mapping)
            if branch_types is None:
                return None
            resolved.update(branch_types)
        if resolved:
            return tuple(sorted(resolved, key=lambda item: item.value))
    return None


def _enum_types(values: Sequence[Any]) -> tuple[JsonSchemaType, ...] | None:
    resolved: set[JsonSchemaType] = set()
    for value in values:
        if value is None:
            resolved.add(JsonSchemaType.NULL)
        elif isinstance(value, bool):
            resolved.add(JsonSchemaType.BOOLEAN)
        elif isinstance(value, int):
            resolved.add(JsonSchemaType.INTEGER)
        elif isinstance(value, float):
            resolved.add(JsonSchemaType.NUMBER)
        elif isinstance(value, str):
            resolved.add(JsonSchemaType.STRING)
        elif isinstance(value, list):
            resolved.add(JsonSchemaType.ARRAY)
        elif isinstance(value, Mapping):
            resolved.add(JsonSchemaType.OBJECT)
        else:
            return None
    return tuple(sorted(resolved, key=lambda item: item.value)) or None


def _enum_values(schema: Mapping[str, Any]) -> tuple[tuple[Any, ...] | None, bool]:
    if "const" in schema:
        return (schema["const"],), False
    raw = schema.get("enum")
    if not isinstance(raw, list) or not raw:
        return None, False
    truncated = len(raw) > MAX_ENUM_VALUES
    return tuple(raw[:MAX_ENUM_VALUES]), truncated


def _constraints(schema: Mapping[str, Any]) -> ValueConstraints:
    enum_values, enum_truncated = _enum_values(schema)
    items = _mapping(schema.get("items"))
    item_types = _declared_types(items) if items is not None else None
    nested = _mapping(schema.get("properties"))
    nested_names = (
        tuple(sorted(str(key) for key in nested)[:MAX_NESTED_PROPERTY_NAMES])
        if nested is not None
        else ()
    )
    unique_items = schema.get("uniqueItems")
    return ValueConstraints(
        enum_values=enum_values,
        enum_truncated=enum_truncated,
        minimum=_bounded_number(schema.get("minimum")),
        maximum=_bounded_number(schema.get("maximum")),
        exclusive_minimum=_bounded_number(schema.get("exclusiveMinimum")),
        exclusive_maximum=_bounded_number(schema.get("exclusiveMaximum")),
        multiple_of=_bounded_number(schema.get("multipleOf")),
        min_length=_bounded_integer(schema.get("minLength")),
        max_length=_bounded_integer(schema.get("maxLength")),
        pattern=_bounded_text(schema.get("pattern"), 1_000),
        string_format=_bounded_text(schema.get("format"), 200),
        min_items=_bounded_integer(schema.get("minItems")),
        max_items=_bounded_integer(schema.get("maxItems")),
        unique_items=unique_items if isinstance(unique_items, bool) else None,
        item_types=item_types or (),
        nested_property_names=nested_names,
    )


def _unsupported_constructs(schema: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(keyword for keyword in _COMPOSITION_KEYWORDS if keyword in schema)
    )


def _parameter(
    name: str,
    schema: Mapping[str, Any] | None,
    *,
    required: bool,
) -> CapabilityParameter:
    pointer = f"/properties/{name.replace('~', '~0').replace('/', '~1')}"
    if schema is None:
        return CapabilityParameter(
            name=name,
            pointer=pointer,
            required=required,
            types_known=False,
            unsupported_constructs=("unreadable_property_schema",),
        )
    constructs = _unsupported_constructs(schema)
    types = _declared_types(schema)
    if types is None:
        types = _union_branch_types(schema)
    if types is None:
        enum_values, _ = _enum_values(schema)
        types = _enum_types(enum_values) if enum_values is not None else None
    # A composition keyword can widen a declared type, so a schema that uses one
    # is only reported as typed when nothing else is ambiguous.
    if constructs and set(constructs) - {"anyOf", "oneOf"}:
        types = None
    return CapabilityParameter(
        name=name,
        pointer=pointer,
        required=required,
        types=types or (),
        types_known=types is not None,
        description=_bounded_text(schema.get("description"), 2_000),
        constraints=_constraints(schema),
        unsupported_constructs=constructs,
    )


def _argument_surface(
    definition: ToolDefinition,
) -> tuple[CapabilityArgumentSurface, tuple[str, ...]]:
    """Read the declared argument contract, returning it with any blocking reason."""

    schema = definition.input_schema
    if not isinstance(schema, Mapping):
        return (
            CapabilityArgumentSurface(schema_known=False),
            ("the tool input schema is not a JSON object",),
        )
    try:
        ensure_local_schema_references(schema)
    except UnsafeSchemaReference as exc:
        return (
            CapabilityArgumentSurface(
                schema_known=False,
                unsupported_constructs=("non_local_schema_reference",),
            ),
            (f"the tool input schema retrieves a non-local reference: {exc}",),
        )

    properties = _mapping(schema.get("properties"))
    if properties is None and "properties" in schema:
        return (
            CapabilityArgumentSurface(schema_known=False),
            ("the tool input schema declares a malformed 'properties' value",),
        )

    raw_required = schema.get("required", [])
    if isinstance(raw_required, list) and all(
        isinstance(item, str) for item in raw_required
    ):
        required_names = set(raw_required)
        required_known = True
    else:
        required_names = set()
        required_known = "required" not in schema

    names = sorted(str(key) for key in (properties or {}))
    truncated = len(names) > MAX_PARAMETERS_PER_TOOL
    parameters = tuple(
        _parameter(
            name,
            _mapping((properties or {}).get(name)),
            required=name in required_names,
        )
        for name in names[:MAX_PARAMETERS_PER_TOOL]
    )
    additional = schema.get("additionalProperties")
    constructs = list(_unsupported_constructs(schema))
    if not required_known:
        constructs.append("malformed_required")
    surface = CapabilityArgumentSurface(
        schema_known=True,
        parameters=parameters,
        additional_properties_allowed=(
            additional if isinstance(additional, bool) else None
        ),
        unsupported_constructs=tuple(sorted(set(constructs))),
        truncated=truncated,
    )
    reasons = (
        ("the tool input schema declares a malformed 'required' value",)
        if not required_known
        else ()
    )
    return surface, reasons


def _classification_signals(
    definition: ToolDefinition,
) -> tuple[ActionKind, bool, bool, float, tuple[CapabilitySignal, ...], bool]:
    name_tokens = _tokens(definition.name)
    rule = _matched_rule(name_tokens)
    locator = f"tool:{definition.name}.name"
    signals: list[CapabilitySignal] = []
    if rule is None:
        signals.append(
            CapabilitySignal(
                kind=CapabilitySignalKind.NO_CLASSIFICATION_EVIDENCE,
                detail=(
                    "No declared metadata or name token establishes an action kind; "
                    "the capability stays unclassified and non-state-changing."
                ),
                locator=locator,
                source=SourceReference(kind=SourceKind.UNKNOWN, locator=locator),
                confidence=_NO_EVIDENCE_CONFIDENCE,
            )
        )
        return (
            ActionKind.OTHER,
            False,
            False,
            _NO_EVIDENCE_CONFIDENCE,
            tuple(signals),
            False,
        )

    vocabulary, action_kind, state_changing, destructive = rule
    matched = sorted(name_tokens & vocabulary)
    signals.append(
        CapabilitySignal(
            kind=CapabilitySignalKind.NAME_TOKEN,
            detail=(
                f"Tool name token(s) {', '.join(repr(token) for token in matched)} "
                f"suggest action kind {action_kind.value!r}. Lexical evidence is "
                "never authoritative."
            ),
            locator=locator,
            source=SourceReference(kind=SourceKind.TOOL_SCHEMA, locator=locator),
            confidence=_NAME_TOKEN_CONFIDENCE,
        )
    )
    confidence = _NAME_TOKEN_CONFIDENCE
    corroborated = False
    if definition.description:
        description_locator = f"tool:{definition.name}.description"
        shared = sorted(_tokens(definition.description) & vocabulary)
        if shared:
            corroborated = True
            confidence = _CORROBORATED_CONFIDENCE
            signals.append(
                CapabilitySignal(
                    kind=CapabilitySignalKind.DESCRIPTION_TOKEN,
                    detail=(
                        f"The declared description repeats token(s) "
                        f"{', '.join(repr(token) for token in shared)}, corroborating "
                        f"action kind {action_kind.value!r}."
                    ),
                    locator=description_locator,
                    source=SourceReference(
                        kind=SourceKind.TOOL_SCHEMA, locator=description_locator
                    ),
                    confidence=_CORROBORATED_CONFIDENCE,
                )
            )
    return (
        action_kind,
        state_changing,
        destructive,
        confidence,
        tuple(signals),
        corroborated,
    )


def _parameter_summary(parameter: CapabilityParameter) -> str:
    requirement = "required" if parameter.required else "optional"
    types = (
        ", ".join(item.value for item in parameter.types)
        if parameter.types_known
        else "unknown type"
    )
    details: list[str] = []
    constraints = parameter.constraints
    if constraints.enum_values is not None:
        suffix = ", truncated" if constraints.enum_truncated else ""
        details.append(f"{len(constraints.enum_values)} allowed value(s){suffix}")
    for label, value in (
        ("minimum", constraints.minimum),
        ("maximum", constraints.maximum),
        ("exclusiveMinimum", constraints.exclusive_minimum),
        ("exclusiveMaximum", constraints.exclusive_maximum),
        ("multipleOf", constraints.multiple_of),
        ("minLength", constraints.min_length),
        ("maxLength", constraints.max_length),
        ("minItems", constraints.min_items),
        ("maxItems", constraints.max_items),
    ):
        if value is not None:
            details.append(f"{label} {value:g}" if isinstance(value, float) else f"{label} {value}")
    if constraints.pattern is not None:
        details.append("a declared pattern")
    if constraints.string_format is not None:
        details.append(f"format {constraints.string_format}")
    if constraints.item_types:
        details.append(
            "items of type " + ", ".join(item.value for item in constraints.item_types)
        )
    if constraints.nested_property_names:
        details.append(f"{len(constraints.nested_property_names)} nested propertie(s)")
    if parameter.unsupported_constructs:
        details.append(
            "unresolved construct(s) " + ", ".join(parameter.unsupported_constructs)
        )
    suffix = f"; {'; '.join(details)}" if details else ""
    return f"Parameter {parameter.name!r} is {requirement}; {types}{suffix}."


def _schema_signals(
    definition: ToolDefinition, surface: CapabilityArgumentSurface
) -> tuple[CapabilitySignal, ...]:
    locator = f"tool:{definition.name}.input_schema"
    if not surface.schema_known:
        return (
            CapabilitySignal(
                kind=CapabilitySignalKind.SCHEMA_UNREADABLE,
                detail=(
                    "The declared input schema could not be read, so the argument "
                    "surface is unknown."
                ),
                locator=locator,
                source=SourceReference(kind=SourceKind.UNKNOWN, locator=locator),
                confidence=0.0,
            ),
        )
    signals: list[CapabilitySignal] = [
        CapabilitySignal(
            kind=CapabilitySignalKind.SCHEMA_PARAMETER,
            detail=(
                f"The declared input schema exposes "
                f"{len(surface.required_parameters)} required and "
                f"{len(surface.optional_parameters)} optional parameter(s)."
            ),
            locator=locator,
            source=SourceReference(kind=SourceKind.TOOL_SCHEMA, locator=locator),
            confidence=1.0,
            authoritative=True,
        )
    ]
    for parameter in surface.parameters:
        if not parameter.constraints.declared:
            continue
        parameter_locator = f"{locator}#{parameter.pointer}"
        signals.append(
            CapabilitySignal(
                kind=CapabilitySignalKind.SCHEMA_CONSTRAINT,
                detail=_parameter_summary(parameter),
                locator=parameter_locator,
                source=SourceReference(
                    kind=SourceKind.TOOL_SCHEMA, locator=parameter_locator
                ),
                confidence=1.0,
                authoritative=True,
            )
        )
    return tuple(signals)


def _capability_evidence(
    definition: ToolDefinition,
    surface: CapabilityArgumentSurface,
    classification_signals: Sequence[CapabilitySignal],
) -> tuple[SpecEvidence, ...]:
    tool_locator = f"tool:{definition.name}"
    schema_locator = f"{tool_locator}.input_schema"
    evidence: list[SpecEvidence] = [
        _evidence(
            tool_locator,
            f"Capability derived from the declared tool {definition.name!r}.",
            discriminator="capability",
        )
    ]
    for index, signal in enumerate(classification_signals):
        evidence.append(
            _evidence(signal.locator, signal.detail, discriminator=f"signal-{index}")
        )
    if not surface.schema_known:
        evidence.append(
            _evidence(
                schema_locator,
                "The declared input schema could not be read; the argument surface "
                "is unknown.",
                discriminator="schema-unknown",
            )
        )
        return tuple(evidence)

    additional = {
        True: "additional properties are accepted",
        False: "additional properties are rejected",
        None: "additional-property handling is unknown",
    }[surface.additional_properties_allowed]
    evidence.append(
        _evidence(
            schema_locator,
            (
                f"The declared input schema exposes "
                f"{len(surface.required_parameters)} required and "
                f"{len(surface.optional_parameters)} optional parameter(s); "
                f"{additional}."
            ),
            discriminator="schema",
        )
    )
    remaining = MAX_EVIDENCE_PER_CAPABILITY - len(evidence) - 1
    for parameter in surface.parameters[: max(remaining, 0)]:
        evidence.append(
            _evidence(
                f"{schema_locator}#{parameter.pointer}",
                _parameter_summary(parameter),
                discriminator="parameter",
            )
        )
    omitted = len(surface.parameters) - max(remaining, 0)
    if omitted > 0 or surface.truncated:
        evidence.append(
            _evidence(
                schema_locator,
                (
                    f"{max(omitted, 0)} further declared parameter(s) are omitted "
                    "from this evidence list."
                ),
                discriminator="parameter-truncation",
            )
        )
    return tuple(evidence)


def _unknown(
    path: str, reason: str, locator: str, summary: str, *, discriminator: str
) -> UnknownProperty:
    return UnknownProperty(
        path=path,
        reason=reason,
        source=SourceReference(kind=SourceKind.UNKNOWN, locator=locator),
        confidence=0.0,
        evidence=(_evidence(locator, summary, discriminator=discriminator),),
    )


class SchemaCapabilityExtractor:
    """Derives capabilities from tool schemas, descriptions, and declared names."""

    def extract(
        self, tools: Sequence[ToolDefinition]
    ) -> tuple[ExtractedCapability, ...]:
        return tuple(
            self._extract_one(index, definition)
            for index, definition in enumerate(tools)
        )

    def _extract_one(self, index: int, definition: ToolDefinition) -> ExtractedCapability:
        (
            action_kind,
            state_changing,
            destructive,
            confidence,
            classification_signals,
            _,
        ) = _classification_signals(definition)
        surface, blocking_reasons = _argument_surface(definition)
        signals = (*classification_signals, *_schema_signals(definition, surface))
        evidence = _capability_evidence(definition, surface, classification_signals)

        unknowns: list[UnknownProperty] = []
        if action_kind is ActionKind.OTHER and not any(
            signal.kind is CapabilitySignalKind.NAME_TOKEN
            for signal in classification_signals
        ):
            locator = f"tool:{definition.name}.name"
            unknowns.append(
                _unknown(
                    f"capabilities.items[{index}].action_kind",
                    "No authoritative source declares this tool's action kind or "
                    "side effects, so it stays unclassified rather than guessed.",
                    locator,
                    f"No name token of {definition.name!r} matches a known action kind.",
                    discriminator="unknown-action-kind",
                )
            )
        for reason in blocking_reasons:
            locator = f"tool:{definition.name}.input_schema"
            unknowns.append(
                _unknown(
                    f"capabilities.items[{index}].arguments",
                    f"The argument surface could not be established: {reason}.",
                    locator,
                    f"The input schema of {definition.name!r} could not be read.",
                    discriminator="unknown-arguments",
                )
            )

        return ExtractedCapability(
            tool_name=definition.name,
            capability=Capability(
                capability_id=f"tool:{definition.name}",
                name=definition.name.replace("_", " ").strip().title(),
                description=definition.description
                or f"Capability exposed by {definition.name}.",
                action_kind=action_kind,
                state_changing=state_changing,
                destructive=destructive,
            ),
            confidence=confidence,
            evidence=evidence,
            signals=signals,
            arguments=surface,
            unknowns=tuple(unknowns),
        )


DEFAULT_CAPABILITY_EXTRACTOR: CapabilityExtractor = SchemaCapabilityExtractor()


def extract_capabilities(
    tools: Sequence[ToolDefinition],
    *,
    extractor: CapabilityExtractor | None = None,
) -> tuple[ExtractedCapability, ...]:
    """Extract one capability per declared tool, preserving input order."""

    return (extractor or DEFAULT_CAPABILITY_EXTRACTOR).extract(tools)


def classify_tool(name: str, description: str | None = None) -> tuple[ActionKind, bool, bool]:
    """Return the inferred action kind and side-effect risk for one tool name.

    Exposed separately because a tool definition must carry its risk markers
    before a capability can be built from it.
    """

    inference = classify_tool_risk(name, description)
    return inference.action_kind, inference.state_changing, inference.destructive


class ToolRiskInference(NamedTuple):
    """The heuristic's answer for one tool, plus whether it found any evidence.

    ``known`` is false only when no name token or description token matched
    any classification rule -- the same condition ``SchemaCapabilityExtractor``
    already uses to record an ``unknown-action-kind`` capability. A tool in
    that state did not test negative for risk; nothing was found to test.
    Reusing the same condition here keeps the two views of one tool from
    silently disagreeing about whether it was actually classified.
    """

    action_kind: ActionKind
    state_changing: bool
    destructive: bool
    confidence: float
    known: bool


def classify_tool_risk(name: str, description: str | None = None) -> ToolRiskInference:
    """Return the heuristic classification for one tool, with its own confidence.

    Never authoritative and never a claim about ground truth: it is lexical
    evidence over a declared name and description, nothing more.
    """

    definition = ToolDefinition(name=name, description=description or None)
    action_kind, state_changing, destructive, confidence, _, _ = _classification_signals(
        definition
    )
    known = confidence > _NO_EVIDENCE_CONFIDENCE
    return ToolRiskInference(action_kind, state_changing, destructive, confidence, known)


__all__ = [
    "AUTHORITY_CONFIDENCE_THRESHOLD",
    "CAPABILITY_EXTRACTION_CONTRACT_VERSION",
    "CapabilityArgumentSurface",
    "CapabilityExtractor",
    "CapabilityParameter",
    "CapabilitySignal",
    "CapabilitySignalKind",
    "ExtractedCapability",
    "JsonSchemaType",
    "MAX_ENUM_VALUES",
    "MAX_PARAMETERS_PER_TOOL",
    "SchemaCapabilityExtractor",
    "ToolRiskInference",
    "ValueConstraints",
    "classify_tool",
    "classify_tool_risk",
    "extract_capabilities",
]
