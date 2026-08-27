from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from agentcheck.domain import ActionKind, SourceKind, ToolDefinition
from agentcheck.inspect.capabilities import (
    AUTHORITY_CONFIDENCE_THRESHOLD,
    CAPABILITY_EXTRACTION_CONTRACT_VERSION,
    MAX_ENUM_VALUES,
    MAX_PARAMETERS_PER_TOOL,
    CapabilitySignal,
    CapabilitySignalKind,
    ExtractedCapability,
    JsonSchemaType,
    SchemaCapabilityExtractor,
    classify_tool,
    extract_capabilities,
)


def _tool(
    name: str,
    *,
    description: str | None = None,
    schema: dict[str, Any] | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        input_schema=schema if schema is not None else {"type": "object"},
        replaceable=True,
    )


def _object_schema(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
    additional: bool | None = False,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required is not None:
        schema["required"] = required
    if additional is not None:
        schema["additionalProperties"] = additional
    return schema


def _parameters(tool: ToolDefinition) -> dict[str, Any]:
    extracted = extract_capabilities([tool])[0]
    return {item.name: item for item in extracted.arguments.parameters}


def test_zero_tool_agent_extracts_no_capabilities() -> None:
    assert extract_capabilities([]) == ()


def test_single_tool_records_capability_identity_and_contract_version() -> None:
    (extracted,) = extract_capabilities(
        [_tool("lookup_account", description="Look up one account.")]
    )

    assert extracted.schema_version == CAPABILITY_EXTRACTION_CONTRACT_VERSION
    assert extracted.tool_name == "lookup_account"
    assert extracted.capability.capability_id == "tool:lookup_account"
    assert extracted.capability.name == "Lookup Account"
    assert extracted.capability.description == "Look up one account."


def test_multiple_tools_preserve_input_order() -> None:
    tools = [_tool("update_email"), _tool("lookup_account"), _tool("delete_account")]

    extracted = extract_capabilities(tools)

    assert [item.tool_name for item in extracted] == [
        "update_email",
        "lookup_account",
        "delete_account",
    ]


def test_extraction_is_deterministic_across_repeated_calls() -> None:
    tool = _tool(
        "update_email",
        description="Replace the stored address.",
        schema=_object_schema(
            {
                "new_email": {"type": "string", "format": "email"},
                "account_id": {"type": "string", "minLength": 1},
                "retries": {"type": "integer", "minimum": 0, "maximum": 5},
            },
            required=["account_id", "new_email"],
        ),
    )

    first = extract_capabilities([tool])[0]
    second = SchemaCapabilityExtractor().extract([tool])[0]

    assert first.canonical_json() == second.canonical_json()


def test_parameters_are_ordered_by_name_regardless_of_schema_order() -> None:
    forward = _tool(
        "sync",
        schema=_object_schema(
            {"zulu": {"type": "string"}, "alpha": {"type": "string"}},
            required=["alpha"],
        ),
    )
    reversed_declaration = _tool(
        "sync",
        schema=_object_schema(
            {"alpha": {"type": "string"}, "zulu": {"type": "string"}},
            required=["alpha"],
        ),
    )

    assert [item.name for item in extract_capabilities([forward])[0].arguments.parameters] == [
        "alpha",
        "zulu",
    ]
    assert (
        extract_capabilities([forward])[0].arguments.canonical_json()
        == extract_capabilities([reversed_declaration])[0].arguments.canonical_json()
    )


def test_required_and_optional_arguments_are_separated() -> None:
    tool = _tool(
        "update_email",
        schema=_object_schema(
            {
                "account_id": {"type": "string"},
                "new_email": {"type": "string"},
                "notify": {"type": "boolean"},
            },
            required=["account_id", "new_email"],
        ),
    )

    surface = extract_capabilities([tool])[0].arguments

    assert surface.schema_known is True
    assert [item.name for item in surface.required_parameters] == [
        "account_id",
        "new_email",
    ]
    assert [item.name for item in surface.optional_parameters] == ["notify"]
    assert surface.additional_properties_allowed is False


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("string", JsonSchemaType.STRING),
        ("integer", JsonSchemaType.INTEGER),
        ("number", JsonSchemaType.NUMBER),
        ("boolean", JsonSchemaType.BOOLEAN),
        ("object", JsonSchemaType.OBJECT),
        ("array", JsonSchemaType.ARRAY),
        ("null", JsonSchemaType.NULL),
    ],
)
def test_primitive_json_schema_types_are_read(
    declared: str, expected: JsonSchemaType
) -> None:
    tool = _tool("act", schema=_object_schema({"value": {"type": declared}}))

    parameter = _parameters(tool)["value"]

    assert parameter.types_known is True
    assert parameter.types == (expected,)


def test_union_type_lists_and_optional_branches_are_resolved() -> None:
    tool = _tool(
        "act",
        schema=_object_schema(
            {
                "listed": {"type": ["string", "null"]},
                "optional": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            }
        ),
    )

    parameters = _parameters(tool)

    assert parameters["listed"].types == (JsonSchemaType.NULL, JsonSchemaType.STRING)
    assert parameters["optional"].types == (JsonSchemaType.NULL, JsonSchemaType.STRING)
    assert parameters["optional"].unsupported_constructs == ("anyOf",)


def test_array_and_object_structure_is_summarized() -> None:
    tool = _tool(
        "act",
        schema=_object_schema(
            {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                    "uniqueItems": True,
                },
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "zip": {"type": "string"}},
                },
            }
        ),
    )

    parameters = _parameters(tool)

    tags = parameters["tags"].constraints
    assert tags.item_types == (JsonSchemaType.STRING,)
    assert (tags.min_items, tags.max_items, tags.unique_items) == (1, 5, True)
    assert parameters["address"].constraints.nested_property_names == ("city", "zip")


def test_enum_and_const_values_are_captured() -> None:
    tool = _tool(
        "act",
        schema=_object_schema(
            {
                "mode": {"enum": ["soft", "hard"]},
                "kind": {"const": "account"},
            }
        ),
    )

    parameters = _parameters(tool)

    assert parameters["mode"].constraints.enum_values == ("soft", "hard")
    assert parameters["mode"].types == (JsonSchemaType.STRING,)
    assert parameters["kind"].constraints.enum_values == ("account",)


def test_enum_values_are_bounded_and_truncation_is_recorded() -> None:
    values = [f"value-{index}" for index in range(MAX_ENUM_VALUES + 5)]
    tool = _tool("act", schema=_object_schema({"mode": {"enum": values}}))

    constraints = _parameters(tool)["mode"].constraints

    assert len(constraints.enum_values or ()) == MAX_ENUM_VALUES
    assert constraints.enum_truncated is True


def test_numeric_and_string_bounds_are_captured() -> None:
    tool = _tool(
        "act",
        schema=_object_schema(
            {
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 11,
                    "multipleOf": 2,
                },
                "label": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": 32,
                    "pattern": "^[a-z]+$",
                    "format": "email",
                },
            }
        ),
    )

    parameters = _parameters(tool)

    count = parameters["count"].constraints
    assert (count.minimum, count.maximum) == (1.0, 10.0)
    assert (count.exclusive_minimum, count.exclusive_maximum) == (0.0, 11.0)
    assert count.multiple_of == 2.0
    label = parameters["label"].constraints
    assert (label.min_length, label.max_length) == (2, 32)
    assert (label.pattern, label.string_format) == ("^[a-z]+$", "email")


def test_parameter_count_is_bounded_and_truncation_is_recorded() -> None:
    properties = {
        f"field_{index:03d}": {"type": "string"}
        for index in range(MAX_PARAMETERS_PER_TOOL + 10)
    }
    tool = _tool("act", schema=_object_schema(properties))

    surface = extract_capabilities([tool])[0].arguments

    assert len(surface.parameters) == MAX_PARAMETERS_PER_TOOL
    assert surface.truncated is True


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"value": {"oneOf": [{"type": "string"}, {"minimum": 1}]}}},
        {"type": "object", "properties": {"value": {"allOf": [{"type": "string"}]}}},
        {"type": "object", "properties": {"value": {"not": {"type": "string"}}}},
        {"type": "object", "properties": {"value": {"$ref": "#/$defs/Nested"}}},
        {"type": "object", "properties": {"value": {"type": "unheard-of"}}},
        {"type": "object", "properties": {"value": {}}},
    ],
)
def test_unresolvable_parameter_schemas_stay_unknown(schema: dict[str, Any]) -> None:
    parameter = _parameters(_tool("act", schema=schema))["value"]

    assert parameter.types_known is False
    assert parameter.types == ()


def test_unsupported_constructs_are_named() -> None:
    tool = _tool(
        "act",
        schema=_object_schema({"value": {"$ref": "#/$defs/Nested", "allOf": [{}]}}),
    )

    parameter = _parameters(tool)["value"]

    assert parameter.unsupported_constructs == ("$ref", "allOf")


def test_non_local_schema_reference_is_refused_and_recorded() -> None:
    tool = _tool(
        "update_record",
        schema=_object_schema({"value": {"$ref": "https://example.test/schema.json"}}),
    )

    extracted = extract_capabilities([tool])[0]

    assert extracted.arguments.schema_known is False
    assert extracted.arguments.parameters == ()
    assert extracted.arguments.unsupported_constructs == ("non_local_schema_reference",)
    assert [item.path for item in extracted.unknowns] == ["capabilities.items[0].arguments"]
    assert any(
        signal.kind is CapabilitySignalKind.SCHEMA_UNREADABLE
        for signal in extracted.signals
    )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": ["not", "a", "mapping"]},
        {"type": "object", "properties": "nonsense"},
    ],
)
def test_malformed_properties_make_the_argument_surface_unknown(
    schema: dict[str, Any],
) -> None:
    extracted = extract_capabilities([_tool("update_record", schema=schema)])[0]

    assert extracted.arguments.schema_known is False
    assert extracted.unknowns[0].path == "capabilities.items[0].arguments"
    assert extracted.unknowns[0].confidence == 0.0


def test_malformed_required_list_is_recorded_without_inventing_requirements() -> None:
    tool = _tool(
        "update_record",
        schema=_object_schema({"value": {"type": "string"}}, required=None),
    )
    malformed = tool.model_copy(
        update={
            "input_schema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": "value",
            }
        }
    )

    extracted = extract_capabilities([malformed])[0]

    assert extracted.arguments.parameters[0].required is False
    assert "malformed_required" in extracted.arguments.unsupported_constructs
    assert [item.path for item in extracted.unknowns] == ["capabilities.items[0].arguments"]


def test_absent_schema_yields_no_parameters_without_failing() -> None:
    extracted = extract_capabilities([_tool("act", schema={})])[0]

    assert extracted.arguments.schema_known is True
    assert extracted.arguments.parameters == ()
    assert extracted.arguments.additional_properties_allowed is None


@pytest.mark.parametrize(
    ("name", "action_kind", "state_changing", "destructive"),
    [
        ("delete_account", ActionKind.DELETE, True, True),
        ("purge_records", ActionKind.DELETE, True, True),
        ("cancel_subscription", ActionKind.MODIFY, True, True),
        ("update_email", ActionKind.MODIFY, True, False),
        ("create_ticket", ActionKind.CREATE, True, False),
        ("write", ActionKind.CREATE, True, False),
        ("save_draft", ActionKind.CREATE, True, False),
        ("send_receipt", ActionKind.SEND, True, False),
        ("schedule_visit", ActionKind.SCHEDULE, True, False),
        ("lookup_account", ActionKind.LOOKUP, False, False),
        ("retrieve_invoice", ActionKind.RETRIEVE, False, False),
        ("summarize_thread", ActionKind.SUMMARIZE, False, False),
        ("frobnicate", ActionKind.OTHER, False, False),
    ],
)
def test_each_action_kind_and_risk_classification_is_reachable(
    name: str, action_kind: ActionKind, state_changing: bool, destructive: bool
) -> None:
    capability = extract_capabilities([_tool(name)])[0].capability

    assert capability.action_kind is action_kind
    assert capability.state_changing is state_changing
    assert capability.destructive is destructive
    assert classify_tool(name) == (action_kind, state_changing, destructive)


def test_a_bare_write_tool_is_not_silently_classified_read_only() -> None:
    """Reproduces a real independent target's file-write tool: named exactly
    "write", with a docstring stating it overwrites existing files. Before
    "write" was added to the name vocabulary, this fell through every rule to
    OTHER/read-only/confidence 0.3 -- the single lowest-confidence,
    least-scrutinized classification -- for a tool that unconditionally
    mutates local files."""

    capability = extract_capabilities(
        [
            _tool(
                "write",
                description=(
                    "Writes a file to the local filesystem. This tool will "
                    "overwrite the existing file if there is one at the "
                    "provided path."
                ),
            )
        ]
    )[0]

    assert capability.capability.state_changing is True
    assert capability.capability.action_kind is ActionKind.CREATE


def test_classification_matches_the_phase_one_name_vocabulary() -> None:
    """The gateway marks ambiguous state from these flags, so they must not drift."""

    assert classify_tool("delete-account") == classify_tool("delete_account")
    assert classify_tool("get_and_delete") == (ActionKind.DELETE, True, True)


def test_description_can_corroborate_but_never_reclassify() -> None:
    plain = extract_capabilities([_tool("update_email")])[0]
    corroborated = extract_capabilities(
        [_tool("update_email", description="Update the stored address.")]
    )[0]
    misleading = extract_capabilities(
        [_tool("update_email", description="Permanently delete the account.")]
    )[0]

    assert plain.confidence == pytest.approx(0.6)
    assert corroborated.confidence == pytest.approx(0.7)
    assert misleading.capability.action_kind is ActionKind.MODIFY
    assert misleading.capability.destructive is False
    assert misleading.confidence == pytest.approx(0.6)


def test_unclassifiable_tool_stays_unknown_rather_than_guessed() -> None:
    extracted = extract_capabilities(
        [_tool("frobnicate", description="Permanently deletes things.")]
    )[0]

    assert extracted.capability.action_kind is ActionKind.OTHER
    assert extracted.capability.state_changing is False
    assert extracted.capability.destructive is False
    assert extracted.confidence == pytest.approx(0.3)
    assert [item.path for item in extracted.unknowns] == [
        "capabilities.items[0].action_kind"
    ]
    assert extracted.unknowns[0].source.kind is SourceKind.UNKNOWN
    assert any(
        signal.kind is CapabilitySignalKind.NO_CLASSIFICATION_EVIDENCE
        for signal in extracted.signals
    )


def test_classification_never_reaches_the_authority_threshold() -> None:
    names = [
        "delete_account",
        "cancel_subscription",
        "update_email",
        "create_ticket",
        "send_receipt",
        "schedule_visit",
        "lookup_account",
        "retrieve_invoice",
        "summarize_thread",
        "frobnicate",
    ]

    extracted = extract_capabilities(
        [_tool(name, description=f"{name.split('_')[0]} something") for name in names]
    )

    assert all(item.confidence < AUTHORITY_CONFIDENCE_THRESHOLD for item in extracted)
    assert all(item.classification_inferred for item in extracted)


def test_lexical_signals_cannot_be_marked_authoritative() -> None:
    with pytest.raises(ValidationError, match="never authoritative"):
        CapabilitySignal(
            kind=CapabilitySignalKind.NAME_TOKEN,
            detail="delete",
            locator="tool:x.name",
            source={"kind": SourceKind.TOOL_SCHEMA, "locator": "tool:x.name"},
            confidence=1.0,
            authoritative=True,
        )


def test_unknown_sourced_signals_cannot_be_authoritative() -> None:
    with pytest.raises(ValidationError, match="cannot be authoritative"):
        CapabilitySignal(
            kind=CapabilitySignalKind.SCHEMA_PARAMETER,
            detail="one parameter",
            locator="tool:x.input_schema",
            source={"kind": SourceKind.UNKNOWN, "locator": "tool:x.input_schema"},
            confidence=1.0,
            authoritative=True,
        )


def test_extracted_capability_rejects_an_unbacked_authoritative_confidence() -> None:
    with pytest.raises(ValidationError, match="authoritative confidence threshold"):
        extract_capabilities([_tool("delete_account")])[
            0
        ].confidence = AUTHORITY_CONFIDENCE_THRESHOLD

    with pytest.raises(ValidationError, match="requires an authoritative signal"):
        extract_capabilities([_tool("delete_account")])[
            0
        ].classification_inferred = False


def test_schema_signals_are_authoritative_and_lexical_signals_are_not() -> None:
    tool = _tool(
        "delete_account",
        description="Permanently delete one account.",
        schema=_object_schema(
            {"account_id": {"type": "string", "minLength": 1}},
            required=["account_id"],
        ),
    )

    extracted = extract_capabilities([tool])[0]
    by_kind = {signal.kind: signal for signal in extracted.signals}

    assert by_kind[CapabilitySignalKind.NAME_TOKEN].authoritative is False
    assert by_kind[CapabilitySignalKind.DESCRIPTION_TOKEN].authoritative is False
    assert by_kind[CapabilitySignalKind.SCHEMA_PARAMETER].authoritative is True
    assert by_kind[CapabilitySignalKind.SCHEMA_PARAMETER].confidence == 1.0
    assert by_kind[CapabilitySignalKind.SCHEMA_CONSTRAINT].authoritative is True
    assert all(
        signal.source.kind is SourceKind.TOOL_SCHEMA
        for signal in extracted.signals
        if signal.authoritative
    )


def test_evidence_records_provenance_for_name_schema_and_parameters() -> None:
    tool = _tool(
        "delete_account",
        description="Permanently delete one account.",
        schema=_object_schema(
            {"account_id": {"type": "string"}}, required=["account_id"]
        ),
    )

    extracted = extract_capabilities([tool])[0]
    locators = [item.locator for item in extracted.evidence]

    assert locators == [
        "tool:delete_account",
        "tool:delete_account.name",
        "tool:delete_account.description",
        "tool:delete_account.input_schema",
        "tool:delete_account.input_schema#/properties/account_id",
    ]
    assert len({item.evidence_id for item in extracted.evidence}) == len(locators)
    assert "never authoritative" in extracted.evidence[1].summary
    assert "1 required and 0 optional parameter(s)" in extracted.evidence[3].summary
    assert "is required; string" in extracted.evidence[4].summary


def test_extracted_capability_round_trips_through_json() -> None:
    tool = _tool(
        "update_email",
        description="Update one stored address.",
        schema=_object_schema(
            {
                "account_id": {"type": "string", "minLength": 1},
                "mode": {"enum": ["soft", "hard"]},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            required=["account_id"],
        ),
    )

    extracted = extract_capabilities([tool])[0]
    restored = ExtractedCapability.model_validate_json(extracted.model_dump_json())

    assert restored == extracted
    assert restored.canonical_json() == extracted.canonical_json()


def test_extracted_capability_rejects_unknown_fields() -> None:
    payload = json.loads(extract_capabilities([_tool("act")])[0].model_dump_json())
    payload["future_field"] = True

    with pytest.raises(ValidationError, match="future_field"):
        ExtractedCapability.model_validate_json(json.dumps(payload))


def test_extraction_reads_only_and_never_mutates_the_input_schema() -> None:
    schema = _object_schema(
        {"account_id": {"type": "string"}}, required=["account_id"]
    )
    tool = _tool("delete_account", schema=schema)
    before = tool.input_schema.copy()

    extract_capabilities([tool])

    assert tool.input_schema == before
