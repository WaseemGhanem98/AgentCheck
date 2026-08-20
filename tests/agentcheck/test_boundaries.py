from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest
from jsonschema import SchemaError  # type: ignore[import-untyped]
from pydantic import ValidationError

from agentcheck.domain import (
    ActionKind,
    AgentProperty,
    AgentSpec,
    CapabilitiesSpec,
    Capability,
    GuardrailsSpec,
    IdentitySpec,
    InspectionProvenance,
    InstructionsSpec,
    InterfaceSpec,
    ObservabilitySpec,
    OracleStrength,
    PoliciesSpec,
    RuntimeSpec,
    Scenario,
    SourceKind,
    SourceReference,
    SpecEvidence,
    ToolDefinition,
    ToolPoliciesSpec,
    ToolsSpec,
    WorkflowsSpec,
)
from agentcheck.generate import lint_scenario, lint_suite
from agentcheck.generate.boundaries import (
    BOUNDARY_CONTRACT_VERSION,
    MAX_BOUNDARIES_PER_PARAMETER,
    MAX_BOUNDARIES_PER_TOOL,
    MAX_BOUNDARY_SCENARIOS_PER_SPEC,
    BoundaryKind,
    SchemaBoundary,
    build_boundary_scenarios,
    build_zero_input_cases,
    derive_boundaries,
    unsupported_boundary_reasons,
)
from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
SEED = 1729


def _tool(name: str = "update_ticket", **schema: Any) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Synthetic {name} tool.",
        input_schema=schema or {"type": "object"},
        replaceable=True,
    )


def _object_tool(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
    additional: bool | None = False,
    name: str = "update_ticket",
) -> ToolDefinition:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required is not None:
        schema["required"] = required
    if additional is not None:
        schema["additionalProperties"] = additional
    return _tool(name, **schema)


def _property(value: Any) -> AgentProperty[Any]:
    return AgentProperty(
        value=value,
        source=SourceReference(kind=SourceKind.TOOL_SCHEMA, locator="test"),
        confidence=1.0,
        evidence=(SpecEvidence(evidence_id="evidence-test", summary="Synthetic."),),
        inferred=True,
        authoritative=False,
    )


def _spec(*tools: ToolDefinition) -> AgentSpec:
    return AgentSpec(
        spec_id="agentspec-boundary-test",
        identity=IdentitySpec(
            name=_property("Boundary Target"),
            framework=_property("openai_agents"),
            framework_version=_property(None),
            provider=_property(None),
            model=_property(None),
        ),
        interface=InterfaceSpec(
            entrypoint=_property("agent.py:agent"),
            input_modalities=_property(("text",)),
            output_modalities=_property(("text",)),
            input_schema=_property(None),
            output_schema=_property(None),
            interactive=_property(True),
        ),
        instructions=InstructionsSpec(
            system=_property("Serve requests."), developer=_property(None)
        ),
        capabilities=CapabilitiesSpec(
            items=tuple(
                _property(
                    Capability(
                        capability_id=f"tool:{tool.name}",
                        name=tool.name,
                        description=f"Capability for {tool.name}.",
                        action_kind=ActionKind.OTHER,
                    )
                )
                for tool in tools
            )
        ),
        tools=ToolsSpec(items=tuple(_property(tool) for tool in tools)),
        tool_policies=ToolPoliciesSpec(),
        guardrails=GuardrailsSpec(),
        workflows=WorkflowsSpec(),
        policies=PoliciesSpec(),
        runtime=RuntimeSpec(
            max_model_turns=_property(None),
            max_tool_calls=_property(None),
            timeout_seconds=_property(None),
            token_budget=_property(None),
            cost_budget_usd=_property(None),
        ),
        observability=ObservabilitySpec(
            supported_event_types=_property(("tool_attempt",)),
            usage_metrics=_property(()),
            provider_request_ids=_property(True),
            source_event_links=_property(True),
        ),
        provenance=InspectionProvenance(
            inspector="test",
            inspector_version="1",
            inspected_at=NOW,
            target="tests",
            sources=(SourceReference(kind=SourceKind.TOOL_SCHEMA, locator="test"),),
        ),
    )


def _kinds(tool: ToolDefinition) -> list[BoundaryKind]:
    return [boundary.kind for boundary in derive_boundaries(tool)]


def _by_kind(tool: ToolDefinition, kind: BoundaryKind) -> SchemaBoundary:
    matching = [item for item in derive_boundaries(tool) if item.kind is kind]
    assert matching, f"expected a {kind.value} boundary"
    return matching[0]


# --- one test per supported schema keyword -----------------------------------


def test_required_keyword_produces_a_missing_property_boundary() -> None:
    tool = _object_tool({"ticket_id": {"type": "string"}}, required=["ticket_id"])

    boundary = _by_kind(tool, BoundaryKind.MISSING_REQUIRED_PROPERTY)

    assert boundary.value_omitted is True
    assert boundary.invalid_value is None
    assert "ticket_id" not in boundary.arguments
    assert boundary.baseline_arguments == {"ticket_id": "agentcheck-boundary-value"}
    assert boundary.pointer == "/properties/ticket_id"


def test_type_keyword_produces_a_wrong_type_boundary() -> None:
    tool = _object_tool({"ticket_id": {"type": "string"}}, required=["ticket_id"])

    boundary = _by_kind(tool, BoundaryKind.WRONG_TYPE)

    assert boundary.invalid_value == 12_345
    assert boundary.arguments["ticket_id"] == 12_345


def test_enum_keyword_produces_an_out_of_enum_boundary() -> None:
    tool = _object_tool({"priority": {"enum": ["low", "high"]}}, required=["priority"])

    boundary = _by_kind(tool, BoundaryKind.OUT_OF_ENUM)

    assert boundary.invalid_value == "agentcheck-out-of-enum"
    assert boundary.baseline_arguments == {"priority": "low"}


def test_numeric_enum_boundary_uses_a_numeric_out_of_range_value() -> None:
    tool = _object_tool({"level": {"enum": [1, 2, 3]}}, required=["level"])

    boundary = _by_kind(tool, BoundaryKind.OUT_OF_ENUM)

    assert boundary.invalid_value == 4


def test_minimum_keyword_produces_a_below_minimum_boundary() -> None:
    tool = _object_tool(
        {"weight": {"type": "integer", "minimum": 1}}, required=["weight"]
    )

    boundary = _by_kind(tool, BoundaryKind.BELOW_MINIMUM)

    assert boundary.invalid_value == 0


def test_maximum_keyword_produces_an_above_maximum_boundary() -> None:
    tool = _object_tool(
        {"weight": {"type": "integer", "maximum": 5}}, required=["weight"]
    )

    boundary = _by_kind(tool, BoundaryKind.ABOVE_MAXIMUM)

    assert boundary.invalid_value == 6


def test_exclusive_numeric_keywords_are_covered() -> None:
    tool = _object_tool(
        {
            "weight": {
                "type": "integer",
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 10,
            }
        },
        required=["weight"],
    )

    kinds = _kinds(tool)

    assert BoundaryKind.BELOW_MINIMUM in kinds
    assert BoundaryKind.ABOVE_MAXIMUM in kinds


def test_min_length_keyword_produces_a_too_short_boundary() -> None:
    tool = _object_tool(
        {"ticket_id": {"type": "string", "minLength": 4}}, required=["ticket_id"]
    )

    boundary = _by_kind(tool, BoundaryKind.BELOW_MIN_LENGTH)

    assert boundary.invalid_value == "xxx"


def test_max_length_keyword_produces_a_too_long_boundary() -> None:
    tool = _object_tool(
        {"ticket_id": {"type": "string", "maxLength": 4}}, required=["ticket_id"]
    )

    boundary = _by_kind(tool, BoundaryKind.ABOVE_MAX_LENGTH)

    assert boundary.invalid_value == "xxxxx"
    assert boundary.baseline_arguments == {"ticket_id": "agen"}


def test_additional_properties_keyword_produces_one_tool_level_boundary() -> None:
    tool = _object_tool({"ticket_id": {"type": "string"}}, required=["ticket_id"])

    boundaries = [
        item
        for item in derive_boundaries(tool)
        if item.kind is BoundaryKind.ADDITIONAL_PROPERTY
    ]

    assert len(boundaries) == 1
    assert boundaries[0].parameter == "agentcheck_unexpected_property"
    assert boundaries[0].pointer == "/additionalProperties"
    assert boundaries[0].arguments["agentcheck_unexpected_property"] == (
        "agentcheck-boundary-value"
    )


def test_permissive_additional_properties_produces_no_such_boundary() -> None:
    tool = _object_tool(
        {"ticket_id": {"type": "string"}}, required=["ticket_id"], additional=True
    )

    assert BoundaryKind.ADDITIONAL_PROPERTY not in _kinds(tool)


# --- proof, emptiness, and fail-closed behavior ------------------------------


def test_every_generated_object_is_proved_invalid_and_the_baseline_valid() -> None:
    tool = _object_tool(
        {
            "ticket_id": {"type": "string", "minLength": 3, "maxLength": 12},
            "priority": {"enum": ["low", "high"]},
            "weight": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        required=["ticket_id", "priority", "weight"],
    )
    validator = offline_validator(tool.input_schema)

    boundaries = derive_boundaries(tool)

    assert boundaries
    for boundary in boundaries:
        assert validator.is_valid(boundary.baseline_arguments), boundary.kind
        assert not validator.is_valid(boundary.arguments), boundary.kind


def test_schema_without_usable_constraints_produces_no_cases() -> None:
    tool = _tool("act", type="object", properties={"free": {}})

    assert derive_boundaries(tool) == ()
    assert any(
        "no established type" in reason for reason in unsupported_boundary_reasons(tool)
    )


def test_schema_without_properties_produces_no_cases() -> None:
    tool = _tool("act", type="object")

    assert derive_boundaries(tool) == ()
    assert "the declared schema exposes no parameters" in unsupported_boundary_reasons(
        tool
    )


def test_empty_object_schema_yields_a_zero_input_invocation() -> None:
    tool = _object_tool(
        {}, required=[], additional=False, name="fetch_random_xkcd"
    )
    spec = _spec(tool)

    assert derive_boundaries(tool) == ()
    cases = build_zero_input_cases(spec, seed=SEED)
    assert len(cases) == 1
    tool_name, scenario = cases[0]
    assert tool_name == "fetch_random_xkcd"
    assert scenario.scenario_id == "zero-input-fetch-random-xkcd"
    assert scenario.tool_fixtures[0].arguments_match == {}
    assert scenario.required_tool_behavior[0].arguments_match == {}
    assert scenario.required_tool_behavior[0].min_calls == 1
    assert scenario.forbidden_tool_behavior == ()
    assert "source:zero_input_invocation" in scenario.dimension_tags
    assert scenario.oracle_provenance[0].strength is OracleStrength.TOOL_CONTRACT
    assert scenario.oracle_provenance[0].supports_hard_failure is True
    assert lint_scenario(scenario, spec) == ()
    assert scenario.fingerprint == scenario.expected_fingerprint()


def test_empty_schema_that_rejects_empty_object_is_not_zero_input() -> None:
    tool = _tool(
        "act", type="object", properties={}, minProperties=1, additionalProperties=False
    )

    assert derive_boundaries(tool) == ()
    assert build_zero_input_cases(_spec(tool), seed=SEED) == ()


def test_parameterized_tools_do_not_gain_zero_input_cases() -> None:
    tool = _object_tool({"ticket_id": {"type": "string"}}, required=["ticket_id"])
    boundaries = derive_boundaries(tool)

    assert boundaries
    assert build_zero_input_cases(_spec(tool), seed=SEED) == ()
    assert {item.kind for item in boundaries} == {
        BoundaryKind.MISSING_REQUIRED_PROPERTY,
        BoundaryKind.WRONG_TYPE,
        BoundaryKind.ADDITIONAL_PROPERTY,
    }


def test_unconstructable_required_parameter_yields_no_cases() -> None:
    tool = _object_tool(
        {"payload": {"allOf": [{"type": "object"}]}}, required=["payload"]
    )

    assert derive_boundaries(tool) == ()
    assert any(
        "no in-contract value could be constructed" in reason
        for reason in unsupported_boundary_reasons(tool)
    )


def test_non_local_reference_is_refused_rather_than_skipped() -> None:
    tool = _object_tool(
        {"payload": {"$ref": "https://example.test/schema.json"}}, required=["payload"]
    )

    with pytest.raises(UnsafeSchemaReference):
        derive_boundaries(tool)
    with pytest.raises(UnsafeSchemaReference):
        unsupported_boundary_reasons(tool)
    with pytest.raises(UnsafeSchemaReference):
        build_boundary_scenarios(_spec(tool), seed=SEED)


def test_invalid_schema_is_refused() -> None:
    tool = _tool("act", type="object", properties={"value": {"type": 17}})

    with pytest.raises(SchemaError):
        derive_boundaries(tool)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("pattern", "^[a-z]+$"),
        ("format", "email"),
        ("multipleOf", 3),
        ("minItems", 2),
        ("uniqueItems", True),
    ],
)
def test_unsupported_keywords_are_named_rather_than_silently_ignored(
    keyword: str, value: Any
) -> None:
    tool = _object_tool(
        {"ticket_id": {"type": "string"}, "extra": {"type": "array", keyword: value}}
    )

    reasons = unsupported_boundary_reasons(tool)

    assert any(keyword in reason for reason in reasons), reasons


def test_nested_object_properties_are_reported_as_untraversed() -> None:
    tool = _object_tool(
        {"payload": {"type": "object", "properties": {"inner": {"type": "string"}}}}
    )

    assert any(
        "nested properties" in reason for reason in unsupported_boundary_reasons(tool)
    )


# --- bounds and ordering ------------------------------------------------------


def test_generation_is_bounded_per_parameter() -> None:
    tool = _object_tool(
        {
            "value": {
                "type": "string",
                "minLength": 4,
                "maxLength": 8,
                "enum": ["abcd", "abcde"],
            }
        },
        required=["value"],
    )

    per_parameter = [
        item for item in derive_boundaries(tool) if item.parameter == "value"
    ]

    assert len(per_parameter) <= MAX_BOUNDARIES_PER_PARAMETER
    assert any(
        "per-parameter cap" in reason for reason in unsupported_boundary_reasons(tool)
    )


def test_generation_is_bounded_per_tool() -> None:
    properties = {
        f"field_{index:02d}": {"type": "integer", "minimum": 1, "maximum": 5}
        for index in range(20)
    }
    tool = _object_tool(properties, required=sorted(properties))

    boundaries = derive_boundaries(tool)

    assert len(boundaries) == MAX_BOUNDARIES_PER_TOOL
    assert any(
        "per-tool cap" in reason for reason in unsupported_boundary_reasons(tool)
    )


def test_generation_is_bounded_per_spec() -> None:
    tools = [
        _object_tool(
            {
                f"field_{index:02d}": {"type": "integer", "minimum": 1, "maximum": 5}
                for index in range(6)
            },
            required=[f"field_{index:02d}" for index in range(6)],
            name=f"tool_{number:02d}",
        )
        for number in range(12)
    ]

    scenarios = build_boundary_scenarios(_spec(*tools), seed=SEED)

    assert len(scenarios) == MAX_BOUNDARY_SCENARIOS_PER_SPEC


def test_tool_level_boundary_survives_truncation() -> None:
    properties = {
        f"field_{index:02d}": {"type": "integer", "minimum": 1, "maximum": 5}
        for index in range(20)
    }
    tool = _object_tool(properties, required=sorted(properties))

    assert derive_boundaries(tool)[0].kind is BoundaryKind.ADDITIONAL_PROPERTY


def test_ordering_is_stable_under_reordered_schema_declarations() -> None:
    forward = _object_tool(
        {"alpha": {"type": "string"}, "zulu": {"type": "string"}},
        required=["alpha", "zulu"],
    )
    reversed_declaration = _object_tool(
        {"zulu": {"type": "string"}, "alpha": {"type": "string"}},
        required=["zulu", "alpha"],
    )

    assert [item.parameter for item in derive_boundaries(forward)] == [
        item.parameter for item in derive_boundaries(reversed_declaration)
    ]


# --- scenario construction ----------------------------------------------------


def test_every_generated_scenario_passes_lint_against_its_spec() -> None:
    tool = _object_tool(
        {
            "ticket_id": {"type": "string", "minLength": 3},
            "priority": {"enum": ["low", "high"]},
        },
        required=["ticket_id", "priority"],
    )
    spec = _spec(tool)

    scenarios = build_boundary_scenarios(spec, seed=SEED)

    assert scenarios
    for scenario in scenarios:
        assert lint_scenario(scenario, spec) == ()


def test_scenarios_stay_on_the_v1_contract_and_verify_their_fingerprint() -> None:
    spec = _spec(_object_tool({"ticket_id": {"type": "string"}}, required=["ticket_id"]))

    scenarios = build_boundary_scenarios(spec, seed=SEED)

    for scenario in scenarios:
        assert scenario.contract_version == "agentcheck.scenario.v1"
        assert scenario.fingerprint == scenario.expected_fingerprint()
        assert scenario.generation_seed == SEED
        restored = Scenario.model_validate_json(scenario.model_dump_json())
        assert restored == scenario


def test_scenarios_are_prohibitions_backed_by_a_tool_contract_oracle() -> None:
    spec = _spec(_object_tool({"priority": {"enum": ["low"]}}, required=["priority"]))

    scenarios = build_boundary_scenarios(spec, seed=SEED)
    out_of_enum = [
        scenario
        for scenario in scenarios
        if "schema:out_of_enum" in scenario.dimension_tags
    ]

    assert len(out_of_enum) == 1
    scenario = out_of_enum[0]
    (constraint,) = scenario.forbidden_tool_behavior
    assert constraint.arguments_match == {"priority": "agentcheck-out-of-enum"}
    assert (constraint.min_calls, constraint.max_calls) == (0, 0)
    assert scenario.required_tool_behavior == ()
    assert scenario.allowed_tool_behavior == ()
    (oracle,) = scenario.oracle_provenance
    assert oracle.strength is OracleStrength.TOOL_CONTRACT
    assert oracle.supports_hard_failure is True
    assert oracle.confidence == 1.0
    assert constraint.oracle_ids == (oracle.oracle_id,)


def test_missing_required_scenario_forbids_the_call_entirely() -> None:
    spec = _spec(_object_tool({"ticket_id": {"type": "string"}}, required=["ticket_id"]))

    scenarios = build_boundary_scenarios(spec, seed=SEED)
    missing = [
        scenario
        for scenario in scenarios
        if "schema:missing_required_property" in scenario.dimension_tags
    ]

    (scenario,) = missing
    (constraint,) = scenario.forbidden_tool_behavior
    assert constraint.arguments_match == {}
    assert "is not available in this conversation" in (
        scenario.conversation_turns[0].content
    )


def test_every_scenario_carries_a_permissive_in_contract_fixture() -> None:
    spec = _spec(
        _object_tool(
            {"ticket_id": {"type": "string", "minLength": 2}}, required=["ticket_id"]
        )
    )

    for scenario in build_boundary_scenarios(spec, seed=SEED):
        (fixture,) = scenario.tool_fixtures
        assert fixture.arguments_match == {}
        assert fixture.outcome.state_effects == ()


def test_oracle_evidence_chains_back_to_capability_extraction() -> None:
    spec = _spec(_object_tool({"ticket_id": {"type": "string"}}, required=["ticket_id"]))

    scenario = build_boundary_scenarios(spec, seed=SEED)[0]

    (oracle,) = scenario.oracle_provenance
    assert oracle.evidence_ids
    assert all(item.startswith("evidence-") for item in oracle.evidence_ids)


def test_generation_is_deterministic_for_one_seed() -> None:
    spec = _spec(
        _object_tool(
            {
                "ticket_id": {"type": "string", "minLength": 3},
                "weight": {"type": "integer", "minimum": 1, "maximum": 9},
            },
            required=["ticket_id", "weight"],
        )
    )

    first = build_boundary_scenarios(spec, seed=SEED)
    second = build_boundary_scenarios(spec, seed=SEED)

    assert [item.fingerprint for item in first] == [item.fingerprint for item in second]
    assert [item.scenario_id for item in first] == [
        item.scenario_id for item in second
    ]


def test_a_different_seed_changes_every_fingerprint() -> None:
    spec = _spec(_object_tool({"ticket_id": {"type": "string"}}, required=["ticket_id"]))

    first = build_boundary_scenarios(spec, seed=SEED)
    other = build_boundary_scenarios(spec, seed=SEED + 1)

    assert {item.fingerprint for item in first}.isdisjoint(
        {item.fingerprint for item in other}
    )


def test_scenario_ids_and_fingerprints_are_unique_within_a_suite() -> None:
    spec = _spec(
        _object_tool(
            {
                "ticket_id": {"type": "string", "minLength": 3},
                "priority": {"enum": ["low", "high"]},
            },
            required=["ticket_id", "priority"],
            name="update_ticket",
        ),
        _object_tool(
            {"account_id": {"type": "string"}},
            required=["account_id"],
            name="delete_account",
        ),
    )

    scenarios = build_boundary_scenarios(spec, seed=SEED)

    assert len({item.scenario_id for item in scenarios}) == len(scenarios)
    assert len({item.fingerprint for item in scenarios}) == len(scenarios)
    assert all(issues == () for _, issues in lint_suite(scenarios, spec))


def test_a_spec_without_tools_produces_no_scenarios() -> None:
    assert build_boundary_scenarios(_spec(), seed=SEED) == ()


def test_an_out_of_range_seed_is_rejected() -> None:
    spec = _spec(_object_tool({"ticket_id": {"type": "string"}}, required=["ticket_id"]))

    with pytest.raises(ValueError, match="seed must be"):
        build_boundary_scenarios(spec, seed=-1)


# --- contract and safety ------------------------------------------------------


def test_schema_boundary_round_trips_and_rejects_unknown_fields() -> None:
    tool = _object_tool({"ticket_id": {"type": "string"}}, required=["ticket_id"])

    boundary = derive_boundaries(tool)[0]
    restored = SchemaBoundary.model_validate_json(boundary.model_dump_json())

    assert restored == boundary
    assert boundary.schema_version == BOUNDARY_CONTRACT_VERSION
    payload = json.loads(boundary.model_dump_json())
    payload["future_field"] = True
    with pytest.raises(ValidationError, match="future_field"):
        SchemaBoundary.model_validate_json(json.dumps(payload))


def test_generated_content_contains_no_plausible_personal_data() -> None:
    spec = _spec(
        _object_tool(
            {
                "email": {"type": "string"},
                "account_id": {"type": "string"},
                "token": {"type": "string"},
            },
            required=["email", "account_id", "token"],
        )
    )

    scenarios = build_boundary_scenarios(spec, seed=SEED)
    rendered = "\n".join(scenario.model_dump_json() for scenario in scenarios)

    assert scenarios
    assert "@" not in rendered
    assert "http" not in rendered
    # Every synthetic string that reaches a fixture or transcript is one of the
    # obviously-generated constants, never something resembling real user data.
    values = {
        boundary.invalid_value
        for tool in (item.value for item in spec.tools.items)
        for boundary in derive_boundaries(tool)
    } | {
        value
        for tool in (item.value for item in spec.tools.items)
        for boundary in derive_boundaries(tool)
        for value in boundary.baseline_arguments.values()
    }
    text_values = {value for value in values if isinstance(value, str)}
    assert text_values
    assert all(
        value.startswith("agentcheck-") or set(value) <= {"x"}
        for value in text_values
    ), text_values


def test_generation_never_touches_the_input_schema_or_spawns_a_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("boundary generation must not execute anything")

    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    monkeypatch.setattr(subprocess, "run", _forbidden)
    tool = _object_tool(
        {"ticket_id": {"type": "string", "minLength": 3}}, required=["ticket_id"]
    )
    before = json.dumps(tool.input_schema, sort_keys=True)

    build_boundary_scenarios(_spec(tool), seed=SEED)

    assert json.dumps(tool.input_schema, sort_keys=True) == before
