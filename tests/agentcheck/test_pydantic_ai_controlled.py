"""``ControlledPydanticModel``: the offline stand-in for a PydanticAI target's provider.

The real bug this file pins down: PydanticAI wraps a non-object
``output_type`` (``bool``, ``int``, ...) in a single-key object
(``{"response": ...}``) before it will accept a reply for it, whether as text
or as a tool-call argument -- the SDK's own ``roulette_wheel.py`` example
(``output_type=bool``) exhausted its retry budget and terminated as
``provider_error: Exceeded maximum output retries`` under
``controlled_model=True`` because the earlier implementation built its reply
from the *raw* declared schema (``{"type": "boolean"}``) computed once at
construction time, instead of the framework's actual per-request negotiated
one. These tests exercise the fix directly, at the level of what
``model_request_parameters`` the real agent graph actually sends, rather than
only through a full end-to-end run.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.tools import ToolDefinition

from agentcheck.adapters.pydantic_ai_controlled import (
    ControlledPydanticModel,
    _negotiated_schema,
    _output_tool_call,
    _schema_instance,
    controlled_output_text,
)


def _params(**overrides: Any) -> ModelRequestParameters:
    return ModelRequestParameters(**overrides)


# --- controlled_output_text / _schema_instance: unchanged, direct schema ----


def test_controlled_output_text_is_deterministic_for_an_object_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"approved": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["approved", "reason"],
    }
    first = controlled_output_text(schema)
    second = controlled_output_text(schema)
    assert first == second
    assert json.loads(first) == {"approved": False, "reason": "agentcheck-controlled-value"}


def test_controlled_output_text_is_neutral_without_a_schema() -> None:
    assert "acknowledged" in controlled_output_text(None)
    assert "acknowledged" in controlled_output_text({})


# --- _negotiated_schema: the actual bug -------------------------------------


def test_negotiated_schema_is_the_wrapped_object_for_a_scalar_output_type() -> None:
    """This is the exact shape a real bool-output_type agent negotiates."""

    params = _params(
        output_object=OutputObjectDefinition(
            json_schema={
                "type": "object",
                "properties": {"response": {"type": "boolean"}},
                "required": ["response"],
            },
            name="bool",
        ),
        allow_text_output=True,
    )

    schema = _negotiated_schema(params)

    assert schema is not None
    assert schema["type"] == "object"
    assert "response" in schema["properties"]


def test_negotiated_schema_is_none_for_a_plain_string_output_type() -> None:
    assert _negotiated_schema(_params()) is None


# --- request(): the fix, exercised end-to-end at the Model level -----------


def test_request_replies_with_the_negotiated_wrapped_shape_not_the_raw_one() -> None:
    """Constructed with the *unwrapped* bool schema (what the old, buggy
    design would have cached) -- request() must still reply in the wrapped
    shape the framework actually asked for in model_request_parameters,
    because that is what PydanticAI's own output validator checks against."""

    model = ControlledPydanticModel({"type": "boolean"})
    params = _params(
        output_object=OutputObjectDefinition(
            json_schema={
                "type": "object",
                "properties": {"response": {"type": "boolean"}},
                "required": ["response"],
            },
            name="bool",
        ),
        allow_text_output=True,
    )

    response = asyncio.run(model.request([], None, params))

    assert len(response.parts) == 1
    body = json.loads(response.parts[0].content)
    assert body == {"response": False}


def test_request_falls_back_to_the_constructor_schema_without_a_negotiated_one() -> None:
    """No output_object at all (a plain-string output_type): the neutral or
    constructor-derived text is still exactly what it always was."""

    model = ControlledPydanticModel({"type": "object", "properties": {"x": {"type": "string"}}})

    response = asyncio.run(model.request([], None, _params()))

    assert len(response.parts) == 1
    assert json.loads(response.parts[0].content) == {"x": "agentcheck-controlled-value"}


# --- _output_tool_call: the tool-based output mode --------------------------


def test_output_tool_call_is_built_when_text_output_is_not_allowed() -> None:
    instance = {"response": False}

    call = _output_tool_call(
        _params(
            allow_text_output=False,
            output_tools=[
                ToolDefinition(
                    name="final_result",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {"response": {"type": "boolean"}},
                    },
                )
            ],
        ),
        instance,
    )

    assert call is not None
    assert call.tool_name == "final_result"
    assert call.args_as_dict() == instance


def test_output_tool_call_is_none_when_text_output_is_allowed() -> None:
    assert (
        _output_tool_call(
            _params(
                allow_text_output=True,
                output_tools=[ToolDefinition(name="final_result", parameters_json_schema={})],
            ),
            {"response": False},
        )
        is None
    )


def test_output_tool_call_is_none_without_an_instance() -> None:
    assert (
        _output_tool_call(
            _params(allow_text_output=False, output_tools=[ToolDefinition(name="x", parameters_json_schema={})]),
            None,
        )
        is None
    )


def test_request_calls_the_output_tool_when_text_output_is_disallowed() -> None:
    model = ControlledPydanticModel(None)
    params = _params(
        output_object=OutputObjectDefinition(
            json_schema={
                "type": "object",
                "properties": {"response": {"type": "boolean"}},
                "required": ["response"],
            },
            name="bool",
        ),
        allow_text_output=False,
        output_tools=[
            ToolDefinition(
                name="final_result",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"response": {"type": "boolean"}},
                },
            )
        ],
    )

    response = asyncio.run(model.request([], None, params))

    assert len(response.parts) == 1
    assert response.parts[0].part_kind == "tool-call"
    assert response.parts[0].args_as_dict() == {"response": False}


def test_schema_instance_derives_a_value_matching_the_dollar_defs_case() -> None:
    schema = {
        "$defs": {"Choice": {"enum": ["yes", "no"]}},
        "properties": {"pick": {"$ref": "#/$defs/Choice"}},
        "required": ["pick"],
        "type": "object",
    }

    assert _schema_instance(schema) == {"pick": "yes"}
