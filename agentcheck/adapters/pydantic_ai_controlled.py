"""Deterministic offline model for evaluating a PydanticAI target without a provider.

The same reasoning as the OpenAI adapter's controlled model: this replaces the
one thing the target does not own, and it stays neutral on purpose. A model
scripted to choose actions would be authoring the behaviour under evaluation,
so any resulting verdict would describe this file rather than the target.

Determinism comes from deriving the reply from the target's declared output
schema. Nothing is sampled and no clock, RNG, or network is consulted.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.usage import RequestUsage


CONTROLLED_MODEL_NAME = "agentcheck-controlled"
_NEUTRAL_TEXT = "AgentCheck controlled model: acknowledged. No tool call was requested."
_MAX_INSTANCE_DEPTH = 6


def _instance_for(
    schema: Mapping[str, Any], definitions: Mapping[str, Any], depth: int
) -> Any:
    """One deterministic value satisfying ``schema``, or None when out of subset."""

    if depth > _MAX_INSTANCE_DEPTH:
        return None
    reference = schema.get("$ref")
    if isinstance(reference, str):
        target = definitions.get(reference.rsplit("/", 1)[-1])
        if not isinstance(target, Mapping):
            return None
        return _instance_for(target, definitions, depth + 1)
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    for keyword in ("anyOf", "oneOf"):
        options = schema.get(keyword)
        if isinstance(options, list) and options:
            for option in options:
                if isinstance(option, Mapping):
                    candidate = _instance_for(option, definitions, depth + 1)
                    if candidate is not None:
                        return candidate
            return None
    declared = schema.get("type")
    if isinstance(declared, list):
        declared = next((item for item in declared if item != "null"), None)
    if declared == "object":
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            return {}
        required = schema.get("required")
        names = required if isinstance(required, list) else list(properties)
        built: dict[str, Any] = {}
        for name in names:
            child = properties.get(name)
            if not isinstance(child, Mapping):
                return None
            value = _instance_for(child, definitions, depth + 1)
            if value is None and "null" not in str(child.get("type", "")):
                return None
            built[str(name)] = value
        return built
    if declared == "array":
        minimum = schema.get("minItems")
        if not isinstance(minimum, int) or minimum <= 0:
            return []
        items = schema.get("items")
        if not isinstance(items, Mapping):
            return None
        element = _instance_for(items, definitions, depth + 1)
        return None if element is None else [element] * minimum
    if declared == "string":
        return "agentcheck-controlled-value"
    if declared == "integer":
        return 0
    if declared == "number":
        return 0.0
    if declared == "boolean":
        return False
    return None


def _schema_instance(schema: Mapping[str, Any]) -> Any:
    """One deterministic value satisfying ``schema``, honoring its ``$defs``."""

    definitions = schema.get("$defs")
    return _instance_for(
        schema, definitions if isinstance(definitions, Mapping) else {}, 0
    )


def controlled_output_text(output_schema: Mapping[str, Any] | None) -> str:
    """The single text response body this model returns.

    Only reached when the framework will accept a plain text reply as the
    final output. PydanticAI's non-string ``output_type``s (``bool``, a
    dataclass, ...) instead require the reply to arrive as a call to a
    framework-synthesized tool -- see ``_output_tool_call`` -- so this
    function is never the right answer for those, and ``request`` below picks
    between the two based on what the framework actually asked for, not by
    guessing from the schema shape alone.
    """

    if not output_schema:
        return _NEUTRAL_TEXT
    instance = _schema_instance(output_schema)
    if instance is None:
        return _NEUTRAL_TEXT
    return json.dumps(instance, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _negotiated_schema(model_request_parameters: Any) -> Mapping[str, Any] | None:
    """The schema this exact request actually needs to satisfy, or ``None``.

    ``model_request_parameters.output_object.json_schema`` is what PydanticAI
    negotiated for *this* request given the target's ``output_type`` -- and it
    is not always the same shape as the raw type's own schema. A scalar or
    non-object ``output_type`` (``bool``, ``int``, ...) is wrapped in a
    single-key object (``{"response": ...}``, ``outer_typed_dict_key``) before
    PydanticAI will accept it as either text or a tool-call argument; an
    object ``output_type`` (a ``BaseModel``, a dataclass) is not. Building the
    reply from a schema computed once at construction time and cached (the
    previous design) got this wrong for the wrapped case: the raw schema for
    ``bool`` is ``{"type": "boolean"}``, but the framework was actually
    validating against ``{"type": "object", "properties": {"response": ...}}``,
    so a bare ``"false"`` reply could never satisfy it no matter the retry
    budget. Reading the schema fresh from what the framework passed into this
    exact call is authoritative by construction: it is the framework's own
    negotiated contract, not a guess reconstructed from the target's static
    declaration.
    """

    output_object = getattr(model_request_parameters, "output_object", None)
    schema = getattr(output_object, "json_schema", None) if output_object is not None else None
    return schema if isinstance(schema, Mapping) else None


def _output_tool_call(
    model_request_parameters: Any, instance: Any
) -> ToolCallPart | None:
    """A call to the framework's own synthesized output tool, or ``None``.

    PydanticAI represents a non-string ``output_type`` (``bool``, a
    dataclass, ``TypedDict``, ...) as a required call to a tool it builds
    itself -- ``final_result`` by default -- rather than as parseable text,
    whenever ``model_request_parameters.allow_text_output`` is ``False``; a
    ``TextPart`` reply can never satisfy that no matter its content or how
    many retries remain. The instance was already derived from
    ``_negotiated_schema`` (the same schema this tool's own
    ``parameters_json_schema`` describes), so this only re-packages it as tool
    arguments rather than as JSON text. The first output tool is used when
    more than one is offered (a ``Union`` output type), which is an arbitrary
    but deterministic and documented choice, not a guess at target behavior.
    """

    if getattr(model_request_parameters, "allow_text_output", True):
        return None
    output_tools = getattr(model_request_parameters, "output_tools", None)
    if not output_tools or not isinstance(instance, Mapping):
        return None
    return ToolCallPart(tool_name=output_tools[0].name, args=dict(instance))


class ControlledPydanticModel(Model):
    """Offline, deterministic stand-in for the target's provider."""

    def __init__(self, output_schema: Mapping[str, Any] | None = None) -> None:
        # Retained as the reply for a request the framework gives no
        # model_request_parameters.output_object to negotiate against (a
        # plain-string output_type, most commonly) -- see request() below,
        # which prefers the freshly negotiated schema whenever one exists.
        self._body = controlled_output_text(output_schema)
        self.calls = 0

    @property
    def model_name(self) -> str:
        return CONTROLLED_MODEL_NAME

    @property
    def system(self) -> str:
        return "agentcheck"

    async def request(
        self,
        messages: list[Any],
        model_settings: Any,
        model_request_parameters: Any,
    ) -> ModelResponse:
        del messages, model_settings
        self.calls += 1
        schema = _negotiated_schema(model_request_parameters)
        instance = _schema_instance(schema) if schema is not None else None
        tool_call = (
            _output_tool_call(model_request_parameters, instance)
            if instance is not None
            else None
        )
        if tool_call is not None:
            part: Any = tool_call
        elif instance is not None:
            part = TextPart(
                json.dumps(instance, ensure_ascii=False, allow_nan=False, sort_keys=True)
            )
        else:
            part = TextPart(self._body)
        return ModelResponse(
            parts=[part],
            usage=RequestUsage(input_tokens=0, output_tokens=0),
            model_name=CONTROLLED_MODEL_NAME,
        )


__all__ = [
    "CONTROLLED_MODEL_NAME",
    "ControlledPydanticModel",
    "controlled_output_text",
]
