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

from pydantic_ai.messages import ModelResponse, TextPart
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


def controlled_output_text(output_schema: Mapping[str, Any] | None) -> str:
    """The single response body this model returns."""

    if not output_schema:
        return _NEUTRAL_TEXT
    definitions = output_schema.get("$defs")
    instance = _instance_for(
        output_schema, definitions if isinstance(definitions, Mapping) else {}, 0
    )
    if instance is None:
        return _NEUTRAL_TEXT
    return json.dumps(instance, ensure_ascii=False, allow_nan=False, sort_keys=True)


class ControlledPydanticModel(Model):
    """Offline, deterministic stand-in for the target's provider."""

    def __init__(self, output_schema: Mapping[str, Any] | None = None) -> None:
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
        del messages, model_settings, model_request_parameters
        self.calls += 1
        return ModelResponse(
            parts=[TextPart(self._body)],
            usage=RequestUsage(input_tokens=0, output_tokens=0),
            model_name=CONTROLLED_MODEL_NAME,
        )


__all__ = [
    "CONTROLLED_MODEL_NAME",
    "ControlledPydanticModel",
    "controlled_output_text",
]
