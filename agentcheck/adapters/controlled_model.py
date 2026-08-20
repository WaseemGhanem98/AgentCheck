"""Deterministic offline model for evaluating a real agent without a provider.

Evaluation of an external target used to stop at the provider boundary: the
target's own ``Model`` points at OpenAI or a local server, so a run without
credentials (or with egress denied) ends in ``INFRA_ERROR`` before any agent
behaviour is observed.

This model replaces the provider for the duration of a run. It keeps the real
SDK orchestration and the authentic target ``Agent`` -- instructions, tool
schemas, handoff topology, and ``output_type`` are the target's own -- and only
substitutes the thing the target does not own: the language model.

Deliberately neutral, and that is a correctness property rather than a
limitation. The model is the agent's decision-maker, so a script that made it
choose a bad action would be authoring the misbehaviour and any resulting
verdict would describe this file, not the target. It therefore emits one
cooperative, in-contract response and calls no tools. What that exercises is
real: the SDK runs the genuine agent, AgentCheck's gateway mediates the tool
surface, and the target's own declared contracts (its output schema, its
forbidden-call constraints) are evaluated against the result.

Determinism comes from deriving the response from the target's declared output
schema; nothing is sampled and no clock, RNG, or network is consulted.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from agents import Model
from agents.items import ModelResponse, TResponseInputItem
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.usage import Usage
from openai.types.responses import ResponseOutputMessage, ResponseOutputText


CONTROLLED_MODEL_NAME = "agentcheck-controlled"

_NEUTRAL_TEXT = (
    "AgentCheck controlled model: acknowledged. No tool call was requested."
)
_MAX_INSTANCE_DEPTH = 6


def _instance_for(schema: Mapping[str, Any], definitions: Mapping[str, Any], depth: int) -> Any:
    """Build one deterministic value satisfying ``schema``.

    Covers the JSON Schema subset pydantic emits for a normal ``output_type``.
    Anything outside it yields ``None`` so the caller can fall back to text
    rather than inventing a value that only looks valid.
    """

    if depth > _MAX_INSTANCE_DEPTH:
        return None

    reference = schema.get("$ref")
    if isinstance(reference, str):
        name = reference.rsplit("/", 1)[-1]
        target = definitions.get(name)
        if not isinstance(target, Mapping):
            return None
        return _instance_for(target, definitions, depth + 1)

    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][0]
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
        items = schema.get("items")
        minimum = schema.get("minItems")
        if not isinstance(minimum, int) or minimum <= 0:
            return []
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
    if declared == "null":
        return None
    return None


def controlled_output_text(output_schema: Mapping[str, Any] | None) -> str:
    """Render the single response body this model returns.

    With a declared structured output the body is a deterministic instance of
    the target's own schema, so the SDK can parse it into the target's
    ``output_type`` exactly as a real provider response would be parsed.
    """

    if not output_schema:
        return _NEUTRAL_TEXT
    definitions = output_schema.get("$defs")
    instance = _instance_for(
        output_schema, definitions if isinstance(definitions, Mapping) else {}, 0
    )
    if instance is None:
        return _NEUTRAL_TEXT
    return json.dumps(instance, ensure_ascii=False, allow_nan=False, sort_keys=True)


class ControlledModel(Model):
    """Offline, deterministic stand-in for the target's provider."""

    def __init__(self, output_schema: Mapping[str, Any] | None = None) -> None:
        self._body = controlled_output_text(output_schema)
        self.calls = 0

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> ModelResponse:
        del (
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            kwargs,
        )
        self.calls += 1
        message = ResponseOutputMessage(
            id=f"agentcheck-controlled-{self.calls}",
            content=[
                ResponseOutputText(annotations=[], text=self._body, type="output_text")
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        return ModelResponse(
            output=[message],
            usage=Usage(),
            response_id=None,
            request_id=None,
            raw_usage=None,
        )

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> Any:  # pragma: no cover - evaluation never streams
        del args, kwargs
        raise NotImplementedError(
            "the AgentCheck controlled model does not implement streaming"
        )


__all__ = ["CONTROLLED_MODEL_NAME", "ControlledModel", "controlled_output_text"]
