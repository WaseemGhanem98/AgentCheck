"""The controlled model must unlock evaluation without loosening any guarantee.

It replaces the target's provider so a run can reach a behavioural verdict
instead of stopping at the provider boundary. These tests pin the properties
that make the resulting verdicts trustworthy: it is opt-in, deterministic,
offline, and it never calls a tool -- because a model that chose actions would
be authoring the behaviour under test rather than observing it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, function_tool
from pydantic import BaseModel

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.adapters.controlled_model import ControlledModel, controlled_output_text


class Decision(BaseModel):
    approved: bool
    score: int
    notes: list[str]
    label: str


def _schema(agent: Agent) -> Any:
    return OpenAIAgentsAdapter().inspect(agent).interface.output_schema.value


def test_controlled_output_satisfies_the_targets_declared_schema() -> None:
    """The body must parse into the target's own output_type, not merely look valid."""

    agent = Agent(
        name="Reviewer", instructions="Review.", output_type=Decision, model="gpt-4.1-mini"
    )

    body = controlled_output_text(_schema(agent))

    # Round-trips through the target's real pydantic model, which is exactly
    # what the SDK does with a provider response.
    parsed = Decision.model_validate_json(body)
    assert isinstance(parsed.approved, bool)
    assert isinstance(parsed.score, int)
    assert parsed.notes == []


def test_controlled_output_is_deterministic() -> None:
    """Same schema in, byte-identical body out: no sampling, no clock, no RNG."""

    agent = Agent(
        name="Reviewer", instructions="Review.", output_type=Decision, model="gpt-4.1-mini"
    )
    schema = _schema(agent)

    assert controlled_output_text(schema) == controlled_output_text(schema)
    assert ControlledModel(schema)._body == ControlledModel(schema)._body


def test_plain_text_agent_gets_neutral_text_not_invented_json() -> None:
    agent = Agent(name="Chat", instructions="Chat.", model="gpt-4.1-mini")

    body = controlled_output_text(_schema(agent))

    assert body
    with pytest.raises(json.JSONDecodeError):
        json.loads(body)


def test_unrepresentable_schema_falls_back_instead_of_inventing_a_value() -> None:
    """A schema outside the supported subset must not produce a fake instance."""

    assert controlled_output_text({"type": "object", "properties": None}) is not None
    body = controlled_output_text(
        {"type": "object", "required": ["x"], "properties": {"x": {"type": "unknown-kind"}}}
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(body)


def test_controlled_model_is_opt_in(tmp_path: Path) -> None:
    """Default config must leave the target's own model in place."""

    assert AgentCheckConfig().controlled_model is False


def test_prepared_agent_uses_the_controlled_model_only_when_requested() -> None:
    """prepare() substitutes the model on the clone, never on the original agent."""

    @function_tool
    def send_notice(recipient: str) -> str:
        """Send a notice."""
        raise AssertionError("original handler must never run")

    original_model = "gpt-4.1-mini"
    agent = Agent(
        name="Notifier",
        instructions="Notify.",
        tools=[send_notice],
        output_type=Decision,
        model=original_model,
    )

    class _Gateway:
        async def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("no tool call is expected")

    default = OpenAIAgentsAdapter().prepare(agent, _Gateway())
    assert default.runtime_agent.model == original_model

    controlled = OpenAIAgentsAdapter().prepare(agent, _Gateway(), controlled_model=True)
    assert isinstance(controlled.runtime_agent.model, ControlledModel)
    # The authentic target object is never mutated.
    assert agent.model == original_model


def test_controlled_model_never_calls_a_tool() -> None:
    """The model emits one message and no tool call.

    A controlled model that selected tools would be authoring the agent's
    decisions, so any verdict would describe AgentCheck rather than the target.
    """

    import asyncio

    from agents.models.interface import ModelTracing

    model = ControlledModel({"type": "object", "properties": {}, "required": []})
    response = asyncio.run(
        model.get_response(
            system_instructions=None,
            input="anything",
            model_settings=None,
            tools=[],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
        )
    )

    assert len(response.output) == 1
    assert response.output[0].type == "message"
    assert not any(getattr(item, "type", "") == "function_call" for item in response.output)


def test_streaming_is_refused_rather_than_faked() -> None:
    model = ControlledModel(None)
    with pytest.raises(NotImplementedError):
        model.stream_response()
