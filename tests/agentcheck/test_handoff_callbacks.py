"""Phase 5B: proven on_handoff context assignments without executing callbacks."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import random
from collections.abc import AsyncIterator
from typing import Any

import pytest
from agents import Agent, Model, function_tool, handoff
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.tool import Tool
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.adapters.base import decode_topology
from agentcheck.adapters.openai_handoff_effects import (
    AgentCheckRunContext,
    analyze_on_handoff_callback,
    apply_context_assignments,
)
from agentcheck.domain import CanonicalEventType, RunTermination
from agentcheck.generate import build_account_support_suite

PINNED_SINGLE_AGENT_SPEC_ID = "agentspec-482269daa366d4ff8f81b74e"
PINNED_HAPPY_LOOKUP_FINGERPRINT = (
    "sha256:adb6e0fb51b9c2311d640bd66f5fe76bad2216ecb6b13add3eccc6063820c539"
)


def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        arguments=json.dumps(arguments),
        call_id=call_id,
        name=name,
        type="function_call",
        status="completed",
    )


def _message(text: str, item_id: str = "message-1") -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=item_id,
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


class ScriptedModel(Model):
    def __init__(self, outputs: list[list[Any]]) -> None:
        self.outputs = outputs
        self.calls = 0

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
    ) -> ModelResponse:
        del (
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        output = self.outputs[self.calls]
        self.calls += 1
        return ModelResponse(
            output=output,
            usage=Usage(),
            response_id=None,
            request_id=None,
            raw_usage=None,
        )

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[TResponseStreamEvent]:
        del args, kwargs
        raise NotImplementedError


class RecordingGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def invoke(
        self, tool_name: str, arguments: dict[str, Any], world_state: Any
    ) -> Any:
        del world_state
        self.calls.append((tool_name, arguments))
        return {"ok": True}


async def _official_style_on_handoff(context: Any) -> None:
    flight_number = f"FLT-{random.randint(100, 999)}"
    context.context.flight_number = flight_number


async def _callback_tripwire(context: Any) -> None:
    raise AssertionError("original on_handoff ran")


def _issue_codes(agent: Any) -> list[str]:
    report = OpenAIAgentsAdapter().preflight(agent)
    return [issue.code for issue in report.issues]


def _seat_graph() -> tuple[Any, Any, dict[str, int]]:
    tripwire = {"tool": 0}

    @function_tool
    def update_seat(confirmation_number: str, new_seat: str) -> str:
        """Update a seat."""

        tripwire["tool"] += 1
        raise AssertionError("original handler ran")

    seat = Agent(
        name="Seat Booking Agent",
        instructions="Book seats.",
        tools=[update_seat],
        model=ScriptedModel(
            [
                [
                    _tool_call(
                        "update_seat",
                        {"confirmation_number": "c1", "new_seat": "12A"},
                        "call-2",
                    )
                ],
                [_message("Seat updated.")],
            ]
        ),
    )
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[
            handoff(
                seat,
                on_handoff=_official_style_on_handoff,
                tool_name_override="transfer_to_seat_booking_agent",
            )
        ],
        model=ScriptedModel(
            [[_tool_call("transfer_to_seat_booking_agent", {}, "call-1")]]
        ),
    )
    return triage, seat, tripwire


def test_official_style_callback_is_recognized_without_executing_randint() -> None:
    calls = {"randint": 0}
    original = random.randint

    def counting(low: int, high: int) -> int:
        calls["randint"] += 1
        return original(low, high)

    random.randint = counting  # type: ignore[method-assign]
    try:
        analysis = analyze_on_handoff_callback(
            _official_style_on_handoff, location="test.on_handoff"
        )
    finally:
        random.randint = original  # type: ignore[method-assign]
    assert analysis.issue is None
    assert calls["randint"] == 0
    assert len(analysis.assignments) == 1
    assert analysis.assignments[0].field == "flight_number"
    assert analysis.assignments[0].value == "FLT-100"


def test_supported_callback_graph_passes_preflight() -> None:
    triage, _, _ = _seat_graph()
    assert _issue_codes(triage) == []


def test_prepare_does_not_capture_original_callback() -> None:
    triage, _, _ = _seat_graph()
    original = triage.handoffs[0]
    prepared = OpenAIAgentsAdapter().prepare(triage, RecordingGateway())
    safe = prepared.runtime_agent.handoffs[0]
    assert safe is not original
    assert safe.on_invoke_handoff is not original.on_invoke_handoff
    closed = inspect.getclosurevars(safe.on_invoke_handoff)
    assert _official_style_on_handoff not in closed.nonlocals.values()
    assert _official_style_on_handoff not in closed.globals.values()


def test_agentcheck_owned_effect_sets_context_and_skips_original_callback() -> None:
    triage, _, tripwire = _seat_graph()
    adapter = OpenAIAgentsAdapter()
    prepared = adapter.prepare(triage, RecordingGateway())
    original_code = _official_style_on_handoff.__code__
    _official_style_on_handoff.__code__ = _callback_tripwire.__code__
    try:
        bag = prepared.metadata["run_context"]
        assert isinstance(bag, AgentCheckRunContext)
        run = asyncio.run(
            adapter.run(prepared, "change my seat", run_id="run-callback", max_turns=6)
        )
    finally:
        _official_style_on_handoff.__code__ = original_code
    assert run.termination == RunTermination.COMPLETED
    assert tripwire["tool"] == 0
    assert getattr(bag, "flight_number") == "FLT-100"
    handoffs = [
        event for event in run.events if event.event_type == CanonicalEventType.HANDOFF
    ]
    assert len(handoffs) == 1
    payload = handoffs[0].payload
    assert payload["from_agent"] == "Triage Agent"
    assert payload["to_agent"] == "Seat Booking Agent"
    assert payload["handoff_tool_name"] == "transfer_to_seat_booking_agent"
    assert payload["call_id"] == "call-1"
    assert payload["ignored"] is False
    assert payload["callback_effect"] == "context_assignment"
    assert payload["context_assignments"] == [
        {"field": "flight_number", "value": "FLT-100"}
    ]
    encoded = json.dumps(payload)
    assert "random.randint" not in encoded
    assert "on_seat_booking_handoff" not in encoded
    assert inspect.getsource(_official_style_on_handoff) not in encoded


def test_replay_prepare_reproduces_the_same_assignment() -> None:
    adapter = OpenAIAgentsAdapter()
    first = adapter.prepare(_seat_graph()[0], RecordingGateway())
    first_run = asyncio.run(
        adapter.run(first, "change my seat", run_id="run-replay-a", max_turns=6)
    )
    second = adapter.prepare(_seat_graph()[0], RecordingGateway())
    second_run = asyncio.run(
        adapter.run(second, "change my seat", run_id="run-replay-b", max_turns=6)
    )
    first_payload = next(
        event.payload
        for event in first_run.events
        if event.event_type == CanonicalEventType.HANDOFF
    )
    second_payload = next(
        event.payload
        for event in second_run.events
        if event.event_type == CanonicalEventType.HANDOFF
    )
    assert first_payload["context_assignments"] == second_payload["context_assignments"]
    assert getattr(second.metadata["run_context"], "flight_number") == "FLT-100"


def test_unsupported_callback_shapes_fail_closed() -> None:
    destination = Agent(name="Dest", instructions="d")

    async def deleted(context: Any) -> None:
        del context

    async def awaited(context: Any) -> None:
        await asyncio.sleep(0)
        context.context.flag = "ready"

    async def branched(context: Any) -> None:
        if True:
            context.context.flag = "ready"

    async def nested_state(context: Any) -> None:
        context.context.state.flight_number = "FLT-100"

    async def helper(context: Any) -> None:
        open("/dev/null", encoding="utf-8").close()
        context.context.flag = "ready"

    cases = [
        deleted,
        awaited,
        branched,
        nested_state,
        helper,
        lambda context: None,
    ]
    for callback in cases:
        triage = Agent(
            name="Triage Agent",
            instructions="Route.",
            handoffs=[handoff(destination, on_handoff=callback)],
        )
        assert "handoff_callback" in _issue_codes(triage)


def test_uninspectable_source_fails_closed() -> None:
    namespace: dict[str, Any] = {}
    exec(
        "async def callback(context):\n    context.context.flag = 'ready'\n",
        namespace,
    )
    destination = Agent(name="Dest", instructions="d")
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[handoff(destination, on_handoff=namespace["callback"])],
    )
    assert "handoff_callback" in _issue_codes(triage)


def test_wrapped_and_partial_callbacks_fail_closed() -> None:
    destination = Agent(name="Dest", instructions="d")

    async def plain(context: Any) -> None:
        context.context.flag = "ready"

    @functools.wraps(plain)
    async def wrapped(context: Any) -> None:
        return await plain(context)

    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[handoff(destination, on_handoff=wrapped)],
    )
    assert "handoff_callback" in _issue_codes(triage)

    partial = functools.partial(plain)
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[handoff(destination, on_handoff=partial)],
    )
    codes = _issue_codes(triage)
    assert "handoff_callback" in codes or "unrecognized_handoff" in codes


def test_static_handoffs_without_callback_remain_supported() -> None:
    destination = Agent(name="Billing Agent", instructions="b")
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[handoff(destination, tool_name_override="transfer_to_billing_agent")],
    )
    assert _issue_codes(triage) == []
    topology = OpenAIAgentsAdapter().describe_topology(triage)
    assert topology is not None
    decoded = decode_topology(topology)
    assert "context_assignments" not in decoded["agents"][0]["handoffs"][0]


def test_single_agent_spec_and_suite_fingerprints_remain_stable() -> None:
    @function_tool
    def get_weather(city: str) -> str:
        """Return weather."""

        return "sunny"

    agent = Agent(
        name="Pin Agent", instructions="You report weather.", tools=[get_weather]
    )
    spec = OpenAIAgentsAdapter().inspect(agent, source="pin:agent.py:agent")
    assert spec.spec_id == PINNED_SINGLE_AGENT_SPEC_ID
    suite = build_account_support_suite(seed=1729)
    assert suite[0].fingerprint == PINNED_HAPPY_LOOKUP_FINGERPRINT


def test_tool_handlers_and_credentials_stay_isolated() -> None:
    triage, _, tripwire = _seat_graph()
    adapter = OpenAIAgentsAdapter()
    prepared = adapter.prepare(triage, RecordingGateway())
    run = asyncio.run(
        adapter.run(prepared, "change my seat", run_id="run-isolation", max_turns=6)
    )
    assert run.termination == RunTermination.COMPLETED
    assert tripwire["tool"] == 0
    assert prepared.metadata.get("run_context") is not None


def test_apply_context_assignments_only_mutates_the_bag() -> None:
    bag = AgentCheckRunContext()
    analysis = analyze_on_handoff_callback(
        _official_style_on_handoff, location="test.on_handoff"
    )
    assert analysis.issue is None
    apply_context_assignments(bag, analysis.assignments)
    assert bag.flight_number == "FLT-100"


def test_topology_includes_assignments_and_decode_round_trips() -> None:
    triage, _, _ = _seat_graph()
    topology = OpenAIAgentsAdapter().describe_topology(triage)
    assert topology is not None
    decoded = decode_topology(topology)
    edge = decoded["agents"][0]["handoffs"][0]
    assert edge["context_assignments"] == [
        {"field": "flight_number", "value": "FLT-100"}
    ]
    with pytest.raises(ValueError, match="unknown fields"):
        decode_topology(
            {
                "framework": "openai_agents",
                "agents": [
                    {
                        "name": "Triage Agent",
                        "location": "agent",
                        "model": None,
                        "instructions_static": True,
                        "tool_names": [],
                        "handoffs": [
                            {
                                "tool_name": "transfer",
                                "target_agent": "Seat",
                                "location": "agent.handoffs[0]",
                                "issue_codes": [],
                                "source": "async def callback",
                            }
                        ],
                    }
                ],
            }
        )
