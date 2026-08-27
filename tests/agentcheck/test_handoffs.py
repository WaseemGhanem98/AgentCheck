"""Phase 5A handoff support: graph preflight, safe reconstruction, evidence.

Every test here runs offline with scripted local models.  The load-bearing
assertions are the safety ones: no original tool handler, handoff routing
closure, or handoff callback is ever invoked, and unsupported handoff surface
fails closed with a precise issue code before any model execution.
"""

from __future__ import annotations

import asyncio
import json
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
from pydantic import BaseModel

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.adapters.base import decode_topology
from agentcheck.domain import (
    CanonicalEvent,
    CanonicalEventType,
    CanonicalRun,
    ConversationRole,
    ConversationTurn,
    OracleProvenance,
    OracleStrength,
    RunTermination,
    Scenario,
    ToolAttempt,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
    Verdict,
    utc_now,
)
from agentcheck.evaluate import evaluate_run
from agentcheck.generate import build_account_support_suite
from agentcheck.generate.lint import lint_scenario


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
    def __init__(self, result: Any = None) -> None:
        self.result = {"ok": True} if result is None else result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def invoke(
        self, tool_name: str, arguments: dict[str, Any], world_state: Any
    ) -> Any:
        del world_state
        self.calls.append((tool_name, arguments))
        return self.result


def _issue_codes(agent: Any) -> list[tuple[str, str | None]]:
    report = OpenAIAgentsAdapter().preflight(agent)
    return [(issue.code, issue.location) for issue in report.issues]


def _billing_pair(
    *,
    triage_outputs: list[list[Any]] | None = None,
    billing_outputs: list[list[Any]] | None = None,
) -> tuple[Any, Any, dict[str, int]]:
    tripwire = {"tool": 0, "callback": 0}

    @function_tool
    def lookup_invoice(invoice_id: str) -> str:
        """Look up one invoice."""

        tripwire["tool"] += 1
        raise AssertionError("original handler ran")

    billing = Agent(
        name="Billing Agent",
        instructions="Billing.",
        tools=[lookup_invoice],
        model=ScriptedModel(
            billing_outputs
            or [
                [_tool_call("lookup_invoice", {"invoice_id": "inv_42"}, "call-2")],
                [_message("Invoice inv_42 total is 42 dollars.")],
            ]
        ),
    )
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[handoff(billing, tool_name_override="transfer_to_billing_agent")],
        model=ScriptedModel(
            triage_outputs or [[_tool_call("transfer_to_billing_agent", {}, "call-1")]]
        ),
    )
    billing.handoffs.append(
        handoff(triage, tool_name_override="transfer_to_triage_agent")
    )
    return triage, billing, tripwire


def test_supported_static_handoff_graph_passes_preflight() -> None:
    triage, _, _ = _billing_pair()
    assert _issue_codes(triage) == []


def test_generic_handoffs_code_is_replaced_by_specific_codes() -> None:
    triage, _, _ = _billing_pair()

    async def callback(context: Any) -> None:
        del context

    triage.handoffs.append(handoff(Agent(name="X", instructions="x"), on_handoff=callback))
    codes = _issue_codes(triage)
    assert ("handoff_callback", "agent.handoffs[1].on_handoff") in codes
    assert all(code != "handoffs" for code, _ in codes)


def test_handoff_input_type_is_rejected_as_payload_schema() -> None:
    class Payload(BaseModel):
        reason: str

    async def callback(context: Any, payload: Payload) -> None:
        del context, payload

    destination = Agent(name="Dest", instructions="d")
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[handoff(destination, on_handoff=callback, input_type=Payload)],
    )
    codes = [code for code, _ in _issue_codes(triage)]
    assert "handoff_callback" in codes
    assert "handoff_input_schema" in codes


def test_handoff_input_filter_and_dynamic_enablement_are_rejected() -> None:
    destination = Agent(name="Dest", instructions="d")
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[
            handoff(destination, input_filter=lambda data: data),
            handoff(destination, is_enabled=lambda context, agent: True),
        ],
    )
    codes = _issue_codes(triage)
    assert ("handoff_input_filter", "agent.handoffs[0].input_filter") in codes
    assert ("dynamic_handoff_enablement", "agent.handoffs[1].is_enabled") in codes


def test_hand_built_handoff_is_unrecognized() -> None:
    async def invoke(context: Any, input_json: str) -> Any:
        del context, input_json
        raise AssertionError("hand-built handoff routing must never run")

    hand_built = Handoff(
        tool_name="transfer_to_mystery",
        tool_description="Hand-built.",
        input_json_schema={},
        on_invoke_handoff=invoke,
        agent_name="Mystery",
    )
    triage = Agent(name="Triage Agent", instructions="Route.", handoffs=[hand_built])
    codes = [code for code, _ in _issue_codes(triage)]
    assert codes == ["unrecognized_handoff"]


def test_plain_agent_entry_is_supported_and_subagents_get_full_preflight() -> None:
    def dynamic_instructions(context: Any, agent: Any) -> str:
        del context, agent
        return "dynamic"

    destination = Agent(name="Dest", instructions=dynamic_instructions)
    triage = Agent(name="Triage Agent", instructions="Route.", handoffs=[destination])
    codes = _issue_codes(triage)
    assert (
        "dynamic_instructions",
        "agent.handoffs[0].agent.instructions",
    ) in codes


def test_oversized_graph_fails_closed() -> None:
    chain = Agent(name="agent-tail", instructions="t")
    for index in range(20):
        chain = Agent(
            name=f"agent-{index}",
            instructions="x",
            handoffs=[handoff(chain)],
        )
    codes = [code for code, _ in _issue_codes(chain)]
    assert "handoff_graph_too_large" in codes


def test_cross_agent_duplicate_tool_and_agent_names_are_rejected() -> None:
    @function_tool(name_override="lookup")
    def lookup_one(a: str) -> str:
        """Duplicate one."""

        raise AssertionError("never runs")

    @function_tool(name_override="lookup")
    def lookup_two(a: str) -> str:
        """Duplicate two."""

        raise AssertionError("never runs")

    first = Agent(name="Shared Name", instructions="1", tools=[lookup_one])
    second = Agent(name="Shared Name", instructions="2", tools=[lookup_two])
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[handoff(first), handoff(second)],
    )
    codes = [code for code, _ in _issue_codes(triage)]
    assert "duplicate_agent_name" in codes
    assert "duplicate_tool_name" in codes


def test_handoff_tool_name_colliding_with_function_tool_is_rejected() -> None:
    @function_tool(name_override="transfer_to_billing_agent")
    def shadow(a: str) -> str:
        """Deliberate collision."""

        raise AssertionError("never runs")

    destination = Agent(name="Billing Agent", instructions="b")
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        tools=[shadow],
        handoffs=[handoff(destination, tool_name_override="transfer_to_billing_agent")],
    )
    codes = [code for code, _ in _issue_codes(triage)]
    assert "handoff_tool_name_collision" in codes


def test_nested_handoff_history_is_rejected() -> None:
    destination = Agent(name="Dest", instructions="d")
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[handoff(destination, nest_handoff_history=True)],
    )
    codes = _issue_codes(triage)
    assert (
        "handoff_history_override",
        "agent.handoffs[0].nest_handoff_history",
    ) in codes


def test_preflight_and_inspect_never_invoke_handoff_callbacks() -> None:
    called = {"n": 0}

    async def callback(context: Any) -> None:
        called["n"] += 1
        del context

    destination = Agent(name="Dest", instructions="d")
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[handoff(destination, on_handoff=callback)],
    )
    adapter = OpenAIAgentsAdapter()
    adapter.preflight(triage)
    adapter.inspect(triage, source="tests/agentcheck/test_handoffs.py")
    adapter.describe_topology(triage)
    assert called["n"] == 0


def test_prepare_reconstructs_graph_without_original_closures() -> None:
    triage, billing, tripwire = _billing_pair()
    adapter = OpenAIAgentsAdapter()
    gateway = RecordingGateway(result={"total": 42})

    prepared = adapter.prepare(triage, gateway)

    safe_triage = prepared.runtime_agent
    assert safe_triage is not triage
    assert len(safe_triage.handoffs) == 1
    safe_edge = safe_triage.handoffs[0]
    assert safe_edge is not triage.handoffs[0]
    assert safe_edge.on_invoke_handoff is not triage.handoffs[0].on_invoke_handoff
    assert safe_edge.input_filter is None
    assert safe_edge.is_enabled is True
    # Outside adapter.run() the AgentCheck invoker refuses to route at all.
    with pytest.raises(RuntimeError, match="outside adapter.run"):
        asyncio.run(safe_edge.on_invoke_handoff(None, ""))
    assert tripwire == {"tool": 0, "callback": 0}
    # The billing clone keeps its own scripted model object.
    safe_billing_names = {edge.agent_name for edge in safe_triage.handoffs}
    assert safe_billing_names == {"Billing Agent"}
    assert prepared.metadata["handoff_tool_names"] == (
        "transfer_to_billing_agent",
        "transfer_to_triage_agent",
    )
    assert billing.tools[0] is not None  # original graph left intact


def test_multi_hop_run_emits_handoff_events_and_attribution() -> None:
    triage, _, tripwire = _billing_pair()
    adapter = OpenAIAgentsAdapter()
    gateway = RecordingGateway(result={"total": 42})

    prepared = adapter.prepare(triage, gateway)
    run = asyncio.run(
        adapter.run(prepared, "What is invoice inv_42?", run_id="run-handoff", max_turns=6)
    )

    assert run.termination == RunTermination.COMPLETED
    assert tripwire == {"tool": 0, "callback": 0}
    assert gateway.calls == [("lookup_invoice", {"invoice_id": "inv_42"})]

    handoffs = [
        event for event in run.events if event.event_type == CanonicalEventType.HANDOFF
    ]
    assert len(handoffs) == 1
    payload = handoffs[0].payload
    assert payload["from_agent"] == "Triage Agent"
    assert payload["to_agent"] == "Billing Agent"
    assert payload["handoff_tool_name"] == "transfer_to_billing_agent"
    assert payload["arguments"] == {}
    assert payload["call_id"] == "call-1"
    assert payload["ignored"] is False
    assert handoffs[0].source_event_ids

    attempts = [
        event.payload
        for event in run.events
        if event.event_type == CanonicalEventType.TOOL_ATTEMPT
    ]
    assert attempts and attempts[0]["agent_name"] == "Billing Agent"
    results = [
        event.payload
        for event in run.events
        if event.event_type == CanonicalEventType.TOOL_RESULT
    ]
    assert results and results[0]["agent_name"] == "Billing Agent"


def test_redundant_same_turn_handoffs_are_recorded_as_ignored() -> None:
    docs = Agent(name="Docs Agent", instructions="d", model=ScriptedModel([[_message("hi")]]))
    billing = Agent(
        name="Billing Agent",
        instructions="b",
        model=ScriptedModel([[_message("Handled by billing.")]]),
    )
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[
            handoff(billing, tool_name_override="transfer_to_billing_agent"),
            handoff(docs, tool_name_override="transfer_to_docs_agent"),
        ],
        model=ScriptedModel(
            [
                [
                    _tool_call("transfer_to_billing_agent", {}, "call-1"),
                    _tool_call("transfer_to_docs_agent", {}, "call-2"),
                ]
            ]
        ),
    )
    adapter = OpenAIAgentsAdapter()
    prepared = adapter.prepare(triage, RecordingGateway())
    run = asyncio.run(adapter.run(prepared, "route me", run_id="run-multi", max_turns=4))

    handoffs = [
        event.payload
        for event in run.events
        if event.event_type == CanonicalEventType.HANDOFF
    ]
    executed = [payload for payload in handoffs if payload["ignored"] is False]
    ignored = [payload for payload in handoffs if payload["ignored"] is True]
    assert len(executed) == 1
    assert executed[0]["to_agent"] == "Billing Agent"
    assert len(ignored) == 1
    assert ignored[0]["handoff_tool_name"] == "transfer_to_docs_agent"
    assert ignored[0]["to_agent"] == "Docs Agent"
    assert ignored[0]["call_id"] == "call-2"
    assert run.final_output == "Handled by billing."


def test_tool_owned_by_another_agent_is_blocked_and_recorded() -> None:
    tripwire = {"docs": 0}

    @function_tool
    def search_docs(query: str) -> str:
        """Search docs."""

        tripwire["docs"] += 1
        raise AssertionError("original docs handler ran")

    docs = Agent(
        name="Docs Agent",
        instructions="Docs.",
        tools=[search_docs],
        model=ScriptedModel([[_message("unused")]]),
    )
    billing = Agent(
        name="Billing Agent",
        instructions="Billing.",
        model=ScriptedModel(
            [[_tool_call("search_docs", {"query": "invoice"}, "call-2")]]
        ),
    )
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[
            handoff(billing, tool_name_override="transfer_to_billing_agent"),
            handoff(docs, tool_name_override="transfer_to_docs_agent"),
        ],
        model=ScriptedModel([[_tool_call("transfer_to_billing_agent", {}, "call-1")]]),
    )
    adapter = OpenAIAgentsAdapter()
    prepared = adapter.prepare(triage, RecordingGateway())
    run = asyncio.run(
        adapter.run(prepared, "invoice docs", run_id="run-wrong-agent", max_turns=4)
    )

    assert tripwire["docs"] == 0
    assert run.termination == RunTermination.COMPLETED
    attempts = [
        event.payload
        for event in run.events
        if event.event_type == CanonicalEventType.TOOL_ATTEMPT
    ]
    assert attempts and attempts[0]["tool_name"] == "search_docs"
    assert attempts[0]["agent_name"] == "Billing Agent"
    results = [
        event.payload
        for event in run.events
        if event.event_type == CanonicalEventType.TOOL_RESULT
    ]
    assert results and results[0]["status"] == "blocked"
    assert results[0]["error"]["code"] == "unknown_tool"


def test_topology_is_described_for_handoff_targets_only() -> None:
    triage, _, _ = _billing_pair()
    adapter = OpenAIAgentsAdapter()

    single = Agent(name="Solo", instructions="s")
    assert adapter.describe_topology(single) is None

    topology = adapter.describe_topology(triage)
    assert topology is not None
    decoded = decode_topology(topology)
    names = [agent["name"] for agent in decoded["agents"]]
    assert names == ["Triage Agent", "Billing Agent"]
    triage_edges = decoded["agents"][0]["handoffs"]
    assert triage_edges[0]["tool_name"] == "transfer_to_billing_agent"
    assert triage_edges[0]["target_agent"] == "Billing Agent"
    assert triage_edges[0]["issue_codes"] == []
    assert "context_assignments" not in triage_edges[0]
    with pytest.raises(ValueError):
        decode_topology({"framework": "openai_agents", "agents": [{"name": "x"}]})


def test_single_agent_spec_id_is_byte_stable_after_handoff_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned against the fingerprinting algorithm, not the installed SDK.

    ``spec_id`` deliberately bakes in ``framework_version`` (see
    SUPPORTED_SDK_MINOR_RANGE's own docstring), so this pin is only
    meaningful against one fixed version -- 0.20.0, whatever it was computed
    against -- regardless of which minor within the adapter's verified range
    (0.20-0.22) actually happens to be installed when the suite runs.
    """

    import agentcheck.adapters.openai_agents as openai_agents_adapter

    monkeypatch.setattr(openai_agents_adapter, "_sdk_version", lambda: "0.20.0")

    @function_tool
    def get_weather(city: str) -> str:
        """Return weather."""

        return "sunny"

    agent = Agent(
        name="Pin Agent", instructions="You report weather.", tools=[get_weather]
    )
    spec = OpenAIAgentsAdapter().inspect(agent, source="pin:agent.py:agent")
    assert spec.spec_id == PINNED_SINGLE_AGENT_SPEC_ID


def test_builtin_suite_scenario_fingerprints_are_unchanged() -> None:
    suite = build_account_support_suite(seed=1729)
    assert suite[0].scenario_id == "happy_lookup"
    assert suite[0].fingerprint == PINNED_HAPPY_LOOKUP_FINGERPRINT


def test_merged_spec_includes_subagent_tools_and_conditional_fingerprint() -> None:
    triage, _, _ = _billing_pair()
    adapter = OpenAIAgentsAdapter()
    spec = adapter.inspect(triage, source="x/agent.py:triage_agent")
    assert [item.value.name for item in spec.tools.items] == ["lookup_invoice"]
    assert "handoff" in spec.observability.supported_event_types.value

    # The same agents without handoffs produce a different spec identity, and a
    # graph change must change the handoff spec identity.
    solo = Agent(name="Triage Agent", instructions="Route.")
    solo_spec = adapter.inspect(solo, source="x/agent.py:triage_agent")
    assert solo_spec.spec_id != spec.spec_id


# --- deterministic handoff evaluation ---------------------------------------


def _oracle(*, hard: bool = True) -> OracleProvenance:
    return OracleProvenance(
        oracle_id="routing_policy",
        strength=OracleStrength.VERSIONED_POLICY,
        source="tests/agentcheck/test_handoffs.py",
        confidence=1.0,
        evidence_ids=("routing-policy-v1",),
        supports_hard_failure=hard,
    )


def _scenario(
    constraints: list[TrajectoryConstraint], *, hard: bool = True
) -> Scenario:
    return Scenario(
        scenario_id="handoff-eval",
        title="Handoff trajectory evaluation",
        conversation_turns=(
            ConversationTurn(
                turn_id="turn-1",
                role=ConversationRole.USER,
                content="Route my request.",
            ),
        ),
        trajectory_constraints=tuple(constraints),
        dimension_tags=("workflow:handoff",),
        oracle_provenance=(_oracle(hard=hard),),
        generation_seed=0,
    )


def _constraint(kind: TrajectoryConstraintKind, **parameters: Any) -> TrajectoryConstraint:
    return TrajectoryConstraint(
        criterion_id=f"handoff:{kind.value}",
        kind=kind,
        description=f"handoff constraint {kind.value}",
        parameters=parameters,
        oracle_ids=("routing_policy",),
    )


def _handoff_run(
    edges: list[tuple[str, str, bool]],
    *,
    attempts: tuple[tuple[str, int], ...] = (),
) -> CanonicalRun:
    """Build a canonical run whose events are USER_TURN + HANDOFF (+ attempts).

    ``attempts`` entries are ``(tool_name, position)`` where position indexes
    the event sequence slot after which the attempt occurs.
    """

    run_id = "run-eval"
    now = utc_now()
    events: list[CanonicalEvent] = [
        CanonicalEvent(
            event_id=f"{run_id}:event:0000",
            run_id=run_id,
            sequence=0,
            event_type=CanonicalEventType.USER_TURN,
            timestamp=now,
            payload={"text": "Route my request."},
        )
    ]
    tool_attempts: list[ToolAttempt] = []
    events_spec: list[tuple[str, Any]] = [("handoff", edge) for edge in edges]
    for attempt in attempts:
        events_spec.insert(attempt[1], ("attempt", attempt))
    for kind, value in events_spec:
        sequence = len(events)
        event_id = f"{run_id}:event:{sequence:04d}"
        if kind == "handoff":
            from_agent, to_agent, ignored = value
            events.append(
                CanonicalEvent(
                    event_id=event_id,
                    run_id=run_id,
                    sequence=sequence,
                    event_type=CanonicalEventType.HANDOFF,
                    timestamp=now,
                    payload={
                        "handoff_tool_name": f"transfer_to_{to_agent}",
                        "from_agent": from_agent,
                        "to_agent": to_agent,
                        "arguments": {},
                        "call_id": f"call-{sequence}",
                        "ignored": ignored,
                    },
                )
            )
        else:
            tool_name = value[0]
            events.append(
                CanonicalEvent(
                    event_id=event_id,
                    run_id=run_id,
                    sequence=sequence,
                    event_type=CanonicalEventType.TOOL_ATTEMPT,
                    timestamp=now,
                    payload={"tool_name": tool_name, "arguments": {}},
                )
            )
            tool_attempts.append(
                ToolAttempt(
                    attempt_id=f"{run_id}:attempt:{len(tool_attempts):04d}",
                    event_id=event_id,
                    tool_name=tool_name,
                    arguments={},
                    sequence=len(tool_attempts),
                    timestamp=now,
                )
            )
    return CanonicalRun(
        run_id=run_id,
        scenario_id="handoff-eval",
        target_id="agentspec-test",
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
        events=tuple(events),
        tool_attempts=tuple(tool_attempts),
    )


def _assertion(evaluation: Any, criterion_id: str) -> Any:
    return next(
        assertion
        for assertion in evaluation.assertions
        if assertion.assertion_id == criterion_id
    )


def test_required_handoff_passes_and_fails_deterministically() -> None:
    constraint = _constraint(
        TrajectoryConstraintKind.REQUIRED_HANDOFF,
        from_agent="Triage",
        to_agent="Billing",
    )
    good = evaluate_run(
        _scenario([constraint]), _handoff_run([("Triage", "Billing", False)])
    )
    assert good.verdict == Verdict.PASS

    bad = evaluate_run(
        _scenario([constraint]), _handoff_run([("Triage", "Docs", False)])
    )
    assert bad.verdict == Verdict.FAIL
    assert _assertion(bad, constraint.criterion_id).result == Verdict.FAIL


def test_ignored_handoffs_do_not_satisfy_required_handoff() -> None:
    constraint = _constraint(
        TrajectoryConstraintKind.REQUIRED_HANDOFF, to_agent="Billing"
    )
    evaluation = evaluate_run(
        _scenario([constraint]), _handoff_run([("Triage", "Billing", True)])
    )
    assert _assertion(evaluation, constraint.criterion_id).result == Verdict.FAIL


def test_forbidden_handoff_and_max_handoffs() -> None:
    forbidden = _constraint(
        TrajectoryConstraintKind.FORBIDDEN_HANDOFF, to_agent="Docs"
    )
    maximum = _constraint(TrajectoryConstraintKind.MAX_HANDOFFS, maximum=1)
    run = _handoff_run(
        [("Triage", "Docs", False), ("Docs", "Triage", False)]
    )
    evaluation = evaluate_run(_scenario([forbidden, maximum]), run)
    assert _assertion(evaluation, forbidden.criterion_id).result == Verdict.FAIL
    assert _assertion(evaluation, maximum.criterion_id).result == Verdict.FAIL


def test_handoff_loop_detection_flags_repeated_edges_only() -> None:
    constraint = _constraint(TrajectoryConstraintKind.NO_HANDOFF_LOOP)
    healthy = _handoff_run(
        [("Triage", "Billing", False), ("Billing", "Triage", False)]
    )
    assert (
        _assertion(
            evaluate_run(_scenario([constraint]), healthy), constraint.criterion_id
        ).result
        == Verdict.PASS
    )
    looping = _handoff_run(
        [
            ("Triage", "Billing", False),
            ("Billing", "Triage", False),
            ("Triage", "Billing", False),
        ]
    )
    assert (
        _assertion(
            evaluate_run(_scenario([constraint]), looping), constraint.criterion_id
        ).result
        == Verdict.FAIL
    )


def test_handoff_before_tool_orders_side_effects_after_routing() -> None:
    constraint = _constraint(
        TrajectoryConstraintKind.HANDOFF_BEFORE_TOOL,
        tool_name="update_seat",
        to_agent="Billing",
    )
    late = _handoff_run(
        [("Triage", "Billing", False)], attempts=[("update_seat", 2)]
    )
    assert (
        _assertion(
            evaluate_run(_scenario([constraint]), late), constraint.criterion_id
        ).result
        == Verdict.PASS
    )
    early = _handoff_run(
        [("Triage", "Billing", False)], attempts=[("update_seat", 0)]
    )
    assert (
        _assertion(
            evaluate_run(_scenario([constraint]), early), constraint.criterion_id
        ).result
        == Verdict.FAIL
    )


def test_handoff_fail_without_hard_oracle_downgrades_to_inconclusive() -> None:
    constraint = _constraint(
        TrajectoryConstraintKind.REQUIRED_HANDOFF, to_agent="Billing"
    )
    evaluation = evaluate_run(
        _scenario([constraint], hard=False), _handoff_run([])
    )
    assert _assertion(evaluation, constraint.criterion_id).result == Verdict.INCONCLUSIVE
    assert evaluation.verdict == Verdict.INCONCLUSIVE


def test_lint_accepts_valid_handoff_kinds_and_rejects_bad_parameters() -> None:
    triage, _, _ = _billing_pair()
    spec = OpenAIAgentsAdapter().inspect(triage, source="x/agent.py:triage_agent")

    valid = _scenario(
        [
            _constraint(
                TrajectoryConstraintKind.REQUIRED_HANDOFF, to_agent="Billing Agent"
            ),
            _constraint(TrajectoryConstraintKind.NO_HANDOFF_LOOP),
            _constraint(
                TrajectoryConstraintKind.HANDOFF_BEFORE_TOOL,
                tool_name="lookup_invoice",
            ),
        ]
    )
    assert lint_scenario(valid, spec) == ()

    invalid = _scenario(
        [
            _constraint(TrajectoryConstraintKind.REQUIRED_HANDOFF),
            _constraint(TrajectoryConstraintKind.FORBIDDEN_HANDOFF),
            _constraint(TrajectoryConstraintKind.MAX_HANDOFFS),
            _constraint(TrajectoryConstraintKind.NO_HANDOFF_LOOP, max_edge_repeats=0),
            _constraint(TrajectoryConstraintKind.HANDOFF_BEFORE_TOOL),
        ]
    )
    codes = {issue.code for issue in lint_scenario(invalid, spec)}
    assert codes == {"invalid_trajectory_parameters"}
