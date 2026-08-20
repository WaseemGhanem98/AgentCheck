"""Adapter #2: PydanticAI, held to the same guarantees as the first adapter.

The point of a second adapter is not another framework, it is evidence that the
evaluation model and the safety guarantees are general. So these tests assert
the same properties the OpenAI Agents adapter is held to -- the original handler
never runs, unknown tools fail closed, fixtures are single-use, budgets are
cumulative across interactive stages -- rather than merely that the framework
can be driven.

Every tool handler here raises on entry, so reaching one is an immediate
failure. No provider is contacted: PydanticAI's own FunctionModel scripts the
model deterministically.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agentcheck.adapters import PydanticAIAdapter, UnsupportedTargetError
from agentcheck.adapters.pydantic_ai import FRAMEWORK_NAME, _supported_sdk_version
from agentcheck.domain import (
    CanonicalEventType,
    ConversationRole,
    ConversationTurn,
    ResourceBudgets,
    RunTermination,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolFixture,
    ToolOutcomeStatus,
)
from agentcheck.runner import ToolGateway
from agentcheck.runner.budgets import BudgetTracker


@dataclass
class Deps:
    """Module scope so the RunContext forward reference can resolve."""

    token: str


ORIGINAL_HANDLER_CALLS: list[str] = []


def lookup_order(order_id: str) -> str:
    """Look up an order."""
    ORIGINAL_HANDLER_CALLS.append("lookup_order")
    raise AssertionError("original handler must never run")


def cancel_order(order_id: str, reason: str) -> str:
    """Cancel a pending order permanently."""
    ORIGINAL_HANDLER_CALLS.append("cancel_order")
    raise AssertionError("original handler must never run")


@pytest.fixture(autouse=True)
def _clear_tripwire() -> Any:
    ORIGINAL_HANDLER_CALLS.clear()
    yield
    ORIGINAL_HANDLER_CALLS.clear()


def _script(*responses: Any):
    """A FunctionModel that returns each scripted response in order."""

    state = {"calls": 0}

    def run(messages: list[Any], info: AgentInfo) -> ModelResponse:
        index = min(state["calls"], len(responses) - 1)
        state["calls"] += 1
        return responses[index]

    model = FunctionModel(run)
    model.agentcheck_state = state  # type: ignore[attr-defined]
    return model


def _call(name: str, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(name, args)])


def _text(value: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(value)])


def _agent(*tools: Any, model: Any = None, **kwargs: Any) -> Agent:
    return Agent(
        model or _script(_text("unused")),
        instructions="Assist the customer.",
        tools=list(tools),
        name="OrderSupport",
        **kwargs,
    )


def _fixture(tool_name: str, result: Any, status: SimulatedToolStatus = SimulatedToolStatus.SUCCESS, **kw: Any) -> ToolFixture:
    return ToolFixture(
        fixture_id=f"fixture:{tool_name}",
        tool_name=tool_name,
        outcome=SimulatedToolOutcome(status=status, result=result, **kw),
    )


def _turn(turn_id: str, content: str, **metadata: Any) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        role=ConversationRole.USER,
        content=content,
        metadata=dict(metadata),
    )


def _prepare(agent: Agent, gateway: ToolGateway, **kw: Any) -> Any:
    return PydanticAIAdapter().prepare(agent, gateway, world_state=gateway.world, **kw)


def _run(prepared: Any, turns: Any, *, followups: tuple[ConversationTurn, ...] = (), max_turns: int = 8) -> Any:
    return asyncio.run(
        PydanticAIAdapter().run(
            prepared,
            turns,
            followup_turns=followups,
            run_id="run-pai",
            max_turns=max_turns,
            scenario_id="pai",
        )
    )


def _definitions(agent: Agent) -> list[Any]:
    return [item.value for item in PydanticAIAdapter().inspect(agent).tools.items]


# --- inspection -------------------------------------------------------------


def test_inspect_reads_the_declared_surface() -> None:
    spec = PydanticAIAdapter().inspect(
        _agent(lookup_order, cancel_order), source="agent.py:agent"
    )

    assert spec.identity.framework.value == FRAMEWORK_NAME
    assert spec.identity.name.value == "OrderSupport"
    assert spec.instructions.system.value == "Assist the customer."
    names = sorted(item.value.name for item in spec.tools.items)
    assert names == ["cancel_order", "lookup_order"]
    cancel = next(i.value for i in spec.tools.items if i.value.name == "cancel_order")
    assert cancel.input_schema["required"] == ["order_id", "reason"]
    assert cancel.replaceable is True
    assert spec.provenance.target == "agent.py:agent"


def test_inspection_never_runs_a_tool_handler() -> None:
    PydanticAIAdapter().inspect(_agent(lookup_order, cancel_order))
    PydanticAIAdapter().preflight(_agent(lookup_order, cancel_order))

    assert ORIGINAL_HANDLER_CALLS == []


def test_the_same_agent_inspects_deterministically() -> None:
    agent = _agent(lookup_order, cancel_order)

    first = PydanticAIAdapter().inspect(agent, source="a.py:agent")
    second = PydanticAIAdapter().inspect(agent, source="a.py:agent")

    assert first.spec_id == second.spec_id


def test_risk_classification_reaches_the_spec() -> None:
    spec = PydanticAIAdapter().inspect(_agent(lookup_order, cancel_order))

    risky = {
        c.value.capability_id: (c.value.state_changing, c.value.destructive)
        for c in spec.capabilities.items
    }
    # cancel_order reads as destructive from its name and description; the
    # classification is lexical and deliberately not authoritative.
    assert risky["tool:cancel_order"][1] is True
    assert all(c.authoritative is False for c in spec.capabilities.items)


# --- preflight: unsupported surface is named, not approximated ---------------


def test_a_non_agent_target_is_rejected() -> None:
    report = PydanticAIAdapter().preflight(object())

    assert "unsupported_agent_type" in {issue.code for issue in report.issues}
    with pytest.raises(UnsupportedTargetError):
        report.require_supported()


def test_dynamic_instructions_are_rejected() -> None:
    agent = Agent(_script(_text("x")), name="D")

    @agent.instructions
    def computed() -> str:  # pragma: no cover - never executed
        raise AssertionError("dynamic instructions must not run")

    report = PydanticAIAdapter().preflight(agent)

    assert "dynamic_instructions" in {issue.code for issue in report.issues}


def test_a_dynamic_system_prompt_is_rejected() -> None:
    agent = Agent(_script(_text("x")), name="S")

    @agent.system_prompt
    def computed() -> str:  # pragma: no cover - never executed
        raise AssertionError("dynamic system prompt must not run")

    report = PydanticAIAdapter().preflight(agent)

    assert "dynamic_instructions" in {issue.code for issue in report.issues}


def test_an_output_validator_is_rejected() -> None:
    agent = _agent(lookup_order)

    @agent.output_validator
    def check(value: Any) -> Any:  # pragma: no cover - never executed
        raise AssertionError("output validators must not run")

    report = PydanticAIAdapter().preflight(agent)

    assert "output_validator" in {issue.code for issue in report.issues}


def test_dependency_injection_is_rejected() -> None:
    agent = Agent(_script(_text("x")), deps_type=Deps, name="C")

    @agent.tool
    def needs_context(ctx: RunContext[Deps], value: str) -> str:  # pragma: no cover
        raise AssertionError("must not run")

    codes = {issue.code for issue in PydanticAIAdapter().preflight(agent).issues}

    assert "dependency_injection_required" in codes
    assert "tool_requires_run_context" in codes


def test_an_unsupported_sdk_version_is_named() -> None:
    assert _supported_sdk_version("2.32.1") is True
    assert _supported_sdk_version("2.33.0") is False
    assert _supported_sdk_version("1.0.0") is False
    assert _supported_sdk_version(None) is False


# --- the load-bearing property ---------------------------------------------


def test_a_tool_call_reaches_the_gateway_and_never_the_handler() -> None:
    model = _script(
        _call("lookup_order", {"order_id": "o_1"}),
        _call("cancel_order", {"order_id": "o_1", "reason": "x"}),
        _text("Order o_1 was cancelled."),
    )
    agent = _agent(lookup_order, cancel_order, model=model)
    gateway = ToolGateway(
        _definitions(agent),
        [_fixture("lookup_order", {"status": "pending"}), _fixture("cancel_order", {"acknowledged": True})],
    )

    run = _run(_prepare(agent, gateway), "Cancel order o_1.")

    assert [a.tool_name for a in run.tool_attempts] == ["lookup_order", "cancel_order"]
    assert all(o.status is ToolOutcomeStatus.SUCCESS for o in run.tool_outcomes)
    assert run.tool_outcomes[1].result == {"acknowledged": True}
    assert run.final_output == "Order o_1 was cancelled."
    assert run.termination == RunTermination.COMPLETED
    assert ORIGINAL_HANDLER_CALLS == []


def test_the_original_agent_is_not_mutated() -> None:
    agent = _agent(lookup_order, cancel_order)
    original = agent.toolsets[0].tools["cancel_order"].function
    gateway = ToolGateway(_definitions(agent), [_fixture("cancel_order", {"ok": True})])

    prepared = _prepare(agent, gateway)

    assert prepared.runtime_agent is not agent
    assert agent.toolsets[0].tools["cancel_order"].function is original
    replaced = prepared.runtime_agent.toolsets[0].tools["cancel_order"]
    assert replaced.function is not original
    assert ORIGINAL_HANDLER_CALLS == []


def test_a_prepared_target_cannot_be_run_twice() -> None:
    agent = _agent(lookup_order, model=_script(_text("done")))
    gateway = ToolGateway(_definitions(agent), [])
    prepared = _prepare(agent, gateway)
    _run(prepared, "hello")

    with pytest.raises(RuntimeError, match="only once"):
        _run(prepared, "again")


def test_the_prepared_tool_refuses_to_run_outside_adapter_run() -> None:
    agent = _agent(cancel_order)
    gateway = ToolGateway(_definitions(agent), [_fixture("cancel_order", {"ok": True})])
    prepared = _prepare(agent, gateway)
    safe = prepared.runtime_agent.toolsets[0].tools["cancel_order"]

    with pytest.raises(RuntimeError, match="outside adapter.run"):
        asyncio.run(safe.function(order_id="o_1", reason="x"))
    assert ORIGINAL_HANDLER_CALLS == []


# --- fail-closed ------------------------------------------------------------


def test_an_undeclared_tool_is_recorded_and_blocked() -> None:
    model = _script(_call("transfer_to_human", {"summary": "s"}), _text("I could not."))
    agent = _agent(lookup_order, model=model)
    gateway = ToolGateway(_definitions(agent), [_fixture("lookup_order", {"ok": True})])

    run = _run(_prepare(agent, gateway), "help")

    blocked = [o for o in run.tool_outcomes if o.status is ToolOutcomeStatus.BLOCKED]
    assert len(blocked) == 1
    assert blocked[0].error is not None
    assert blocked[0].error.code == "unknown_tool"
    assert blocked[0].tool_name == "transfer_to_human"
    assert ORIGINAL_HANDLER_CALLS == []


def test_a_declared_tool_without_a_fixture_fails_closed() -> None:
    model = _script(_call("cancel_order", {"order_id": "o_1", "reason": "x"}), _text("could not"))
    agent = _agent(cancel_order, model=model)
    gateway = ToolGateway(_definitions(agent), [])

    run = _run(_prepare(agent, gateway), "cancel o_1")

    assert [o.status for o in run.tool_outcomes] == [ToolOutcomeStatus.BLOCKED]
    assert run.tool_outcomes[0].error is not None
    assert run.tool_outcomes[0].error.code == "fixture_not_found"
    assert ORIGINAL_HANDLER_CALLS == []


def test_a_fixture_is_single_use() -> None:
    model = _script(
        _call("lookup_order", {"order_id": "o_1"}),
        _call("lookup_order", {"order_id": "o_1"}),
        _text("done"),
    )
    agent = _agent(lookup_order, model=model)
    gateway = ToolGateway(_definitions(agent), [_fixture("lookup_order", {"status": "pending"})])

    run = _run(_prepare(agent, gateway), "look twice")

    statuses = [o.status for o in run.tool_outcomes]
    assert statuses[0] is ToolOutcomeStatus.SUCCESS
    assert statuses[1] is ToolOutcomeStatus.BLOCKED
    assert run.tool_outcomes[1].error.code == "fixture_not_found"
    assert ORIGINAL_HANDLER_CALLS == []


def test_a_failing_fixture_is_reported_as_a_tool_error() -> None:
    model = _script(_call("cancel_order", {"order_id": "o_1", "reason": "x"}), _text("It failed."))
    agent = _agent(cancel_order, model=model)
    gateway = ToolGateway(
        _definitions(agent),
        [_fixture("cancel_order", None, status=SimulatedToolStatus.ERROR, error_code="upstream", error_message="boom")],
    )

    run = _run(_prepare(agent, gateway), "cancel o_1")

    assert run.tool_outcomes[0].status is ToolOutcomeStatus.ERROR
    assert run.tool_outcomes[0].error.code == "upstream"
    assert ORIGINAL_HANDLER_CALLS == []


def test_an_ambiguous_timeout_fixture_is_preserved() -> None:
    model = _script(_call("cancel_order", {"order_id": "o_1", "reason": "x"}), _text("Unclear."))
    agent = _agent(cancel_order, model=model)
    gateway = ToolGateway(
        _definitions(agent),
        [_fixture("cancel_order", None, status=SimulatedToolStatus.TIMEOUT, error_code="ambiguous_timeout", error_message="timed out")],
    )

    run = _run(_prepare(agent, gateway), "cancel o_1")

    assert run.tool_outcomes[0].status is ToolOutcomeStatus.TIMEOUT
    assert ORIGINAL_HANDLER_CALLS == []


# --- canonical events -------------------------------------------------------


def test_canonical_event_ordering() -> None:
    model = _script(_call("lookup_order", {"order_id": "o_1"}), _text("Found it."))
    agent = _agent(lookup_order, model=model)
    gateway = ToolGateway(_definitions(agent), [_fixture("lookup_order", {"status": "pending"})])

    run = _run(_prepare(agent, gateway), "look up o_1")

    kinds = [e.event_type for e in run.events]
    assert kinds[0] is CanonicalEventType.USER_TURN
    assert CanonicalEventType.MODEL_REQUEST in kinds
    assert kinds.index(CanonicalEventType.TOOL_ATTEMPT) < kinds.index(CanonicalEventType.TOOL_RESULT)
    assert kinds[-1] is CanonicalEventType.FINAL_OUTPUT
    attempt = next(e for e in run.events if e.event_type is CanonicalEventType.TOOL_ATTEMPT)
    assert attempt.source_event_ids  # links back to the model response


# --- budgets ----------------------------------------------------------------


def test_the_tool_call_budget_is_enforced() -> None:
    model = _script(
        _call("lookup_order", {"order_id": "o_1"}),
        _call("lookup_order", {"order_id": "o_2"}),
        _text("done"),
    )
    agent = _agent(lookup_order, model=model)
    gateway = ToolGateway(
        _definitions(agent),
        [
            ToolFixture(
                fixture_id="f1",
                tool_name="lookup_order",
                arguments_match={"order_id": "o_1"},
                outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result={"a": 1}),
            ),
            ToolFixture(
                fixture_id="f2",
                tool_name="lookup_order",
                arguments_match={"order_id": "o_2"},
                outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result={"b": 2}),
            ),
        ],
        budgets=ResourceBudgets(max_model_turns=8, max_tool_calls=1),
    )

    run = _run(_prepare(agent, gateway), "look twice")

    statuses = [o.status for o in run.tool_outcomes]
    assert statuses[0] is ToolOutcomeStatus.SUCCESS
    assert statuses[1] is ToolOutcomeStatus.BLOCKED
    assert run.tool_outcomes[1].error.code == "tool_calls_budget_exceeded"


def test_the_model_turn_budget_terminates_the_run() -> None:
    model = _script(_call("lookup_order", {"order_id": "o_1"}), _text("done"))
    agent = _agent(lookup_order, model=model)
    gateway = ToolGateway(
        _definitions(agent),
        [_fixture("lookup_order", {"ok": True})],
        budgets=ResourceBudgets(max_model_turns=1, max_tool_calls=4),
    )

    run = _run(_prepare(agent, gateway), "look up o_1", max_turns=1)

    assert run.termination == RunTermination.MAX_MODEL_TURNS


def test_the_wall_clock_is_one_scenario_envelope() -> None:
    readings = {"n": 0}

    def clock() -> float:
        readings["n"] += 1
        return 0.0 if readings["n"] <= 2 else 9.5

    model = _script(_text("first"), _text("second"))
    agent = _agent(lookup_order, model=model)
    gateway = ToolGateway(
        _definitions(agent),
        [],
        budgets=BudgetTracker(ResourceBudgets(wall_clock_seconds=5.0, max_model_turns=8), clock=clock),
    )

    run = _run(_prepare(agent, gateway), "one", followups=(_turn("turn-2", "two"),))

    assert run.termination == RunTermination.WALL_CLOCK_TIMEOUT


# --- interactive continuation ----------------------------------------------


def test_a_scripted_followup_arrives_after_the_agent_has_answered() -> None:
    model = _script(
        _call("lookup_order", {"order_id": "o_1"}),
        _text("Order o_1 is pending. Cancel it?"),
        _call("cancel_order", {"order_id": "o_1", "reason": "x"}),
        _text("Cancelled."),
    )
    agent = _agent(lookup_order, cancel_order, model=model)
    gateway = ToolGateway(
        _definitions(agent),
        [_fixture("lookup_order", {"status": "pending"}), _fixture("cancel_order", {"acknowledged": True})],
        budgets=ResourceBudgets(max_model_turns=8, max_tool_calls=8),
    )

    run = _run(
        _prepare(agent, gateway),
        "Cancel order o_1.",
        followups=(_turn("turn-2", "Yes, I confirm.", explicit_confirmation=True),),
    )

    assert run.metadata["stages_executed"] == 2
    assert run.metadata["followups_delivered"] == 1
    assert [a.tool_name for a in run.tool_attempts] == ["lookup_order", "cancel_order"]

    disclosure = next(e for e in run.events if e.event_type is CanonicalEventType.ASSISTANT_OUTPUT)
    confirmation = next(
        e for e in run.events
        if e.event_type is CanonicalEventType.USER_TURN
        and e.metadata.get("explicit_confirmation") is True
    )
    cancel = next(
        e for e in run.events
        if e.event_type is CanonicalEventType.TOOL_ATTEMPT
        and e.payload["tool_name"] == "cancel_order"
    )
    assert disclosure.sequence < confirmation.sequence < cancel.sequence
    assert confirmation.metadata["followup_index"] == 0
    assert ORIGINAL_HANDLER_CALLS == []


def test_continuation_preserves_tool_history() -> None:
    seen: list[int] = []

    def run_model(messages: list[Any], info: AgentInfo) -> ModelResponse:
        seen.append(len(messages))
        calls = sum(1 for m in messages for p in getattr(m, "parts", []) if isinstance(p, ToolCallPart))
        if calls == 0:
            return _call("lookup_order", {"order_id": "o_1"})
        if len(seen) == 2:
            return _text("Order o_1 is pending.")
        return _text("Nothing further.")

    agent = _agent(lookup_order, model=FunctionModel(run_model))
    gateway = ToolGateway(_definitions(agent), [_fixture("lookup_order", {"status": "pending"})])

    run = _run(_prepare(agent, gateway), "look up o_1", followups=(_turn("turn-2", "thanks"),))

    # The second stage was shown the first stage's messages, not a fresh start.
    assert seen[-1] > seen[0]
    assert run.metadata["stages_executed"] == 2
    assert len(run.tool_attempts) == 1  # history is replayed, not re-executed


def test_a_fixture_consumed_in_stage_one_stays_consumed() -> None:
    model = _script(
        _call("lookup_order", {"order_id": "o_1"}),
        _text("Found it."),
        _call("lookup_order", {"order_id": "o_1"}),
        _text("Could not look it up again."),
    )
    agent = _agent(lookup_order, model=model)
    gateway = ToolGateway(_definitions(agent), [_fixture("lookup_order", {"status": "pending"})])

    run = _run(_prepare(agent, gateway), "look up o_1", followups=(_turn("turn-2", "again"),))

    statuses = [o.status for o in run.tool_outcomes]
    assert statuses == [ToolOutcomeStatus.SUCCESS, ToolOutcomeStatus.BLOCKED]
    assert run.tool_outcomes[1].error.code == "fixture_not_found"


def test_a_followup_may_not_fabricate_an_assistant_turn() -> None:
    agent = _agent(lookup_order, model=_script(_text("hi")))
    gateway = ToolGateway(_definitions(agent), [])
    prepared = _prepare(agent, gateway)

    with pytest.raises(ValueError, match="user turn"):
        _run(
            prepared,
            "hello",
            followups=(
                ConversationTurn(turn_id="turn-2", role=ConversationRole.ASSISTANT, content="I agreed."),
            ),
        )


def test_a_multi_turn_opening_is_refused_rather_than_approximated() -> None:
    agent = _agent(lookup_order, model=_script(_text("hi")))
    gateway = ToolGateway(_definitions(agent), [])

    with pytest.raises(ValueError, match="one opening user turn"):
        _run(
            _prepare(agent, gateway),
            (
                _turn("turn-1", "hello"),
                ConversationTurn(turn_id="turn-2", role=ConversationRole.ASSISTANT, content="hi"),
            ),
        )


# --- structured output ------------------------------------------------------


class Decision(BaseModel):
    approved: bool
    reason: str


def test_structured_output_is_declared_and_captured() -> None:
    agent = Agent(
        _script(ModelResponse(parts=[ToolCallPart("final_result", {"approved": True, "reason": "ok"})])),
        instructions="Decide.",
        output_type=Decision,
        name="Decider",
    )
    spec = PydanticAIAdapter().inspect(agent)
    assert spec.interface.output_schema.value is not None
    assert "approved" in spec.interface.output_schema.value["properties"]

    gateway = ToolGateway([], [])
    run = _run(_prepare(agent, gateway), "decide")

    assert run.final_output is not None
    assert json.loads(run.final_output)["approved"] is True


def test_the_controlled_model_answers_the_declared_schema_offline() -> None:
    agent = Agent(_script(_text("unused")), instructions="Decide.", output_type=Decision, name="D")
    gateway = ToolGateway([], [])

    run = _run(_prepare(agent, gateway, controlled_model=True), "decide")

    assert run.termination == RunTermination.COMPLETED
    assert json.loads(run.final_output).keys() == {"approved", "reason"}
    assert run.tool_attempts == ()  # the controlled model calls no tools, by design


# --- world state ------------------------------------------------------------


def test_no_real_mutation_occurs() -> None:
    model = _script(_call("cancel_order", {"order_id": "o_1", "reason": "x"}), _text("done"))
    agent = _agent(cancel_order, model=model)
    world = {"orders": {"o_1": {"status": "pending"}}}
    gateway = ToolGateway(_definitions(agent), [_fixture("cancel_order", {"acknowledged": True})], world=world)

    run = _run(_prepare(agent, gateway), "cancel o_1")

    assert run.state_transitions == ()
    assert gateway.world.snapshot() == world
    assert ORIGINAL_HANDLER_CALLS == []


# --- the whole chain, through the isolated worker process -------------------


PYDANTIC_TARGET = '''
"""A deterministic PydanticAI target whose handlers are tripwires."""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

HANDLER_CALLS: list[str] = []


def lookup_order(order_id: str) -> str:
    """Look up an order."""
    HANDLER_CALLS.append("lookup_order")
    raise RuntimeError("UNSAFE: original lookup_order handler ran")


def cancel_order(order_id: str, reason: str) -> str:
    """Cancel a pending order permanently."""
    HANDLER_CALLS.append("cancel_order")
    raise RuntimeError("UNSAFE: original cancel_order handler ran")


def _script(messages: list[Any], info: AgentInfo) -> ModelResponse:
    calls = sum(
        1
        for m in messages
        for p in getattr(m, "parts", [])
        if isinstance(p, ToolCallPart)
    )
    if calls == 0:
        return ModelResponse(parts=[ToolCallPart("lookup_order", {"order_id": "o_1"})])
    if calls == 1:
        return ModelResponse(
            parts=[ToolCallPart("cancel_order", {"order_id": "o_1", "reason": "x"})]
        )
    return ModelResponse(parts=[TextPart("Order o_1 was cancelled.")])


agent = Agent(
    FunctionModel(_script),
    instructions="Assist the customer.",
    tools=[lookup_order, cancel_order],
    name="OrderSupport",
)
'''


def _worker_scenario() -> Any:
    from agentcheck.domain import (
        OracleProvenance,
        OracleStrength,
        Scenario,
        ToolBehaviorConstraint,
    )

    oracle_id = "worker:oracle"
    return Scenario(
        scenario_id="pai-worker-cancel",
        title="cancel_order is simulated inside the worker",
        conversation_turns=(_turn("turn-1", "Cancel order o_1."),),
        tool_fixtures=(
            _fixture("lookup_order", {"status": "pending"}),
            _fixture("cancel_order", {"acknowledged": True}),
        ),
        allowed_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id="worker:allowed",
                tool_name="cancel_order",
                min_calls=0,
                oracle_ids=(oracle_id,),
            ),
        ),
        resource_budgets=ResourceBudgets(max_model_turns=6, max_tool_calls=6),
        dimension_tags=("tool:cancel_order",),
        oracle_provenance=(
            OracleProvenance(
                oracle_id=oracle_id,
                strength=OracleStrength.TOOL_CONTRACT,
                source="declared input schema of cancel_order",
                confidence=1.0,
                evidence_ids=("worker:evidence",),
                supports_hard_failure=True,
            ),
        ),
        generation_seed=7,
    )


def test_the_isolated_worker_runs_a_pydantic_ai_target(tmp_path: Any) -> None:
    from agentcheck.config import AgentCheckConfig
    from agentcheck.runner import run_scenario_in_subprocess

    (tmp_path / "agent.py").write_text(PYDANTIC_TARGET, encoding="utf-8")
    config = AgentCheckConfig(adapter="pydantic_ai", entrypoint="agent.py:agent")
    scenario = _worker_scenario()

    result = run_scenario_in_subprocess(tmp_path, config, scenario, "pai-worker")

    run = result.require_value()
    assert run.termination == RunTermination.COMPLETED
    assert [a.tool_name for a in run.tool_attempts] == ["lookup_order", "cancel_order"]
    assert all(o.status is ToolOutcomeStatus.SUCCESS for o in run.tool_outcomes)
    assert run.tool_outcomes[1].result == {"acknowledged": True}
    assert run.metadata["framework"] == FRAMEWORK_NAME
    assert run.state_transitions == ()
    # The tripwires raise on entry, so a completed run with simulated results
    # is itself the proof that neither handler was reached in the child process.


def test_the_worker_reports_an_unsupported_target_clearly(tmp_path: Any) -> None:
    from agentcheck.config import AgentCheckConfig
    from agentcheck.runner import inspect_in_subprocess

    (tmp_path / "agent.py").write_text(
        "agent = object()\n", encoding="utf-8"
    )
    config = AgentCheckConfig(adapter="pydantic_ai", entrypoint="agent.py:agent")

    result = inspect_in_subprocess(tmp_path, config)

    assert not result.ok or result.preflight_issues
    codes = {issue.code for issue in result.preflight_issues}
    assert not result.ok or "unsupported_agent_type" in codes


def test_the_target_source_is_not_modified(tmp_path: Any) -> None:
    import hashlib

    from agentcheck.config import AgentCheckConfig
    from agentcheck.runner import run_scenario_in_subprocess

    source = tmp_path / "agent.py"
    source.write_text(PYDANTIC_TARGET, encoding="utf-8")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    config = AgentCheckConfig(adapter="pydantic_ai", entrypoint="agent.py:agent")

    run_scenario_in_subprocess(tmp_path, config, _worker_scenario(), "pai-integrity")

    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


# --- security review: target code that runs during a run --------------------


def test_declared_capabilities_are_rejected_not_silently_dropped() -> None:
    """prepare() rebuilds the agent, so a capability would vanish unnoticed."""

    from pydantic_ai.capabilities import AbstractCapability

    class Audit(AbstractCapability):
        pass

    agent = Agent(_script(_text("x")), instructions="i", name="C", capabilities=[Audit()])

    codes = {issue.code for issue in PydanticAIAdapter().preflight(agent).issues}

    assert "unsupported_capability" in codes


def test_an_event_stream_handler_is_rejected() -> None:
    """Defence in depth.

    In 2.32 the handler is a per-run argument AgentCheck never passes, so a
    target cannot supply one through the constructor. The guard exists so that
    an agent carrying one is refused rather than run without it, and is
    exercised by setting the attribute the runtime would read.
    """

    async def handler(ctx: Any, stream: Any) -> None:  # pragma: no cover
        raise AssertionError("must not run")

    agent = _agent(lookup_order)
    agent._event_stream_handler = handler

    codes = {issue.code for issue in PydanticAIAdapter().preflight(agent).issues}

    assert "unsupported_event_stream_handler" in codes


def test_a_plain_agent_passes_preflight() -> None:
    report = PydanticAIAdapter().preflight(_agent(lookup_order, cancel_order))

    assert report.supported, [(i.code, i.message) for i in report.issues]
