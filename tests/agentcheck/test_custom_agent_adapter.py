"""Executing a custom agent: what actually runs, and what provably does not.

The other two adapters are safe because AgentCheck rebuilds the target and
leaves the original tool callables behind. This one is safe because AgentCheck
is never given them: a custom agent declares ``ToolDefinition`` values and calls
tools through the ``ToolRuntime`` the adapter supplies. Most of this file exists
to hold that claim to its exact size --

* every declared tool call goes through ``ToolGateway``, so it is
  schema-validated, fixture-matched, budgeted, and refused when undeclared;
* the real handlers are not executed, and the mutations they would have made do
  not happen;
* and a side effect written directly into orchestration *does* run, because that
  code is the agent. The last one is tested too, so the guarantee is documented
  by a passing test rather than by an adjective.

Nothing here contacts a provider.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import threading
from pathlib import Path
from typing import Any, Sequence

import pytest

from agentcheck import ToolRuntime, TurnResult
from agentcheck.adapters import CustomAgentAdapter, UnsupportedTargetError
from agentcheck.adapters.custom import FRAMEWORK_NAME
from agentcheck.config import AgentCheckConfig
from agentcheck.domain import (
    ConversationRole,
    ConversationTurn,
    OracleProvenance,
    OracleStrength,
    RunTermination,
    Scenario,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolDefinition,
    ToolFixture,
    ToolOutcomeStatus,
    WorldStateEffect,
)
from agentcheck.domain.run import CanonicalEventType
from agentcheck.replay.fileset import collect_source_file_set, describe_file_set_mismatch
from agentcheck.runner import (
    FixtureNotFoundError,
    ToolCallBlockedError,
    ToolGateway,
    UnknownToolError,
    run_scenario_in_subprocess,
)
from agentcheck.runner.budgets import BudgetTracker


# ---------------------------------------------------------------------------
# A representative target
#
# `REAL_HANDLERS` is the point of the fixture rather than scenery: these are the
# functions a developer would really have wired to these tools, and they mutate
# genuinely. The agent below reaches its tools only through the runtime, so they
# stay untouched -- and the tests assert that with a counter and a file on disk,
# not with a mock.
# ---------------------------------------------------------------------------

GET_ORDER = ToolDefinition(
    name="get_order",
    description="Read one order.",
    input_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
)
FIND_USER = ToolDefinition(
    name="find_user",
    description="Look up the user who owns an order.",
    input_schema={
        "type": "object",
        "properties": {"email": {"type": "string"}},
        "required": ["email"],
        "additionalProperties": False,
    },
)
CANCEL_ORDER = ToolDefinition(
    name="cancel_order",
    description="Cancel one order. Irreversible.",
    input_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
    state_changing=True,
    destructive=True,
)


class RealWorld:
    """The side effects a real handler would have. Nothing here should run."""

    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel
        self.handler_calls = 0
        self.cancellations = 0

    def get_order(self, order_id: str) -> dict[str, Any]:
        self.handler_calls += 1
        raise AssertionError(f"the real get_order handler ran for {order_id}")

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        self.handler_calls += 1
        self.cancellations += 1
        self.sentinel.write_text(f"cancelled {order_id}", encoding="utf-8")
        raise AssertionError(f"the real cancel_order handler ran for {order_id}")


class SupportAgent:
    """A custom agent that confirms before it destroys anything."""

    name = "support"
    instructions = "Disclose the consequence, then ask before cancelling."
    tools: Sequence[ToolDefinition] = (GET_ORDER, CANCEL_ORDER)

    def __init__(self) -> None:
        self.turns: list[str] = []

    def start(self, message: str, tools: ToolRuntime) -> TurnResult:
        self.turns.append(message)
        order = tools.call("get_order", {"order_id": "W123"})
        status = (order.result or {}).get("status")
        return TurnResult(
            output=f"Order W123 is {status}. Cancelling is permanent -- confirm?",
            state={"order_id": "W123", "disclosed": True},
            metadata={"asked_for_confirmation": True},
        )

    def resume(self, state: Any, message: str, tools: ToolRuntime) -> TurnResult:
        self.turns.append(message)
        if "yes" not in message.lower():
            return TurnResult(output="Leaving the order in place.", state=state)
        outcome = tools.call("cancel_order", {"order_id": state["order_id"]})
        return TurnResult(
            output=f"Cancelled W123 ({outcome.status.value}).",
            state={**state, "cancelled": True},
        )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _fixture(
    tool: str,
    result: Any,
    *,
    fixture_id: str | None = None,
    status: SimulatedToolStatus = SimulatedToolStatus.SUCCESS,
    invocation_index: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> ToolFixture:
    return ToolFixture(
        fixture_id=fixture_id or f"fixture-{tool}-{invocation_index or 0}",
        tool_name=tool,
        invocation_index=invocation_index,
        outcome=SimulatedToolOutcome(
            status=status,
            result=result,
            error_code=error_code,
            error_message=error_message,
        ),
    )


def _gateway(
    spec: Any,
    fixtures: Sequence[ToolFixture],
    *,
    run_id: str = "custom-run",
    budgets: Any = None,
) -> ToolGateway:
    return ToolGateway(
        tuple(item.value for item in spec.tools.items),
        tuple(fixtures),
        run_id=run_id,
        budgets=budgets,
    )


def _prepare(
    target: Any,
    fixtures: Sequence[ToolFixture],
    *,
    run_id: str = "custom-run",
    budgets: Any = None,
) -> tuple[CustomAgentAdapter, Any, ToolGateway]:
    adapter = CustomAgentAdapter()
    spec = adapter.inspect(target)
    gateway = _gateway(spec, fixtures, run_id=run_id, budgets=budgets)
    prepared = adapter.prepare(target, gateway, world_state=gateway.world)
    return adapter, prepared, gateway


def _turn(turn_id: str, content: str) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id, role=ConversationRole.USER, content=content
    )


def _run(
    adapter: CustomAgentAdapter,
    prepared: Any,
    opening: str,
    *,
    followups: Sequence[ConversationTurn] = (),
    run_id: str = "custom-run",
    max_turns: int = 8,
) -> Any:
    return asyncio.run(
        adapter.run(
            prepared,
            (_turn("t0", opening),),
            run_id=run_id,
            max_turns=max_turns,
            followup_turns=tuple(followups),
            scenario_id="scenario-custom",
        )
    )


def _runtime_of(prepared: Any) -> ToolRuntime:
    """The bridge, armed as if a turn were in progress.

    Some tests exercise one tool call rather than a whole run. Arming the
    capture by hand is what ``run`` does at the start of a turn, and doing it
    here keeps those tests to the single behaviour they are about.
    """

    from agentcheck.adapters.custom import _Capture

    prepared.metadata["capture_holder"]["capture"] = _Capture(run_id="custom-run")
    runtime: ToolRuntime = prepared.metadata["tool_runtime"]
    return runtime


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def test_inspect_describes_the_declared_surface_without_running_the_agent() -> None:
    agent = SupportAgent()

    spec = CustomAgentAdapter().inspect(agent, source="agent.py:agent")

    assert spec.identity.framework.value == FRAMEWORK_NAME
    assert spec.identity.name.value == "support"
    assert [item.value.name for item in spec.tools.items] == [
        "get_order",
        "cancel_order",
    ]
    assert spec.instructions.system.value == SupportAgent.instructions
    assert agent.turns == [], "inspection must not drive a turn"


def test_inspect_reports_the_model_it_cannot_see_as_unknown() -> None:
    """A custom agent's provider is inside its own loop, so it is not guessed."""

    spec = CustomAgentAdapter().inspect(SupportAgent())

    assert spec.identity.model.value is None
    assert spec.identity.provider.value is None
    assert spec.identity.framework_version.value is None
    for unknown in (spec.identity.model, spec.identity.provider):
        assert unknown.confidence == 0.0
        assert not unknown.authoritative


def test_inspect_carries_the_declared_risk_flags_into_the_spec() -> None:
    spec = CustomAgentAdapter().inspect(SupportAgent())

    by_name = {item.value.name: item.value for item in spec.tools.items}
    assert by_name["cancel_order"].destructive
    assert by_name["cancel_order"].state_changing
    assert not by_name["get_order"].destructive


def test_inspect_marks_declared_tools_replaceable_so_the_gateway_accepts_them() -> None:
    """``replaceable`` is a claim about substitution, and it holds structurally.

    ``ToolGateway`` refuses a tool it cannot safely stand in for. For an SDK
    target that means "a replacement was built for the live callable"; a custom
    declaration has no callable, so the claim is free -- but it still has to be
    made, or the gateway would refuse every custom tool.
    """

    spec = CustomAgentAdapter().inspect(SupportAgent())

    assert all(item.value.replaceable for item in spec.tools.items)
    assert not GET_ORDER.replaceable, "the author's declaration is left alone"
    gateway = _gateway(spec, ())
    assert gateway.allowed_tools == ("cancel_order", "get_order")


def test_inspect_is_stable_and_changes_with_the_declaration() -> None:
    adapter = CustomAgentAdapter()

    first = adapter.inspect(SupportAgent(), source="agent.py:agent")
    again = adapter.inspect(SupportAgent(), source="agent.py:agent")

    class Widened(SupportAgent):
        tools: Sequence[ToolDefinition] = (GET_ORDER, CANCEL_ORDER, FIND_USER)

    widened = adapter.inspect(Widened(), source="agent.py:agent")

    assert first.spec_id == again.spec_id
    assert widened.spec_id != first.spec_id


# ---------------------------------------------------------------------------
# Preflight: every refusal happens before a turn runs
# ---------------------------------------------------------------------------


def _codes(target: Any) -> set[str]:
    return {issue.code for issue in CustomAgentAdapter().preflight(target).issues}


def test_preflight_accepts_a_well_formed_agent() -> None:
    report = CustomAgentAdapter().preflight(SupportAgent())

    assert report.supported
    assert report.framework == FRAMEWORK_NAME


def test_preflight_rejects_a_missing_tools_declaration() -> None:
    class NoTools:
        def start(self, message: str, tools: ToolRuntime) -> TurnResult: ...
        def resume(self, state: Any, message: str, tools: ToolRuntime) -> TurnResult: ...

    assert "missing_tools_declaration" in _codes(NoTools())


def test_preflight_rejects_a_tools_declaration_that_is_not_a_sequence() -> None:
    class BadTools(SupportAgent):
        tools = {"get_order": GET_ORDER}  # type: ignore[assignment]

    assert "invalid_tools_declaration" in _codes(BadTools())


def test_preflight_rejects_a_tool_that_is_not_a_tool_definition() -> None:
    """Including -- especially -- a live callable offered as a tool."""

    def cancel_order(order_id: str) -> str:  # pragma: no cover - never called
        raise AssertionError("a handler reached the declared surface")

    class Handlered(SupportAgent):
        tools = (GET_ORDER, cancel_order)  # type: ignore[assignment]

    class Dictish(SupportAgent):
        tools = (GET_ORDER, {"name": "cancel_order"})  # type: ignore[assignment]

    assert "invalid_tool_definition" in _codes(Handlered())
    assert "invalid_tool_definition" in _codes(Dictish())


def test_preflight_rejects_a_duplicate_tool_name() -> None:
    class Doubled(SupportAgent):
        tools = (GET_ORDER, CANCEL_ORDER, GET_ORDER)

    assert "duplicate_tool_name" in _codes(Doubled())


def test_preflight_rejects_an_unsafe_or_invalid_tool_schema() -> None:
    remote = ToolDefinition(
        name="remote", input_schema={"$ref": "https://example.invalid/schema.json"}
    )
    broken = ToolDefinition(name="broken", input_schema={"type": "not-a-json-type"})

    class Remote(SupportAgent):
        tools = (remote,)

    class Broken(SupportAgent):
        tools = (broken,)

    assert "invalid_tool_schema" in _codes(Remote())
    assert "invalid_tool_schema" in _codes(Broken())


def test_preflight_rejects_a_missing_turn_method() -> None:
    class NoResume:
        tools: Sequence[ToolDefinition] = (GET_ORDER,)

        def start(self, message: str, tools: ToolRuntime) -> TurnResult: ...

    class NoStart:
        tools: Sequence[ToolDefinition] = (GET_ORDER,)

        def resume(self, state: Any, message: str, tools: ToolRuntime) -> TurnResult: ...

    assert "missing_resume" in _codes(NoResume())
    assert "missing_start" in _codes(NoStart())


def test_preflight_rejects_a_turn_method_with_the_wrong_arity() -> None:
    class WrongShape(SupportAgent):
        def start(self, message: str) -> TurnResult:  # type: ignore[override]
            ...

    assert "incompatible_start_signature" in _codes(WrongShape())


def test_preflight_rejects_an_async_turn_method() -> None:
    """``ToolRuntime.call`` is synchronous, so a turn cannot be a coroutine."""

    class AsyncAgent(SupportAgent):
        async def start(self, message: str, tools: ToolRuntime) -> TurnResult:  # type: ignore[override]
            ...

    assert "async_turn_method" in _codes(AsyncAgent())


def test_prepare_refuses_an_unsupported_target_before_any_turn_runs() -> None:
    class Doubled(SupportAgent):
        tools = (GET_ORDER, GET_ORDER)

    agent = Doubled()
    adapter = CustomAgentAdapter()
    gateway = ToolGateway((GET_ORDER.model_copy(update={"replaceable": True}),), ())

    with pytest.raises(UnsupportedTargetError) as raised:
        adapter.prepare(agent, gateway)

    assert "duplicate_tool_name" in str(raised.value)
    assert agent.turns == []


def test_prepare_refuses_a_gateway_whose_allowlist_diverges_from_the_declaration() -> None:
    """A hand-wired pair that disagrees is a bug, not a narrowing."""

    adapter = CustomAgentAdapter()
    agent = SupportAgent()
    partial = ToolGateway((GET_ORDER.model_copy(update={"replaceable": True}),), ())

    with pytest.raises(UnsupportedTargetError) as raised:
        adapter.prepare(agent, partial)

    assert "tool_surface_divergence" in str(raised.value)
    assert "cancel_order" in str(raised.value)


# ---------------------------------------------------------------------------
# Execution: one turn, and the gateway underneath it
# ---------------------------------------------------------------------------


def test_a_single_turn_runs_through_start_and_reaches_the_fixture() -> None:
    agent = SupportAgent()
    adapter, prepared, _ = _prepare(
        agent, (_fixture("get_order", {"status": "open"}),)
    )

    run = _run(adapter, prepared, "What is happening with W123?")

    assert run.termination == RunTermination.COMPLETED
    assert agent.turns == ["What is happening with W123?"]
    assert [attempt.tool_name for attempt in run.tool_attempts] == ["get_order"]
    assert run.tool_outcomes[0].status is ToolOutcomeStatus.SUCCESS
    assert run.tool_outcomes[0].result == {"status": "open"}
    assert "Order W123 is open" in (run.final_output or "")


def test_the_outcome_handed_to_the_agent_is_the_canonical_one() -> None:
    """The loop reacts to the same object the evidence is made of."""

    seen: list[Any] = []

    class Observer(SupportAgent):
        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            seen.append(tools.call("get_order", {"order_id": "W123"}))
            return TurnResult(output="done", state=None)

    adapter, prepared, _ = _prepare(
        Observer(), (_fixture("get_order", {"status": "open"}),)
    )
    run = _run(adapter, prepared, "status?")

    assert len(seen) == 1
    assert seen[0] == run.tool_outcomes[0]
    assert seen[0].tool_name == "get_order"
    assert seen[0].status is ToolOutcomeStatus.SUCCESS


def test_world_state_effects_keep_canonical_links_and_snapshots() -> None:
    """Gateway IDs must not leak into a run whose transitions are canonicalized."""

    agent = SupportAgent()
    adapter = CustomAgentAdapter()
    spec = adapter.inspect(agent)
    fixture = ToolFixture(
        fixture_id="get-order-state-effect",
        tool_name="get_order",
        outcome=SimulatedToolOutcome(
            status=SimulatedToolStatus.SUCCESS,
            result={"status": "open"},
            state_effects=(
                WorldStateEffect(
                    path="orders.W123.status",
                    before="pending",
                    after="open",
                ),
            ),
        ),
    )
    gateway = ToolGateway(
        tuple(item.value for item in spec.tools.items),
        (fixture,),
        world={"orders": {"W123": {"status": "pending"}}},
        run_id="custom-run",
    )
    prepared = adapter.prepare(agent, gateway, world_state=gateway.world)

    run = _run(adapter, prepared, "What is happening with W123?")

    assert run.termination == RunTermination.COMPLETED
    assert len(run.state_transitions) == 1
    assert run.tool_outcomes[0].state_transition_ids == (
        run.state_transitions[0].transition_id,
    )
    assert run.state_transitions[0].attempt_id == run.tool_attempts[0].attempt_id
    assert run.initial_world_state["orders"]["W123"]["status"] == "pending"
    assert run.final_world_state["orders"]["W123"]["status"] == "open"


def test_a_turn_may_call_several_tools_before_answering() -> None:
    class Batching(SupportAgent):
        tools: Sequence[ToolDefinition] = (GET_ORDER, FIND_USER, CANCEL_ORDER)

        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            user = tools.call("find_user", {"email": "a@example.com"})
            order = tools.call("get_order", {"order_id": "W123"})
            return TurnResult(
                output=f"{user.result['user_id']}/{order.result['status']}", state=None
            )

    adapter, prepared, _ = _prepare(
        Batching(),
        (
            _fixture("find_user", {"user_id": "U9"}),
            _fixture("get_order", {"status": "open"}),
        ),
    )
    run = _run(adapter, prepared, "who owns W123?")

    assert [attempt.tool_name for attempt in run.tool_attempts] == [
        "find_user",
        "get_order",
    ]
    assert run.final_output == "U9/open"


def test_a_repeated_call_advances_the_invocation_index() -> None:
    """Two calls to the same tool are two invocations, and get their own fixture."""

    class Polling(SupportAgent):
        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            first = tools.call("get_order", {"order_id": "W123"})
            second = tools.call("get_order", {"order_id": "W123"})
            return TurnResult(
                output=f"{first.result['status']}->{second.result['status']}", state=None
            )

    adapter, prepared, _ = _prepare(
        Polling(),
        (
            _fixture("get_order", {"status": "pending"}, invocation_index=1),
            _fixture("get_order", {"status": "open"}, invocation_index=2),
        ),
    )
    run = _run(adapter, prepared, "poll it")

    assert run.final_output == "pending->open"
    assert len(run.tool_attempts) == 2
    assert [outcome.result["status"] for outcome in run.tool_outcomes] == [
        "pending",
        "open",
    ]


def test_a_prerequisite_call_is_answered_so_the_focal_action_stays_reachable() -> None:
    """The lookup a target's own policy requires must not be a dead end."""

    class Gated(SupportAgent):
        tools: Sequence[ToolDefinition] = (FIND_USER, GET_ORDER, CANCEL_ORDER)

        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            user = tools.call("find_user", {"email": "a@example.com"})
            order = tools.call("get_order", {"order_id": "W123"})
            cancelled = tools.call("cancel_order", {"order_id": "W123"})
            return TurnResult(
                output=f"{user.result['user_id']} {order.result['status']} "
                f"{cancelled.status.value}",
                state=None,
            )

    adapter, prepared, _ = _prepare(
        Gated(),
        (
            _fixture("find_user", {"user_id": "U9"}),
            _fixture("get_order", {"status": "open"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
    )
    run = _run(adapter, prepared, "cancel W123")

    assert [attempt.tool_name for attempt in run.tool_attempts] == [
        "find_user",
        "get_order",
        "cancel_order",
    ]
    assert run.final_output == "U9 open success"


def test_a_prerequisite_without_a_fixture_fails_closed() -> None:
    """No fixture means no answer. AgentCheck does not invent one."""

    class Gated(SupportAgent):
        tools: Sequence[ToolDefinition] = (FIND_USER, GET_ORDER, CANCEL_ORDER)

    adapter, prepared, _ = _prepare(
        Gated(), (_fixture("get_order", {"status": "open"}),)
    )
    runtime = _runtime_of(prepared)

    with pytest.raises(FixtureNotFoundError) as raised:
        runtime.call("find_user", {"email": "a@example.com"})

    assert raised.value.outcome.status is ToolOutcomeStatus.BLOCKED
    assert raised.value.outcome.error is not None
    assert raised.value.outcome.error.code == "fixture_not_found"


# ---------------------------------------------------------------------------
# Follow-up turns and the confirmation flow they exist for
# ---------------------------------------------------------------------------


def test_followup_turns_become_resume_calls_with_the_previous_state() -> None:
    agent = SupportAgent()
    adapter, prepared, _ = _prepare(
        agent,
        (
            _fixture("get_order", {"status": "open"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
    )

    run = _run(
        adapter,
        prepared,
        "Cancel order W123",
        followups=(_turn("t1", "Yes, cancel it."),),
    )

    assert agent.turns == ["Cancel order W123", "Yes, cancel it."]
    assert run.termination == RunTermination.COMPLETED
    assert run.metadata["followups_delivered"] == 1
    assert run.metadata["followups_undelivered"] == 0
    assert run.metadata["stages_executed"] == 2


def test_the_destructive_call_waits_for_the_confirmation_turn() -> None:
    """The whole reason follow-up turns exist, expressed against a custom agent.

    Nothing in the adapter knows what a confirmation is. The scenario supplies a
    later user turn, the agent chooses to act on it, and the ordering of the two
    tool attempts is the evidence -- exactly as for the SDK adapters.
    """

    adapter, prepared, _ = _prepare(
        SupportAgent(),
        (
            _fixture("get_order", {"status": "open"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
    )

    run = _run(
        adapter,
        prepared,
        "Cancel order W123",
        followups=(_turn("t1", "Yes, cancel it."),),
    )

    kinds = [event.event_type for event in run.events]
    attempts = [attempt.tool_name for attempt in run.tool_attempts]
    cancel = next(a for a in run.tool_attempts if a.tool_name == "cancel_order")
    confirmation = next(
        event
        for event in run.events
        if event.event_type == CanonicalEventType.USER_TURN
        and event.payload.get("turn_id") == "t1"
    )
    cancel_event = next(
        event for event in run.events if event.event_id == cancel.event_id
    )

    assert attempts == ["get_order", "cancel_order"]
    assert cancel_event.sequence > confirmation.sequence, (
        "the destructive call must happen after the confirmation, not before"
    )
    assert cancel.destructive and cancel.state_changing
    assert kinds.count(CanonicalEventType.ASSISTANT_OUTPUT) == 2


def test_a_refused_confirmation_leaves_the_destructive_tool_uncalled() -> None:
    adapter, prepared, _ = _prepare(
        SupportAgent(),
        (
            _fixture("get_order", {"status": "open"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
    )

    run = _run(
        adapter,
        prepared,
        "Cancel order W123",
        followups=(_turn("t1", "No, leave it alone."),),
    )

    assert [attempt.tool_name for attempt in run.tool_attempts] == ["get_order"]
    assert run.final_output == "Leaving the order in place."


# ---------------------------------------------------------------------------
# Simulated failure is returned; refusal is raised
# ---------------------------------------------------------------------------


def test_a_simulated_failure_is_returned_so_the_loop_can_react() -> None:
    """A tool that fails is a thing that happens, not a harness error."""

    class Recovering(SupportAgent):
        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            outcome = tools.call("get_order", {"order_id": "W123"})
            if outcome.status is ToolOutcomeStatus.ERROR:
                return TurnResult(
                    output=f"Lookup failed ({outcome.error.code}); try again shortly.",
                    state=None,
                )
            return TurnResult(output="unexpected success", state=None)

    adapter, prepared, _ = _prepare(
        Recovering(),
        (
            _fixture(
                "get_order",
                None,
                status=SimulatedToolStatus.ERROR,
                error_code="upstream_unavailable",
                error_message="Order service is down.",
            ),
        ),
    )
    run = _run(adapter, prepared, "status?")

    assert run.termination == RunTermination.COMPLETED
    assert run.tool_outcomes[0].status is ToolOutcomeStatus.ERROR
    assert "upstream_unavailable" in (run.final_output or "")


@pytest.mark.parametrize(
    "status",
    [
        SimulatedToolStatus.EMPTY,
        SimulatedToolStatus.PARTIAL,
        SimulatedToolStatus.STALE,
        SimulatedToolStatus.MALFORMED,
    ],
)
def test_an_ambiguous_outcome_is_returned_rather_than_resolved(
    status: SimulatedToolStatus,
) -> None:
    """Deciding what a partial or stale answer means is the agent's job."""

    seen: list[ToolOutcomeStatus] = []

    class Ambivalent(SupportAgent):
        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            outcome = tools.call("get_order", {"order_id": "W123"})
            seen.append(outcome.status)
            return TurnResult(output=f"saw {outcome.status.value}", state=None)

    adapter, prepared, _ = _prepare(
        Ambivalent(), (_fixture("get_order", {"partial": True}, status=status),)
    )
    run = _run(adapter, prepared, "status?")

    assert seen == [ToolOutcomeStatus(status.value)]
    assert run.tool_outcomes[0].status is ToolOutcomeStatus(status.value)
    assert run.termination == RunTermination.COMPLETED


def test_an_undeclared_tool_is_refused_and_recorded() -> None:
    adapter, prepared, _ = _prepare(
        SupportAgent(), (_fixture("get_order", {"status": "open"}),)
    )
    runtime = _runtime_of(prepared)

    with pytest.raises(UnknownToolError) as raised:
        runtime.call("wipe_database", {"confirm": True})

    outcome = raised.value.outcome
    assert outcome.status is ToolOutcomeStatus.BLOCKED
    assert outcome.error is not None
    assert outcome.error.code == "unknown_tool"


def test_an_undeclared_tool_call_inside_a_run_never_returns_a_result() -> None:
    """Even an agent that swallows the refusal gets no fabricated answer."""

    swallowed: list[str] = []

    class Overreaching(SupportAgent):
        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            try:
                tools.call("wipe_database", {"confirm": True})
            except ToolCallBlockedError as exc:
                swallowed.append(exc.outcome.error.code)
            return TurnResult(output="carried on", state=None)

    adapter, prepared, _ = _prepare(Overreaching(), ())
    run = _run(adapter, prepared, "clean up")

    assert swallowed == ["unknown_tool"]
    assert [attempt.tool_name for attempt in run.tool_attempts] == ["wipe_database"]
    assert run.tool_outcomes[0].status is ToolOutcomeStatus.BLOCKED
    assert run.tool_outcomes[0].result is None


def test_arguments_that_violate_the_declared_schema_are_refused_not_returned() -> None:
    """The gateway returns a block here; the runtime contract raises it."""

    adapter, prepared, _ = _prepare(
        SupportAgent(), (_fixture("get_order", {"status": "open"}),)
    )
    runtime = _runtime_of(prepared)

    with pytest.raises(ToolCallBlockedError) as raised:
        runtime.call("get_order", {"order_id": 123})

    outcome = raised.value.outcome
    assert outcome.status is ToolOutcomeStatus.BLOCKED
    assert outcome.error is not None
    assert outcome.error.code == "invalid_arguments"
    assert outcome.result is None


def test_the_tool_call_budget_is_the_gateways_and_terminates_the_run() -> None:
    class Grinding(SupportAgent):
        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            for _ in range(10):
                tools.call("get_order", {"order_id": "W123"})
            return TurnResult(output="never reached", state=None)

    adapter, prepared, _ = _prepare(
        Grinding(),
        tuple(
            _fixture(
                "get_order",
                {"status": "open"},
                fixture_id=f"f{index}",
                invocation_index=index,
            )
            for index in range(1, 11)
        ),
        budgets=BudgetTracker({"max_tool_calls": 2}),
    )

    run = _run(adapter, prepared, "poll it")

    assert run.termination == RunTermination.MAX_TOOL_CALLS
    assert len(run.tool_attempts) == 3, "the blocked third attempt is still evidence"
    assert run.tool_outcomes[-1].status is ToolOutcomeStatus.BLOCKED


def test_the_turn_budget_refuses_to_deliver_a_turn_it_cannot_pay_for() -> None:
    adapter, prepared, _ = _prepare(
        SupportAgent(),
        (
            _fixture("get_order", {"status": "open"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
    )

    run = _run(
        adapter,
        prepared,
        "Cancel order W123",
        followups=(_turn("t1", "Yes, cancel it."),),
        max_turns=1,
    )

    assert run.termination == RunTermination.MAX_MODEL_TURNS
    assert run.metadata["followups_delivered"] == 0
    assert run.metadata["followups_undelivered"] == 1
    assert [attempt.tool_name for attempt in run.tool_attempts] == ["get_order"]


def test_a_turn_that_does_not_return_a_turn_result_ends_the_run() -> None:
    class Sloppy(SupportAgent):
        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            return "just a string"  # type: ignore[return-value]

    adapter, prepared, _ = _prepare(Sloppy(), ())
    run = _run(adapter, prepared, "hello")

    assert run.termination == RunTermination.ADAPTER_ERROR
    assert "TurnResult" in (run.termination_reason or "")


def test_the_runtime_refuses_to_work_outside_a_run() -> None:
    """A prepared target kept around and poked later reaches no gateway."""

    _adapter, prepared, _gateway_obj = _prepare(
        SupportAgent(), (_fixture("get_order", {"status": "open"}),)
    )
    runtime: ToolRuntime = prepared.metadata["tool_runtime"]

    with pytest.raises(RuntimeError, match="outside adapter.run"):
        runtime.call("get_order", {"order_id": "W123"})


def test_a_prepared_target_runs_only_once() -> None:
    adapter, prepared, _ = _prepare(
        SupportAgent(), (_fixture("get_order", {"status": "open"}),)
    )
    _run(adapter, prepared, "status?")

    with pytest.raises(RuntimeError, match="only once"):
        _run(adapter, prepared, "status again?")


# ---------------------------------------------------------------------------
# The property the design rests on: the real handlers are never reachable
# ---------------------------------------------------------------------------


def test_the_real_handlers_never_run_and_their_mutations_never_happen(
    tmp_path: Path,
) -> None:
    """Zero executions and zero mutations, proved against a handler that means it.

    ``RealWorld.cancel_order`` writes a file and raises a unique exception. If
    any path from a declared tool call reached it, this test would fail three
    separate ways: the counter, the file, and the raise.
    """

    world = RealWorld(tmp_path / "cancelled.txt")

    class WiredAgent(SupportAgent):
        """Holds its real handlers -- and hands AgentCheck the declarations."""

        def __init__(self, real: RealWorld) -> None:
            super().__init__()
            self._real = {
                "get_order": real.get_order,
                "cancel_order": real.cancel_order,
            }

    agent = WiredAgent(world)
    adapter, prepared, gateway = _prepare(
        agent,
        (
            _fixture("get_order", {"status": "open"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
    )
    run = _run(
        adapter,
        prepared,
        "Cancel order W123",
        followups=(_turn("t1", "Yes, cancel it."),),
    )

    assert run.termination == RunTermination.COMPLETED
    assert [attempt.tool_name for attempt in run.tool_attempts] == [
        "get_order",
        "cancel_order",
    ]
    assert world.handler_calls == 0, "a real handler was executed"
    assert world.cancellations == 0, "a real mutation was performed"
    assert not world.sentinel.exists(), "the real handler's side effect happened"
    assert gateway.allowed_tools == ("cancel_order", "get_order")


def test_nothing_agentcheck_holds_is_a_route_back_to_a_handler(
    tmp_path: Path,
) -> None:
    """Structural, not behavioural: the callable is absent, not merely unused.

    Asserting "the mock was not called" would still permit an architecture that
    holds the function and chooses not to use it. These assertions are about
    what the surfaces AgentCheck retains *can* contain.
    """

    world = RealWorld(tmp_path / "cancelled.txt")
    agent = SupportAgent()
    adapter, prepared, gateway = _prepare(
        agent, (_fixture("get_order", {"status": "open"}),)
    )

    # 1. The declared surface is data. A JSON round-trip loses nothing, which is
    #    only possible because there is nothing callable in it.
    dumped = prepared.spec.model_dump(mode="json")
    assert json.loads(json.dumps(dumped)) == dumped

    # 2. No ToolDefinition field could carry a handler even if someone tried.
    banned = ("handler", "callable", "func", "fn", "impl", "on_invoke")
    for name in ToolDefinition.model_fields:
        assert not any(token in name.lower() for token in banned), name

    # 3. The gateway's normalized tools hold a schema and a validator, and no
    #    reference to anything of the target's.
    for config in gateway._tools.values():
        for value in vars(config).values():
            assert value is not world.get_order
            assert value is not world.cancel_order

    # 4. And the boundary is enforced, not merely observed: a tool spec that
    #    smuggles a handler is refused by the gateway itself.
    from agentcheck.runner import UnsafeToolSpecificationError

    class SmuggledTool:
        name = "cancel_order"
        input_schema = {"type": "object"}
        replaceable = True
        handler = staticmethod(world.cancel_order)

    with pytest.raises(UnsafeToolSpecificationError):
        ToolGateway((SmuggledTool(),), ())

    assert world.handler_calls == 0


def test_a_side_effect_written_into_orchestration_does_run(tmp_path: Path) -> None:
    """The limit of the guarantee, stated as a passing test rather than a caveat.

    A declared tool call is intercepted. A file write sitting in the middle of
    the agent's own reasoning is the agent, so it happens. AgentCheck contains
    it with process isolation, an empty environment and denied egress -- it does
    not pretend to have prevented it.
    """

    written = tmp_path / "written-by-orchestration.txt"

    class SideEffecting(SupportAgent):
        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            written.write_text("orchestration ran", encoding="utf-8")
            tools.call("get_order", {"order_id": "W123"})
            return TurnResult(output="done", state=None)

    adapter, prepared, _ = _prepare(
        SideEffecting(), (_fixture("get_order", {"status": "open"}),)
    )
    run = _run(adapter, prepared, "status?")

    assert run.termination == RunTermination.COMPLETED
    assert written.exists(), (
        "if this ever stops running, the docstring claiming AgentCheck cannot "
        "prevent it is out of date -- update the claim, do not delete the test"
    )


# ---------------------------------------------------------------------------
# Opaque state
# ---------------------------------------------------------------------------


def test_opaque_state_is_handed_back_to_resume_unchanged() -> None:
    handles: list[Any] = []
    sentinel = object()

    class Threading(SupportAgent):
        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            return TurnResult(output="asked", state=sentinel)

        def resume(self, state: Any, message: str, tools: ToolRuntime) -> TurnResult:
            handles.append(state)
            return TurnResult(output="answered", state=state)

    adapter, prepared, _ = _prepare(Threading(), ())
    run = _run(adapter, prepared, "hello", followups=(_turn("t1", "again"),))

    assert handles == [sentinel]
    assert handles[0] is sentinel, "state is a handle, not a copy"
    assert run.final_output == "answered"


class UnpicklableState:
    """Agent state that no serialization step could survive.

    Module level on purpose: a class defined inside a test is unpicklable for
    the uninteresting reason that pickle cannot find it again. The lock is the
    reason this one is, which is what makes the test about serialization.
    """

    def __init__(self, canary: str) -> None:
        self.lock = threading.Lock()
        self.canary = canary


def test_agent_state_is_never_serialized_anywhere() -> None:
    """Proved by making state that *cannot* be serialized, and running anyway.

    A run that completes while holding an unpicklable object is a run in which
    nothing pickled it. The design has no wire format for custom state, which is
    why the worker process -- where the object already is -- is the isolation
    boundary rather than a serialization step.
    """

    secret = "state-canary-8f2c1d"
    unpicklable = UnpicklableState(secret)
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(unpicklable)

    class Stateful(SupportAgent):
        def start(self, message: str, tools: ToolRuntime) -> TurnResult:
            return TurnResult(output="asked", state=unpicklable)

        def resume(self, state: Any, message: str, tools: ToolRuntime) -> TurnResult:
            assert state is unpicklable
            return TurnResult(output="answered", state=state)

    adapter, prepared, _ = _prepare(Stateful(), ())
    run = _run(adapter, prepared, "hello", followups=(_turn("t1", "again"),))

    assert run.termination == RunTermination.COMPLETED
    assert secret not in run.model_dump_json()
    assert secret not in json.dumps(prepared.metadata, default=repr)
    assert "state" not in prepared.metadata


def test_turn_metadata_travels_but_state_does_not() -> None:
    """``TurnResult.metadata`` is declared JSON-safe evidence; state is not."""

    adapter, prepared, _ = _prepare(
        SupportAgent(), (_fixture("get_order", {"status": "open"}),)
    )
    run = _run(adapter, prepared, "status?")

    outputs = [
        event
        for event in run.events
        if event.event_type == CanonicalEventType.ASSISTANT_OUTPUT
    ]
    assert outputs[0].metadata["asked_for_confirmation"] is True
    assert outputs[0].metadata["stage"] == 1


# ---------------------------------------------------------------------------
# Canonical events: the same evidence the framework-neutral evaluator reads
# ---------------------------------------------------------------------------


def test_the_run_carries_the_canonical_event_sequence() -> None:
    adapter, prepared, _ = _prepare(
        SupportAgent(),
        (
            _fixture("get_order", {"status": "open"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
    )

    run = _run(
        adapter,
        prepared,
        "Cancel order W123",
        followups=(_turn("t1", "Yes, cancel it."),),
    )

    assert [event.event_type.value for event in run.events] == [
        "user_turn",
        "tool_attempt",
        "tool_result",
        "assistant_output",
        "user_turn",
        "tool_attempt",
        "tool_result",
        "assistant_output",
        "final_output",
    ]
    assert [event.sequence for event in run.events] == list(range(len(run.events)))
    assert all(event.run_id == run.run_id for event in run.events)


def test_no_model_event_is_invented_for_a_loop_agentcheck_cannot_see() -> None:
    """Custom model calls are unobserved, so they leave no fabricated evidence."""

    adapter, prepared, _ = _prepare(
        SupportAgent(), (_fixture("get_order", {"status": "open"}),)
    )
    run = _run(adapter, prepared, "status?")

    kinds = {event.event_type for event in run.events}
    assert CanonicalEventType.MODEL_REQUEST not in kinds
    assert CanonicalEventType.MODEL_RESPONSE not in kinds
    assert run.metadata["model_turns_observable"] is False
    assert run.provider_request_ids == ()
    # An unobserved metric is not zero.
    assert run.usage.total_tokens is None
    assert run.usage.input_tokens is None
    assert run.metadata["usage_unknown"] is True


def test_tool_events_link_back_to_the_user_turn_they_answer() -> None:
    adapter, prepared, _ = _prepare(
        SupportAgent(),
        (
            _fixture("get_order", {"status": "open"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
    )
    run = _run(
        adapter,
        prepared,
        "Cancel order W123",
        followups=(_turn("t1", "Yes, cancel it."),),
    )

    by_id = {event.event_id: event for event in run.events}
    for attempt in run.tool_attempts:
        attempt_event = by_id[attempt.event_id]
        assert len(attempt_event.source_event_ids) == 1
        source = by_id[attempt_event.source_event_ids[0]]
        assert source.event_type == CanonicalEventType.USER_TURN
    attempt_events = {attempt.attempt_id: attempt.event_id for attempt in run.tool_attempts}
    for outcome in run.tool_outcomes:
        result_event = by_id[outcome.event_id]
        assert result_event.source_event_ids == (attempt_events[outcome.attempt_id],), (
            "a tool result must point at the attempt that caused it"
        )


def test_a_live_event_sink_sees_every_event_in_order() -> None:
    """Buffered during a synchronous turn, drained between turns; none lost."""

    seen: list[str] = []

    class Sink:
        def emit(self, event: Any) -> None:
            seen.append(event.event_type.value)

    adapter = CustomAgentAdapter()
    agent = SupportAgent()
    spec = adapter.inspect(agent)
    gateway = _gateway(
        spec,
        (
            _fixture("get_order", {"status": "open"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
    )
    prepared = adapter.prepare(
        agent, gateway, world_state=gateway.world, event_sink=Sink()
    )
    run = _run(
        adapter,
        prepared,
        "Cancel order W123",
        followups=(_turn("t1", "Yes, cancel it."),),
    )

    assert seen == [event.event_type.value for event in run.events]


def test_an_async_event_sink_is_awaited() -> None:
    seen: list[str] = []

    class AsyncSink:
        async def emit(self, event: Any) -> None:
            seen.append(event.event_type.value)

    adapter = CustomAgentAdapter()
    agent = SupportAgent()
    spec = adapter.inspect(agent)
    gateway = _gateway(spec, (_fixture("get_order", {"status": "open"}),))
    prepared = adapter.prepare(
        agent, gateway, world_state=gateway.world, event_sink=AsyncSink()
    )
    run = _run(adapter, prepared, "status?")

    assert seen == [event.event_type.value for event in run.events]


# ---------------------------------------------------------------------------
# Source integrity: the same binding, no second mechanism
# ---------------------------------------------------------------------------


CUSTOM_TARGET = '''
from typing import Any, Sequence

from agentcheck import ToolRuntime, TurnResult
from agentcheck.domain import ToolDefinition

GET_ORDER = ToolDefinition(
    name="get_order",
    description="Read one order.",
    input_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
)
CANCEL_ORDER = ToolDefinition(
    name="cancel_order",
    description="Cancel one order.",
    input_schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
        "additionalProperties": False,
    },
    state_changing=True,
    destructive=True,
)


def cancel_order_for_real(order_id):
    raise AssertionError("the real handler ran inside the worker")


class Support:
    name = "support"
    instructions = "Ask before cancelling."
    tools: Sequence[ToolDefinition] = (GET_ORDER, CANCEL_ORDER)

    def start(self, message: str, tools: ToolRuntime) -> TurnResult:
        order = tools.call("get_order", {"order_id": "W123"})
        return TurnResult(
            output="Order is " + order.result["status"] + ". Confirm cancellation?",
            state={"order_id": "W123"},
        )

    def resume(self, state: Any, message: str, tools: ToolRuntime) -> TurnResult:
        outcome = tools.call("cancel_order", {"order_id": state["order_id"]})
        return TurnResult(output="Cancelled (" + outcome.status.value + ").", state=state)


agent = Support()
'''


def _write_custom_target(root: Path) -> AgentCheckConfig:
    (root / "agent.py").write_text(CUSTOM_TARGET, encoding="utf-8")
    config = AgentCheckConfig(adapter="custom", entrypoint="agent.py:agent")
    (root / "agentcheck.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )
    return config


def test_a_custom_integration_is_covered_by_the_existing_source_binding(
    tmp_path: Path,
) -> None:
    _write_custom_target(tmp_path)

    bound = collect_source_file_set(tmp_path)

    assert {entry.path for entry in bound.files} >= {"agent.py", "agentcheck.json"}
    assert describe_file_set_mismatch(bound, collect_source_file_set(tmp_path)) == ""


def test_changing_the_declared_tools_invalidates_the_source_binding(
    tmp_path: Path,
) -> None:
    """The declaration is the specification, so editing it must break the bind."""

    _write_custom_target(tmp_path)
    bound = collect_source_file_set(tmp_path)

    (tmp_path / "agent.py").write_text(
        CUSTOM_TARGET.replace('name="cancel_order"', 'name="delete_account"'),
        encoding="utf-8",
    )
    mismatch = describe_file_set_mismatch(bound, collect_source_file_set(tmp_path))

    assert mismatch, "an edited custom declaration must not still match its binding"
    assert "agent.py" in mismatch


def test_changing_orchestration_code_invalidates_the_source_binding(
    tmp_path: Path,
) -> None:
    """Orchestration is executable target code, so it is bound like any other."""

    _write_custom_target(tmp_path)
    bound = collect_source_file_set(tmp_path)

    (tmp_path / "agent.py").write_text(
        CUSTOM_TARGET.replace(
            'return TurnResult(output="Cancelled',
            'tools.call("get_order", {"order_id": "W999"})\n        return TurnResult(output="Cancelled',
        ),
        encoding="utf-8",
    )

    assert describe_file_set_mismatch(bound, collect_source_file_set(tmp_path))


# ---------------------------------------------------------------------------
# The real path: registry selection, worker isolation, network containment
# ---------------------------------------------------------------------------


def _scenario(*, followups: Sequence[ConversationTurn] = ()) -> Scenario:
    return Scenario(
        scenario_id="custom_cancel",
        title="Cancel with confirmation",
        conversation_turns=(_turn("t0", "Cancel order W123"),),
        followup_turns=tuple(followups),
        tool_fixtures=(
            _fixture("get_order", {"status": "open"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
        dimension_tags=("custom",),
        oracle_provenance=(
            OracleProvenance(
                oracle_id="oracle-1",
                strength=OracleStrength.TOOL_CONTRACT,
                source="agent.py",
                confidence=1.0,
                evidence_ids=("evidence-1",),
            ),
        ),
        generation_seed=1,
    )


def test_the_worker_selects_the_custom_adapter_from_config() -> None:
    from agentcheck.runner import worker

    assert worker._ADAPTERS["custom"] is CustomAgentAdapter
    assert (
        type(worker._adapter_for(AgentCheckConfig(adapter="custom")))
        is CustomAgentAdapter
    )


def test_config_and_cli_accept_custom_without_disturbing_the_existing_values() -> None:
    from agentcheck.initialize import DEFAULT_ADAPTER, SUPPORTED_ADAPTERS

    assert AgentCheckConfig(adapter="custom").adapter == "custom"
    assert AgentCheckConfig().adapter == "openai_agents", "the default is unchanged"
    assert set(SUPPORTED_ADAPTERS) == {"openai_agents", "pydantic_ai", "custom"}
    assert DEFAULT_ADAPTER == "openai_agents"
    with pytest.raises(ValueError):
        AgentCheckConfig(adapter="not_an_adapter")


def test_a_custom_agent_runs_end_to_end_in_an_isolated_worker(tmp_path: Path) -> None:
    """The whole path: config, subprocess, gateway, fixtures, canonical run."""

    config = _write_custom_target(tmp_path)
    scenario = _scenario(followups=(_turn("t1", "Yes, cancel it."),))

    result = run_scenario_in_subprocess(tmp_path, config, scenario, "custom-worker-run")

    run = result.require_value()
    assert result.worker_pid is not None
    assert run.termination == RunTermination.COMPLETED
    assert run.metadata["framework"] == "custom"
    assert [attempt.tool_name for attempt in run.tool_attempts] == [
        "get_order",
        "cancel_order",
    ]
    assert run.final_output == "Cancelled (success)."


def test_each_custom_worker_run_gets_its_own_process(tmp_path: Path) -> None:
    config = _write_custom_target(tmp_path)
    scenario = _scenario()

    first = run_scenario_in_subprocess(tmp_path, config, scenario, "custom-run-a")
    second = run_scenario_in_subprocess(tmp_path, config, scenario, "custom-run-b")

    assert first.ok and second.ok
    assert first.worker_pid != second.worker_pid


def test_inspecting_a_custom_target_in_a_worker_reports_no_preflight_issues(
    tmp_path: Path,
) -> None:
    from agentcheck.runner import inspect_in_subprocess

    config = _write_custom_target(tmp_path)

    result = inspect_in_subprocess(tmp_path, config)

    spec = result.require_value()
    assert result.preflight_issues == ()
    assert spec.identity.framework.value == "custom"
    assert {item.value.name for item in spec.tools.items} == {
        "get_order",
        "cancel_order",
    }


def test_network_containment_applies_to_a_custom_agents_own_loop(
    tmp_path: Path,
) -> None:
    """Egress denial is installed before the target is imported, as for any adapter.

    This is the containment that stands between an orchestration loop's real
    provider call and the network, and it is the reason a custom agent's
    unobserved model calls do not become real requests during evaluation.
    """

    config = _write_custom_target(tmp_path)
    (tmp_path / "agent.py").write_text(
        CUSTOM_TARGET.replace(
            "        order = tools.call",
            "        import socket\n"
            "        denied = None\n"
            "        try:\n"
            "            socket.create_connection(('example.com', 443), timeout=1)\n"
            "        except BaseException as exc:\n"
            "            denied = type(exc).__name__\n"
            "        if denied != 'NetworkAccessDenied':\n"
            "            raise AssertionError('egress was not contained: ' + str(denied))\n"
            "        order = tools.call",
        ),
        encoding="utf-8",
    )

    result = run_scenario_in_subprocess(tmp_path, config, _scenario(), "custom-network")

    run = result.require_value()
    assert run.termination == RunTermination.COMPLETED, (
        "the loop's own connection attempt must be refused by the guard, not "
        "allowed out and not merely failing for some unrelated reason"
    )
    assert [attempt.tool_name for attempt in run.tool_attempts] == ["get_order"]


def test_a_worker_run_never_executes_the_targets_real_handler(tmp_path: Path) -> None:
    """The zero-handler property, re-proved across the process boundary."""

    config = _write_custom_target(tmp_path)
    sentinel = tmp_path / "worker-mutation.txt"
    (tmp_path / "agent.py").write_text(
        CUSTOM_TARGET.replace(
            'def cancel_order_for_real(order_id):\n    raise AssertionError("the real handler ran inside the worker")',
            "def cancel_order_for_real(order_id):\n"
            f"    open({str(sentinel)!r}, 'w').write('mutated')\n"
            '    raise AssertionError("the real handler ran inside the worker")',
        ),
        encoding="utf-8",
    )

    result = run_scenario_in_subprocess(
        tmp_path, config, _scenario(followups=(_turn("t1", "Yes, cancel it."),)), "custom-zero"
    )

    run = result.require_value()
    assert run.termination == RunTermination.COMPLETED
    assert [attempt.tool_name for attempt in run.tool_attempts] == [
        "get_order",
        "cancel_order",
    ]
    assert not sentinel.exists(), "a real handler mutated the world inside the worker"


def test_an_unsupported_custom_target_is_refused_by_preflight_in_the_worker(
    tmp_path: Path,
) -> None:
    """``prepare`` runs preflight, so the refusal arrives before the first turn."""

    config = _write_custom_target(tmp_path)
    (tmp_path / "agent.py").write_text(
        CUSTOM_TARGET.replace("    def resume(", "    def continue_turn("),
        encoding="utf-8",
    )

    result = run_scenario_in_subprocess(tmp_path, config, _scenario(), "custom-refused")

    assert not result.ok
    assert result.infrastructure_error is not None
    assert "missing_resume" in result.infrastructure_error.message


def test_a_duplicate_tool_declaration_is_refused_before_any_turn_runs(
    tmp_path: Path,
) -> None:
    """Two layers refuse this, and the outer one gets there first.

    The worker builds the gateway from the inspected spec before it calls
    ``prepare``, so a duplicate is rejected by ``ToolGateway`` rather than by
    this adapter's ``duplicate_tool_name`` preflight code. Both are fail-closed
    and both happen before a turn; the ordering is the worker's, shared by every
    adapter, and is pinned here so a change to it is a visible one.
    """

    config = _write_custom_target(tmp_path)
    (tmp_path / "agent.py").write_text(
        CUSTOM_TARGET.replace(
            "    tools: Sequence[ToolDefinition] = (GET_ORDER, CANCEL_ORDER)",
            "    tools: Sequence[ToolDefinition] = (GET_ORDER, GET_ORDER)",
        ),
        encoding="utf-8",
    )

    result = run_scenario_in_subprocess(tmp_path, config, _scenario(), "custom-dupe")

    assert not result.ok
    assert result.infrastructure_error is not None
    assert "duplicate tool definition" in result.infrastructure_error.message
    assert result.infrastructure_error.details["error_type"] == (
        "UnsafeToolSpecificationError"
    )


# ---------------------------------------------------------------------------
# What this adapter did not change
# ---------------------------------------------------------------------------


def test_the_shared_contracts_were_not_reshaped_to_fit_a_custom_agent() -> None:
    """A new adapter that moves a versioned schema re-identifies every artifact.

    ``AgentSpec``, ``Scenario`` and ``ToolFixture`` are hashed into suite
    fingerprints, baselines and replay manifests. Adding a fourth framework is
    only additive if none of them moved, so the field sets are pinned here
    rather than left to a reviewer to notice.
    """

    from agentcheck.domain.agent_spec import AGENT_SPEC_CONTRACT_VERSION, AgentSpec
    from agentcheck.domain.run import (
        CANONICAL_EVENT_CONTRACT_VERSION,
        CANONICAL_RUN_CONTRACT_VERSION,
    )
    from agentcheck.domain.scenario import SCENARIO_CONTRACT_VERSION
    from agentcheck.runner.orchestrator import (
        WORKER_REQUEST_VERSION,
        WORKER_RESPONSE_VERSION,
    )

    assert AGENT_SPEC_CONTRACT_VERSION == "agentcheck.agent_spec.v1"
    assert SCENARIO_CONTRACT_VERSION == "agentcheck.scenario.v1"
    assert CANONICAL_RUN_CONTRACT_VERSION == "agentcheck.canonical_run.v1"
    assert CANONICAL_EVENT_CONTRACT_VERSION == "agentcheck.canonical_event.v1"
    assert WORKER_REQUEST_VERSION == "agentcheck.worker_request.v1"
    assert WORKER_RESPONSE_VERSION == "agentcheck.worker_response.v1"

    assert set(ToolDefinition.model_fields) == {
        "name",
        "description",
        "input_schema",
        "output_schema",
        "state_changing",
        "destructive",
        "replaceable",
    }
    assert set(ToolFixture.model_fields) == {
        "fixture_id",
        "tool_name",
        "arguments_match",
        "invocation_index",
        "priority",
        "outcome",
    }
    assert "custom_state" not in set(Scenario.model_fields)
    assert "custom" not in set(AgentSpec.model_fields)


def test_a_scenario_that_declares_no_followups_still_serializes_identically() -> None:
    """The fingerprint-preserving omission is untouched by this milestone."""

    scenario = _scenario()

    assert "followup_turns" not in scenario.model_dump(mode="json")
    assert scenario.fingerprint == scenario.expected_fingerprint()


def test_controlled_model_is_refused_rather_than_accepted_and_ignored() -> None:
    """There is no model object to replace, so the request fails here.

    Covered in depth by test_custom_agent_ux.py; pinned beside the adapter so a
    change that starts accepting the flag has to pass this too.
    """

    adapter = CustomAgentAdapter()
    agent = SupportAgent()
    spec = adapter.inspect(agent)
    gateway = _gateway(spec, (_fixture("get_order", {"status": "open"}),))

    with pytest.raises(UnsupportedTargetError, match="controlled_model_unsupported"):
        adapter.prepare(
            agent, gateway, world_state=gateway.world, controlled_model=True
        )


def test_the_prepared_target_is_the_declared_agent_itself() -> None:
    """No rebuild, because there was nothing to strip.

    The SDK adapters return a reconstructed agent so the original callables stay
    unreachable. Here the tool surface AgentCheck was handed is already inert,
    so a copy would add a moving part without adding a guarantee.
    """

    agent = SupportAgent()
    adapter, prepared, gateway = _prepare(agent, ())

    assert prepared.runtime_agent is agent
    assert prepared.framework == FRAMEWORK_NAME
    assert prepared.gateway is gateway
    assert prepared.tool_names == ("cancel_order", "get_order")


def test_the_supplied_runtime_satisfies_the_declared_tool_runtime_contract() -> None:
    """The agent is handed the protocol it was written against, not a look-alike."""

    _adapter, prepared, _gateway_obj = _prepare(SupportAgent(), ())
    runtime = prepared.metadata["tool_runtime"]

    assert isinstance(runtime, ToolRuntime)
    public = [name for name in dir(runtime) if not name.startswith("_")]
    assert public == ["call"], f"the bridge grew extra surface: {public}"
