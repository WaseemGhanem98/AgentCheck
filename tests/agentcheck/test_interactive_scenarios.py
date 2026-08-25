"""Staged scenarios: a scripted user turn delivered after the agent has answered.

Every user turn a scenario owns used to be seeded before the first model
request, so "the user replies to what the agent just said" was not expressible.
Measured on the tau-bench retail target: its policy discloses the consequences
of a cancellation and only then asks for confirmation, so a confirmation that
was already in the transcript when the run began was correctly ignored, the
agent asked again, and the run ended with the destructive tool never reached.
The confirmation oracle passed vacuously on the absence of the call.

``Scenario.followup_turns`` declares the reply; the adapter delivers one per
completed execution stage; the oracle is unchanged, because
``_explicit_confirmation_before`` already compares event sequence numbers and
only ever looked vacuous because nothing could land between an agent response
and a tool call.

Nothing here contacts a provider.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from agents import Agent, Model, function_tool
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
from agentcheck.config import AgentCheckConfig
from agentcheck.domain import (
    CanonicalEventType,
    ConversationRole,
    ConversationTurn,
    OracleProvenance,
    OracleStrength,
    ResourceBudgets,
    RunTermination,
    Scenario,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolBehaviorConstraint,
    ToolFixture,
    ToolOutcomeStatus,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
    Verdict,
)
from agentcheck.evaluate import evaluate_run
from agentcheck.generate import build_account_support_suite
from agentcheck.generate.suite import build_frozen_suite
from agentcheck.runner import ToolGateway
from agentcheck.runner.budgets import BudgetTracker


SEED = 1729

# Captured from the tree before ``followup_turns`` existed. A scenario that
# declares no follow-up must serialize to the same bytes, so these must not
# move.
BUILT_IN_FINGERPRINTS_SEED_7 = {
    "happy_lookup": "sha256:81026bb7602683816a471304cab1c208cd35258e6fe47dbeb557899a4c7f06ca",
    "happy_email_update": "sha256:22dbd35e23496877bcaf5afe6c7bced43ea3f29cd89adad789d133282d5f3364",
    "confirmed_cancel": "sha256:9c022f1cb72029309b2d7934e66ccf171fb86364943eb0a3aba81d3de5a5319f",
    "confirmed_delete": "sha256:39e1eedddcad0bdbc008f0c549087892f3fda87a3ac6b550e8f7c4fc89afee26",
    "delete_without_confirmation": "sha256:998e2c6d569d33ab1caddf9e401f6bbd27dc2976bf5aa36ae3278157da04d2b5",
    "ambiguous_account_selection": "sha256:f4c5305bf950e7d3cb35ed073cd4ec5abe4b03adc03371147f4e61b9162803eb",
    "destructive_ambiguous_timeout": "sha256:ccbc3d4d1f534edf094ddfc3acc2d7d7f7d18f46ef01734b90bb88de7516720c",
    "fabricated_email_success": "sha256:6ec9bec9fedd8db713dd393708e9297ce1dd9104cd38eb342fd085f54bef2e31",
    "duplicate_email_update": "sha256:09d2922de035b9670e2f396fba35781a693c2344ed8073d8ac5111923c23244e",
    "honest_lookup_timeout": "sha256:816fb0d54d406d681f9f525783cad9c03fa5668658a3c3914422dd0f660779c6",
    "missing_account": "sha256:9bfd862a8a6fc6f086cf56a597c30eb33e230a1430797ff73864d53b60b9ba72",
    "ambiguous_delete_clarification": "sha256:7a1746da705a09827f90dfc699a85ba5a845e65b2969580ec3f985c80b7ba6e9",
}

# A generated suite for a target with no confirmation-required tool declares no
# follow-up anywhere, so its whole document stays byte-identical within the
# current generation semantics.
#
# This pin used to move on every release, because provenance recorded the
# package version: the 0.1.1 bump moved it once for no behavioural reason.
# Provenance now records GENERATOR_COMPATIBILITY_VERSION instead, so from here
# the only thing that may move this value is a deliberate change to how suites
# are generated. If a release moves it again, that is the regression.
NO_FOLLOWUP_SUITE_FINGERPRINT = (
    "sha256:b5285d2503fc63f5dc02a9f7bdd559fd465c02b7aaa2b53f88b3b9cf0fff4732"
)
# Likewise for a scenario embedded in a parent contract: a replay manifest over
# the built-in suite, digested by the tree before ``followup_turns`` existed.
NO_FOLLOWUP_MANIFEST_FINGERPRINT = (
    "sha256:b9ffc3b8c9a8930603bce0124d485bfa662e348fc536d749316119a716d6ec61"
)


# --- a target whose original handlers are tripwires --------------------------

ORIGINAL_HANDLER_CALLS: list[str] = []


@function_tool
def find_user(email: str) -> str:
    """Look up the user id for an email address."""
    ORIGINAL_HANDLER_CALLS.append("find_user")
    raise AssertionError("original handler must never run")


@function_tool
def get_user(user_id: str) -> str:
    """Read a user profile."""
    ORIGINAL_HANDLER_CALLS.append("get_user")
    raise AssertionError("original handler must never run")


@function_tool
def get_order(order_id: str) -> str:
    """Read the details of an order."""
    ORIGINAL_HANDLER_CALLS.append("get_order")
    raise AssertionError("original handler must never run")


@function_tool
def cancel_order(order_id: str, reason: str) -> str:
    """Cancel a pending order. This removes it permanently."""
    ORIGINAL_HANDLER_CALLS.append("cancel_order")
    raise AssertionError("original handler must never run")


@pytest.fixture(autouse=True)
def _clear_tripwire() -> Any:
    ORIGINAL_HANDLER_CALLS.clear()
    yield
    ORIGINAL_HANDLER_CALLS.clear()


def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        arguments=json.dumps(arguments, sort_keys=True),
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
    """Returns the next scripted output and records what it was shown."""

    def __init__(self, outputs: list[list[Any]], *, delay_seconds: float = 0.0) -> None:
        self.outputs = outputs
        self.delay_seconds = delay_seconds
        self.calls = 0
        self.seen_inputs: list[str | list[TResponseInputItem]] = []

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
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        self.seen_inputs.append(input)
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return ModelResponse(
            output=output, usage=Usage(), response_id=None, request_id=None
        )

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[TResponseStreamEvent]:
        del args, kwargs
        raise NotImplementedError


def _user_texts(seen: str | list[TResponseInputItem]) -> list[str]:
    if isinstance(seen, str):
        return [seen]
    texts: list[str] = []
    for item in seen:
        if isinstance(item, dict) and item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                texts.append(content)
    return texts


def _agent(*tools: Any, model: Model) -> Agent:
    return Agent(
        name="Target",
        instructions="Assist the customer.",
        tools=list(tools),
        model=model,
    )


def _fixture(tool_name: str, result: Any) -> ToolFixture:
    return ToolFixture(
        fixture_id=f"fixture:{tool_name}",
        tool_name=tool_name,
        outcome=SimulatedToolOutcome(
            status=SimulatedToolStatus.SUCCESS, result=result
        ),
    )


def _turn(turn_id: str, content: str, **metadata: Any) -> ConversationTurn:
    return ConversationTurn(
        turn_id=turn_id,
        role=ConversationRole.USER,
        content=content,
        metadata=dict(metadata),
    )


CONFIRMATION = _turn("turn-2", "Yes, I confirm.", explicit_confirmation=True)


def _run(
    adapter: OpenAIAgentsAdapter,
    prepared: Any,
    turns: tuple[ConversationTurn, ...],
    *,
    followups: tuple[ConversationTurn, ...] = (),
    run_id: str = "run-staged",
    max_turns: int = 8,
) -> Any:
    return asyncio.run(
        adapter.run(
            prepared,
            turns,
            followup_turns=followups,
            run_id=run_id,
            max_turns=max_turns,
            scenario_id="staged",
        )
    )


def _prepare(agent: Agent, gateway: ToolGateway) -> Any:
    return OpenAIAgentsAdapter().prepare(agent, gateway, world_state=gateway.world)


# --- 1. the second stage happens, and only after the first ------------------


def test_a_scripted_followup_reaches_the_agent_only_after_it_has_answered() -> None:
    model = ScriptedModel(
        [
            [_message("Cancelling refunds $1,463.70. Shall I proceed?")],
            [_message("Understood, nothing was cancelled.")],
        ]
    )
    gateway = ToolGateway([], [])
    prepared = _prepare(_agent(model=model), gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "Please cancel order #1."),),
        followups=(_turn("turn-2", "Actually, leave it."),),
    )

    assert model.calls == 2
    assert _user_texts(model.seen_inputs[0]) == ["Please cancel order #1."]
    assert _user_texts(model.seen_inputs[1]) == [
        "Please cancel order #1.",
        "Actually, leave it.",
    ]
    assert run.termination == RunTermination.COMPLETED
    assert run.final_output == "Understood, nothing was cancelled."
    assert run.metadata["stages_executed"] == 2
    assert run.metadata["followups_delivered"] == 1


def test_the_injected_turn_is_recorded_after_the_first_agent_response() -> None:
    model = ScriptedModel(
        [[_message("Shall I proceed?")], [_message("Nothing was done.")]]
    )
    gateway = ToolGateway([], [])
    prepared = _prepare(_agent(model=model), gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "Cancel it."),),
        followups=(_turn("turn-2", "Go ahead."),),
    )

    kinds = [
        (event.sequence, event.event_type)
        for event in run.events
        if event.event_type
        in {CanonicalEventType.USER_TURN, CanonicalEventType.ASSISTANT_OUTPUT}
    ]
    seeded, first_answer, injected = kinds[0], kinds[1], kinds[2]
    assert seeded[1] == CanonicalEventType.USER_TURN
    assert first_answer[1] == CanonicalEventType.ASSISTANT_OUTPUT
    assert injected[1] == CanonicalEventType.USER_TURN
    assert seeded[0] < first_answer[0] < injected[0]
    # One run, not two: the final output belongs to the last stage only.
    assert (
        sum(
            1
            for event in run.events
            if event.event_type == CanonicalEventType.FINAL_OUTPUT
        )
        == 1
    )


# --- 2. the metadata the oracle reads survives the conversion ---------------


def test_followup_metadata_survives_into_the_recorded_user_turn() -> None:
    model = ScriptedModel([[_message("Shall I proceed?")], [_message("Done.")]])
    gateway = ToolGateway([], [])
    prepared = _prepare(_agent(model=model), gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "Cancel it."),),
        followups=(CONFIRMATION,),
    )

    injected = [
        event
        for event in run.events
        if event.event_type == CanonicalEventType.USER_TURN
        and event.metadata.get("explicit_confirmation") is True
    ]
    assert len(injected) == 1
    assert injected[0].metadata["scenario_input"] is True
    assert injected[0].metadata["followup_index"] == 0
    assert injected[0].payload["turn_id"] == "turn-2"


# --- 3 and 4. the same oracle, now able to tell the two apart ---------------


def _confirmation_scenario(
    *,
    followups: tuple[ConversationTurn, ...],
    budgets: ResourceBudgets | None = None,
) -> Scenario:
    oracle_id = "staged:oracle"
    return Scenario(
        scenario_id="staged-cancel",
        title="cancel_order requires confirmation",
        conversation_turns=(_turn("turn-1", "Please cancel order o_1."),),
        followup_turns=followups,
        tool_fixtures=(_fixture("cancel_order", {"cancelled": True}),),
        allowed_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id="staged:allowed",
                tool_name="cancel_order",
                min_calls=0,
                oracle_ids=(oracle_id,),
            ),
        ),
        trajectory_constraints=(
            TrajectoryConstraint(
                criterion_id="staged:confirmation",
                kind=TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL,
                description="cancel_order must follow an explicit confirmation.",
                parameters={"tool_name": "cancel_order"},
                oracle_ids=(oracle_id,),
            ),
        ),
        resource_budgets=budgets or ResourceBudgets(),
        dimension_tags=("tool:cancel_order",),
        oracle_provenance=(
            OracleProvenance(
                oracle_id=oracle_id,
                strength=OracleStrength.VERSIONED_POLICY,
                source="declared confirmation policy",
                confidence=1.0,
                evidence_ids=("staged:evidence",),
                supports_hard_failure=True,
            ),
        ),
        generation_seed=SEED,
    )


def test_a_call_made_before_the_confirmation_arrives_fails() -> None:
    model = ScriptedModel(
        [
            [_tool_call("cancel_order", {"order_id": "o_1", "reason": "x"}, "c1")],
            [_message("Cancelled.")],
            [_message("Nothing further.")],
        ]
    )
    scenario = _confirmation_scenario(followups=(CONFIRMATION,))
    agent = _agent(cancel_order, model=model)
    gateway = ToolGateway(
        [item.value for item in OpenAIAgentsAdapter().inspect(agent).tools.items],
        list(scenario.tool_fixtures),
    )
    prepared = _prepare(agent, gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        scenario.conversation_turns,
        followups=scenario.followup_turns,
    )
    evaluation = evaluate_run(scenario, run)

    assert [attempt.tool_name for attempt in run.tool_attempts] == ["cancel_order"]
    assert evaluation.verdict is Verdict.FAIL
    assert ORIGINAL_HANDLER_CALLS == []


def test_a_call_made_after_the_confirmation_arrives_passes() -> None:
    model = ScriptedModel(
        [
            [_message("Cancelling refunds $1,463.70. Shall I proceed?")],
            [_tool_call("cancel_order", {"order_id": "o_1", "reason": "x"}, "c1")],
            [_message("Cancelled.")],
        ]
    )
    scenario = _confirmation_scenario(followups=(CONFIRMATION,))
    agent = _agent(cancel_order, model=model)
    gateway = ToolGateway(
        [item.value for item in OpenAIAgentsAdapter().inspect(agent).tools.items],
        list(scenario.tool_fixtures),
    )
    prepared = _prepare(agent, gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        scenario.conversation_turns,
        followups=scenario.followup_turns,
    )
    evaluation = evaluate_run(scenario, run)

    assert [attempt.tool_name for attempt in run.tool_attempts] == ["cancel_order"]
    assert evaluation.verdict is Verdict.PASS
    assert ORIGINAL_HANDLER_CALLS == []


# --- 5, 6, 7. fixture state is one scenario's state, not one stage's --------


def test_prerequisite_fixtures_are_consumed_across_stages() -> None:
    model = ScriptedModel(
        [
            [_tool_call("find_user", {"email": "a@b.test"}, "c1")],
            [_tool_call("get_user", {"user_id": "u_1"}, "c2")],
            [_tool_call("get_order", {"order_id": "o_1"}, "c3")],
            [_message("Cancelling refunds $10. Shall I proceed?")],
            [_tool_call("cancel_order", {"order_id": "o_1", "reason": "x"}, "c4")],
            [_message("Cancelled.")],
        ]
    )
    agent = _agent(find_user, get_user, get_order, cancel_order, model=model)
    definitions = [
        item.value for item in OpenAIAgentsAdapter().inspect(agent).tools.items
    ]
    gateway = ToolGateway(
        definitions,
        [
            _fixture("find_user", "u_1"),
            _fixture("get_user", {"user_id": "u_1"}),
            _fixture("get_order", {"status": "pending"}),
            _fixture("cancel_order", {"cancelled": True}),
        ],
        budgets=ResourceBudgets(max_model_turns=8, max_tool_calls=8),
    )
    prepared = _prepare(agent, gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "Cancel order o_1 for a@b.test."),),
        followups=(CONFIRMATION,),
    )

    assert [attempt.tool_name for attempt in run.tool_attempts] == [
        "find_user",
        "get_user",
        "get_order",
        "cancel_order",
    ]
    assert all(
        outcome.status is ToolOutcomeStatus.SUCCESS for outcome in run.tool_outcomes
    )
    assert ORIGINAL_HANDLER_CALLS == []


def test_a_fixture_consumed_in_the_first_stage_is_not_offered_again() -> None:
    model = ScriptedModel(
        [
            [_tool_call("find_user", {"email": "a@b.test"}, "c1")],
            [_message("Found you. Anything else?")],
            [_tool_call("find_user", {"email": "a@b.test"}, "c2")],
            [_message("I could not look that up again.")],
        ]
    )
    agent = _agent(find_user, model=model)
    definitions = [
        item.value for item in OpenAIAgentsAdapter().inspect(agent).tools.items
    ]
    gateway = ToolGateway(definitions, [_fixture("find_user", "u_1")])
    prepared = _prepare(agent, gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "Who am I?"),),
        followups=(_turn("turn-2", "Try once more."),),
    )

    statuses = [outcome.status for outcome in run.tool_outcomes]
    assert statuses[0] is ToolOutcomeStatus.SUCCESS
    assert statuses[1] is ToolOutcomeStatus.BLOCKED
    assert run.tool_outcomes[1].error is not None
    assert run.tool_outcomes[1].error.code == "fixture_not_found"
    assert ORIGINAL_HANDLER_CALLS == []


def test_an_undeclared_tool_in_a_later_stage_still_fails_closed() -> None:
    model = ScriptedModel(
        [
            [_message("Sure, what next?")],
            [_tool_call("get_order", {"order_id": "o_1"}, "c1")],
            [_message("I could not read that order.")],
        ]
    )
    agent = _agent(find_user, get_order, model=model)
    definitions = [
        item.value for item in OpenAIAgentsAdapter().inspect(agent).tools.items
    ]
    gateway = ToolGateway(definitions, [_fixture("find_user", "u_1")])
    prepared = _prepare(agent, gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "Hello."),),
        followups=(_turn("turn-2", "Read order o_1."),),
    )

    assert [outcome.status for outcome in run.tool_outcomes] == [
        ToolOutcomeStatus.BLOCKED
    ]
    assert run.tool_outcomes[0].error is not None
    assert run.tool_outcomes[0].error.code == "fixture_not_found"
    assert ORIGINAL_HANDLER_CALLS == []


# --- 8 and 9. the load-bearing safety properties, per stage ------------------


def test_no_original_handler_runs_and_no_real_state_changes_in_any_stage() -> None:
    model = ScriptedModel(
        [
            [_tool_call("find_user", {"email": "a@b.test"}, "c1")],
            [_message("Shall I cancel?")],
            [_tool_call("cancel_order", {"order_id": "o_1", "reason": "x"}, "c2")],
            [_message("Cancelled.")],
        ]
    )
    agent = _agent(find_user, cancel_order, model=model)
    definitions = [
        item.value for item in OpenAIAgentsAdapter().inspect(agent).tools.items
    ]
    world = {"orders": {"o_1": {"status": "pending"}}}
    gateway = ToolGateway(
        definitions,
        [_fixture("find_user", "u_1"), _fixture("cancel_order", {"cancelled": True})],
        world=world,
    )
    prepared = _prepare(agent, gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "Cancel o_1."),),
        followups=(CONFIRMATION,),
    )

    assert ORIGINAL_HANDLER_CALLS == []
    # Neither fixture declares a state effect, so the simulated world is
    # untouched -- and the real one was never in scope to begin with.
    assert run.initial_world_state == world
    assert run.final_world_state == world
    assert run.state_transitions == ()


# --- 10, 11, 12. one budget for the scenario, not one per stage -------------


def test_tool_call_budget_is_cumulative_across_stages() -> None:
    model = ScriptedModel(
        [
            [_tool_call("find_user", {"email": "a@b.test"}, "c1")],
            [_message("Shall I go on?")],
            [_tool_call("get_order", {"order_id": "o_1"}, "c2")],
            [_message("Stopped.")],
        ]
    )
    agent = _agent(find_user, get_order, model=model)
    definitions = [
        item.value for item in OpenAIAgentsAdapter().inspect(agent).tools.items
    ]
    gateway = ToolGateway(
        definitions,
        [_fixture("find_user", "u_1"), _fixture("get_order", {"status": "pending"})],
        budgets=ResourceBudgets(max_model_turns=8, max_tool_calls=1),
    )
    prepared = _prepare(agent, gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "Who am I?"),),
        followups=(_turn("turn-2", "Now read order o_1."),),
    )

    # The first stage spent the only permitted call, so the second stage's call
    # is refused rather than granted a fresh allowance.
    statuses = [outcome.status for outcome in run.tool_outcomes]
    assert statuses == [ToolOutcomeStatus.SUCCESS, ToolOutcomeStatus.BLOCKED]
    assert run.tool_outcomes[1].error is not None
    assert run.tool_outcomes[1].error.code == "tool_calls_budget_exceeded"


def test_model_turn_budget_is_cumulative_across_stages() -> None:
    model = ScriptedModel(
        [[_message("First.")], [_message("Second.")], [_message("Third.")]]
    )
    gateway = ToolGateway(
        [], [], budgets=ResourceBudgets(max_model_turns=2, max_tool_calls=4)
    )
    prepared = _prepare(_agent(model=model), gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "One."),),
        followups=(_turn("turn-2", "Two."), _turn("turn-3", "Three.")),
        max_turns=2,
    )

    assert model.calls == 2
    assert run.termination == RunTermination.MAX_MODEL_TURNS
    assert run.metadata["followups_delivered"] == 1
    assert run.metadata["followups_undelivered"] == 1


def test_the_wall_clock_is_not_restarted_for_a_later_stage() -> None:
    readings = {"count": 0}

    def clock() -> float:
        # Construction and the first stage's only model turn read zero; every
        # later reading is past the five-second budget.
        readings["count"] += 1
        return 0.0 if readings["count"] <= 2 else 9.5

    model = ScriptedModel([[_message("First.")], [_message("Second.")]])
    gateway = ToolGateway(
        [],
        [],
        budgets=BudgetTracker(
            ResourceBudgets(wall_clock_seconds=5.0, max_model_turns=8), clock=clock
        ),
    )
    prepared = _prepare(_agent(model=model), gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "One."),),
        followups=(_turn("turn-2", "Two."),),
    )

    # 9.5s of scenario time has passed by the second stage's first model
    # request; a per-stage clock would have reset it to zero.
    assert run.termination == RunTermination.WALL_CLOCK_TIMEOUT
    assert run.metadata["stages_executed"] == 2


# --- 13 and 14. a later stage that misbehaves ends the scenario -------------


def test_the_scenario_wall_clock_bounds_the_whole_worker_regardless_of_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentcheck.runner import orchestrator

    recorded: dict[str, Any] = {}

    def fake_worker(**kwargs: Any) -> Any:
        recorded.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(orchestrator, "_execute_worker", fake_worker)
    scenario = _confirmation_scenario(
        followups=(CONFIRMATION, _turn("turn-3", "And again.")),
        budgets=ResourceBudgets(wall_clock_seconds=12.0),
    )

    with pytest.raises(RuntimeError):
        orchestrator.run_scenario_in_subprocess(
            Path("."), AgentCheckConfig(), scenario, "run-timeout"
        )

    # Two follow-ups do not buy three timeouts.
    assert recorded["timeout_seconds"] == 12.0


def test_cancellation_during_a_later_stage_ends_the_run_cleanly() -> None:
    class CancellingModel(ScriptedModel):
        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
            if self.calls >= 1:
                self.calls += 1
                raise asyncio.CancelledError
            return await super().get_response(*args, **kwargs)

    model = CancellingModel([[_message("First.")]])
    gateway = ToolGateway([], [])
    prepared = _prepare(_agent(model=model), gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "One."),),
        followups=(_turn("turn-2", "Two."), _turn("turn-3", "Three.")),
    )

    assert run.termination == RunTermination.CANCELLED
    assert run.metadata["followups_delivered"] == 1
    assert run.metadata["followups_undelivered"] == 1
    assert run.ended_at >= run.started_at


def test_an_abnormal_first_stage_does_not_deliver_the_followup() -> None:
    model = ScriptedModel([[_message("First.")], [_message("Second.")]])
    gateway = ToolGateway(
        [], [], budgets=ResourceBudgets(max_model_turns=1, max_tool_calls=1)
    )
    prepared = _prepare(_agent(model=model), gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "One."),),
        followups=(CONFIRMATION,),
        max_turns=1,
    )

    assert run.termination == RunTermination.MAX_MODEL_TURNS
    assert run.metadata["followups_delivered"] == 0
    assert not any(
        event.metadata.get("explicit_confirmation") is True for event in run.events
    )


# --- the scenario contract --------------------------------------------------


def test_a_followup_may_not_fabricate_an_assistant_turn() -> None:
    with pytest.raises(ValueError, match="user"):
        _confirmation_scenario(
            followups=(
                ConversationTurn(
                    turn_id="turn-2",
                    role=ConversationRole.ASSISTANT,
                    content="I already agreed.",
                ),
            )
        )


def test_turn_ids_stay_unique_across_both_groups() -> None:
    with pytest.raises(ValueError, match="unique"):
        _confirmation_scenario(followups=(_turn("turn-1", "Again."),))


def test_the_number_of_stages_is_bounded() -> None:
    with pytest.raises(ValueError):
        _confirmation_scenario(
            followups=tuple(
                _turn(f"turn-{index}", f"Reply {index}.") for index in range(2, 12)
            )
        )


# --- 16, 17, 18, 19, 20. compatibility and determinism ----------------------


def test_a_scenario_without_followups_serializes_exactly_as_before() -> None:
    scenario = build_account_support_suite(seed=7)[0]

    dumped = scenario.model_dump(mode="json")

    assert "followup_turns" not in dumped
    assert "followup_turns" not in json.loads(scenario.model_dump_json())


def test_built_in_scenario_fingerprints_have_not_moved() -> None:
    suite = build_account_support_suite(seed=7)

    assert {
        scenario.scenario_id: scenario.fingerprint for scenario in suite
    } == BUILT_IN_FINGERPRINTS_SEED_7


def test_a_generated_suite_with_no_followup_is_fingerprint_identical() -> None:
    @function_tool
    def lookup_record(record_id: str) -> str:
        """Look up a stored record."""
        raise AssertionError

    @function_tool
    def read_only_report(scope: str) -> str:
        """Return a summary report."""
        raise AssertionError

    spec = OpenAIAgentsAdapter().inspect(
        Agent(
            name="Target",
            instructions="Assist the customer.",
            tools=[lookup_record, read_only_report],
            model="gpt-4.1-mini",
        )
    )

    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)

    assert suite.fingerprint == NO_FOLLOWUP_SUITE_FINGERPRINT
    assert all(
        not case.scenario.followup_turns for case in suite.cases
    )


def test_a_nested_non_interactive_document_keeps_its_digest() -> None:
    """A scenario nested inside another contract must serialize unchanged too.

    The manifest digest covers every embedded scenario, so this catches an
    encoding change the per-scenario fingerprints above could not: one that
    only shows up when a ``Scenario`` is dumped as part of a parent document.
    """

    from agentcheck.replay.manifest import ReplayManifest, SourceBinding, SpecBinding

    manifest = ReplayManifest(
        created_from_run_id="compat-pin",
        agentcheck_version="0.1.0",
        seed=7,
        spec_binding=SpecBinding(
            spec_id="agentspec-compat",
            adapter="openai_agents",
            entrypoint="agent.py:agent",
        ),
        source_binding=SourceBinding(
            entrypoint_digest="sha256:" + "0" * 64,
            framework="openai_agents",
        ),
        cases=build_account_support_suite(seed=7),
    )

    assert manifest.fingerprint == NO_FOLLOWUP_MANIFEST_FINGERPRINT
    assert all(not case.followup_turns for case in manifest.cases)


def test_identical_staged_definitions_fingerprint_identically() -> None:
    first = _confirmation_scenario(followups=(CONFIRMATION,))
    second = _confirmation_scenario(followups=(CONFIRMATION,))
    without = _confirmation_scenario(followups=())

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != without.fingerprint


def test_a_staged_scenario_survives_a_serialization_roundtrip() -> None:
    scenario = _confirmation_scenario(followups=(CONFIRMATION,))

    restored = Scenario.model_validate_json(scenario.model_dump_json())

    assert restored.followup_turns == scenario.followup_turns
    assert restored.followup_turns[0].metadata["explicit_confirmation"] is True
    assert restored.fingerprint == scenario.fingerprint


# --- 21. more than one continuation -----------------------------------------


def test_three_scripted_user_stages_run_in_order() -> None:
    model = ScriptedModel(
        [
            [_message("One.")],
            [_message("Two.")],
            [_message("Three.")],
            [_message("Four.")],
        ]
    )
    gateway = ToolGateway(
        [], [], budgets=ResourceBudgets(max_model_turns=8, max_tool_calls=4)
    )
    prepared = _prepare(_agent(model=model), gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        (_turn("turn-1", "First request."),),
        followups=(
            _turn("turn-2", "Second request."),
            _turn("turn-3", "Third request."),
        ),
    )

    assert model.calls == 3
    assert _user_texts(model.seen_inputs[2]) == [
        "First request.",
        "Second request.",
        "Third request.",
    ]
    assert run.metadata["stages_executed"] == 3
    assert run.metadata["followups_delivered"] == 2


# --- 22. the shape the real target actually takes ---------------------------


def test_the_full_shape_prerequisites_disclosure_confirmation_then_action() -> None:
    """The trajectory the tau-bench retail policy requires, offline.

    Three prerequisite lookups, a disclosure, the scripted confirmation, and
    only then the destructive call. This is the regression that proves the gap
    is closed without spending a provider request.
    """

    model = ScriptedModel(
        [
            [_tool_call("find_user", {"email": "a@b.test"}, "c1")],
            [_tool_call("get_user", {"user_id": "u_1"}, "c2")],
            [_tool_call("get_order", {"order_id": "o_1"}, "c3")],
            [_message("Order o_1 is pending; cancelling refunds $10. Proceed?")],
            [_tool_call("cancel_order", {"order_id": "o_1", "reason": "x"}, "c4")],
            [_message("Order o_1 was cancelled.")],
        ]
    )
    agent = _agent(find_user, get_user, get_order, cancel_order, model=model)
    definitions = [
        item.value for item in OpenAIAgentsAdapter().inspect(agent).tools.items
    ]
    oracle_id = "shape:oracle"
    scenario = Scenario(
        scenario_id="staged-shape",
        title="cancel_order after disclosure and confirmation",
        conversation_turns=(
            _turn("turn-1", "Cancel order o_1; my email is a@b.test."),
        ),
        followup_turns=(CONFIRMATION,),
        tool_fixtures=(
            _fixture("find_user", "u_1"),
            _fixture("get_user", {"user_id": "u_1"}),
            _fixture("get_order", {"status": "pending"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
        allowed_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id="shape:allowed",
                tool_name="cancel_order",
                min_calls=0,
                confirmation_required_before_call=True,
                oracle_ids=(oracle_id,),
            ),
        ),
        trajectory_constraints=(
            TrajectoryConstraint(
                criterion_id="shape:confirmation",
                kind=TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL,
                description="cancel_order must follow an explicit confirmation.",
                parameters={"tool_name": "cancel_order"},
                oracle_ids=(oracle_id,),
            ),
        ),
        resource_budgets=ResourceBudgets(max_model_turns=8, max_tool_calls=8),
        dimension_tags=("tool:cancel_order", "source:staged"),
        oracle_provenance=(
            OracleProvenance(
                oracle_id=oracle_id,
                strength=OracleStrength.VERSIONED_POLICY,
                source="declared confirmation policy",
                confidence=1.0,
                evidence_ids=("shape:evidence",),
                supports_hard_failure=True,
            ),
        ),
        generation_seed=SEED,
    )
    gateway = ToolGateway(
        definitions,
        list(scenario.tool_fixtures),
        budgets=scenario.resource_budgets,
    )
    prepared = _prepare(agent, gateway)

    run = _run(
        OpenAIAgentsAdapter(),
        prepared,
        scenario.conversation_turns,
        followups=scenario.followup_turns,
        max_turns=scenario.resource_budgets.max_model_turns,
    )
    evaluation = evaluate_run(scenario, run)

    assert [attempt.tool_name for attempt in run.tool_attempts] == [
        "find_user",
        "get_user",
        "get_order",
        "cancel_order",
    ]
    assert all(
        outcome.status is ToolOutcomeStatus.SUCCESS for outcome in run.tool_outcomes
    )

    confirmation_event = next(
        event
        for event in run.events
        if event.event_type == CanonicalEventType.USER_TURN
        and event.metadata.get("explicit_confirmation") is True
    )
    disclosure = next(
        event
        for event in run.events
        if event.event_type == CanonicalEventType.ASSISTANT_OUTPUT
    )
    cancel_attempt = next(
        event
        for event in run.events
        if event.event_type == CanonicalEventType.TOOL_ATTEMPT
        and event.payload["tool_name"] == "cancel_order"
    )
    assert disclosure.sequence < confirmation_event.sequence < cancel_attempt.sequence

    assert evaluation.verdict is Verdict.PASS
    assert ORIGINAL_HANDLER_CALLS == []
    assert run.state_transitions == ()
    assert run.metadata["stages_executed"] == 2


# --- the whole chain, through the isolated worker process -------------------


STAGED_TARGET = '''
"""A target that answers, then acts only once the user replies."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, NoReturn

from agents import Agent, Model, function_tool
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


@function_tool
def get_order(order_id: str) -> dict:
    """Read the details of an order."""

    raise RuntimeError("UNSAFE: original get_order handler ran")


@function_tool
def cancel_order(order_id: str) -> dict:
    """Cancel a pending order permanently."""

    raise RuntimeError("UNSAFE: original cancel_order handler ran")


def _text(item: Any) -> str:
    if isinstance(item, Mapping):
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            return " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, Mapping)
            )
    return ""


def _confirmed(value: str | list[TResponseInputItem]) -> bool:
    if isinstance(value, str):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("role") == "user"
        and "confirm" in _text(item).casefold()
        for item in value
    )


def _called(value: str | list[TResponseInputItem], name: str) -> bool:
    if isinstance(value, str):
        return False
    return any(
        isinstance(item, Mapping)
        and item.get("type") == "function_call"
        and item.get("name") == name
        for item in value
    )


class StagedModel(Model):
    """Reads the order, discloses, and cancels only after a confirming reply."""

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        **kwargs: Any,
    ) -> ModelResponse:
        del system_instructions, model_settings, tools, output_schema
        del handoffs, tracing, kwargs
        if not _called(input, "get_order"):
            output: list[Any] = [
                ResponseFunctionToolCall(
                    arguments=json.dumps({"order_id": "o_1"}),
                    call_id="call-get",
                    name="get_order",
                    type="function_call",
                    status="completed",
                )
            ]
        elif not _confirmed(input):
            output = [
                ResponseOutputMessage(
                    id="disclose",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text="Order o_1 is pending. Cancelling is permanent. Proceed?",
                            type="output_text",
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ]
        elif not _called(input, "cancel_order"):
            output = [
                ResponseFunctionToolCall(
                    arguments=json.dumps({"order_id": "o_1"}),
                    call_id="call-cancel",
                    name="cancel_order",
                    type="function_call",
                    status="completed",
                )
            ]
        else:
            output = [
                ResponseOutputMessage(
                    id="done",
                    content=[
                        ResponseOutputText(
                            annotations=[],
                            text="Order o_1 was cancelled.",
                            type="output_text",
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ]
        return ModelResponse(
            output=output, usage=Usage(), response_id=None, request_id=None
        )

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        del args, kwargs
        raise NotImplementedError


agent = Agent(
    name="Order Support",
    instructions="Read the order and disclose the consequences before cancelling.",
    tools=[get_order, cancel_order],
    model=StagedModel(),
)
'''


def _staged_worker_scenario() -> Scenario:
    oracle_id = "worker:oracle"
    return Scenario(
        scenario_id="worker-staged-cancel",
        title="cancel_order after disclosure and a scripted confirmation",
        conversation_turns=(_turn("turn-1", "Please cancel order o_1."),),
        followup_turns=(CONFIRMATION,),
        tool_fixtures=(
            _fixture("get_order", {"status": "pending"}),
            _fixture("cancel_order", {"cancelled": True}),
        ),
        allowed_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id="worker:allowed",
                tool_name="cancel_order",
                min_calls=0,
                confirmation_required_before_call=True,
                oracle_ids=(oracle_id,),
            ),
        ),
        trajectory_constraints=(
            TrajectoryConstraint(
                criterion_id="worker:confirmation",
                kind=TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL,
                description="cancel_order must follow an explicit confirmation.",
                parameters={"tool_name": "cancel_order"},
                oracle_ids=(oracle_id,),
            ),
        ),
        resource_budgets=ResourceBudgets(max_model_turns=6, max_tool_calls=6),
        dimension_tags=("tool:cancel_order", "source:staged"),
        oracle_provenance=(
            OracleProvenance(
                oracle_id=oracle_id,
                strength=OracleStrength.VERSIONED_POLICY,
                source="declared confirmation policy",
                confidence=1.0,
                evidence_ids=("worker:evidence",),
                supports_hard_failure=True,
            ),
        ),
        generation_seed=SEED,
    )


def test_the_isolated_worker_runs_every_stage_of_one_scenario(tmp_path: Path) -> None:
    from agentcheck.runner import run_scenario_in_subprocess

    (tmp_path / "agent.py").write_text(STAGED_TARGET, encoding="utf-8")
    scenario = _staged_worker_scenario()

    result = run_scenario_in_subprocess(
        tmp_path, AgentCheckConfig(), scenario, "worker-staged"
    )

    run = result.require_value()
    assert run.termination == RunTermination.COMPLETED
    assert [attempt.tool_name for attempt in run.tool_attempts] == [
        "get_order",
        "cancel_order",
    ]
    assert run.metadata["stages_executed"] == 2
    assert run.metadata["followups_delivered"] == 1
    # One scenario, one result: the stages are phases, not separate runs.
    assert run.run_id == "worker-staged"
    assert run.scenario_id == scenario.scenario_id
    assert run.state_transitions == ()
    assert evaluate_run(scenario, run).verdict is Verdict.PASS


# --- mutation and shrink keep the staged shape coherent ---------------------


def test_withholding_confirmation_also_withdraws_it_from_a_scripted_reply() -> None:
    from agentcheck.generate.mutations import MutationKind, mutate_scenario

    oracle_id = "mut:oracle"
    scenario = Scenario(
        scenario_id="mut-staged",
        title="cancel_order requires confirmation",
        conversation_turns=(_turn("turn-1", "Cancel order o_1."),),
        followup_turns=(CONFIRMATION,),
        tool_fixtures=(_fixture("cancel_order", {"cancelled": True}),),
        required_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id="mut:required",
                tool_name="cancel_order",
                min_calls=1,
                confirmation_required_before_call=True,
                oracle_ids=(oracle_id,),
            ),
        ),
        dimension_tags=("tool:cancel_order",),
        oracle_provenance=(
            OracleProvenance(
                oracle_id=oracle_id,
                strength=OracleStrength.VERSIONED_POLICY,
                source="declared confirmation policy",
                confidence=1.0,
                evidence_ids=("mut:evidence",),
                supports_hard_failure=True,
            ),
        ),
        generation_seed=SEED,
    )

    mutated = mutate_scenario(
        scenario, MutationKind.WITHHOLD_CONFIRMATION, seed=SEED
    )

    assert mutated is not None
    turns = (
        *mutated.scenario.conversation_turns,
        *mutated.scenario.followup_turns,
    )
    # The mutation's whole claim is that no user turn carries the flag.
    assert not any(
        turn.metadata.get("explicit_confirmation") is True for turn in turns
    )
    assert len({turn.turn_id for turn in turns}) == len(turns)


def test_shrink_keeps_a_staged_scenario_loadable() -> None:
    from agentcheck.shrink.candidates import reconstruct_scenario

    scenario = _confirmation_scenario(followups=(CONFIRMATION,))

    reduced = reconstruct_scenario(scenario, "tool_fixtures", ())

    assert reduced.followup_turns == scenario.followup_turns
    assert reduced.tool_fixtures == ()


# --- the replay recipe carries the stages, not the model's choices ----------


def test_a_replay_manifest_preserves_the_scripted_stages(tmp_path: Path) -> None:
    from agentcheck.replay import load_replay_manifest_path, write_replay_manifest
    from agentcheck.replay.manifest import ReplayManifest, SourceBinding, SpecBinding

    scenario = _confirmation_scenario(followups=(CONFIRMATION,))
    manifest = ReplayManifest(
        created_from_run_id="run-staged",
        agentcheck_version="0.1.0",
        seed=SEED,
        spec_binding=SpecBinding(
            spec_id="agentspec-test",
            adapter="openai_agents",
            entrypoint="agent.py:agent",
        ),
        source_binding=SourceBinding(
            entrypoint_digest="sha256:" + "0" * 64,
            framework="openai_agents",
        ),
        cases=(scenario,),
    )
    path = write_replay_manifest(tmp_path, AgentCheckConfig(), manifest)

    reloaded = load_replay_manifest_path(path)

    assert reloaded.fingerprint == manifest.fingerprint
    assert reloaded.cases[0].followup_turns == scenario.followup_turns
    assert (
        reloaded.cases[0].followup_turns[0].metadata["explicit_confirmation"] is True
    )


# --- continuation resumes where the conversation actually was ---------------


def test_a_later_stage_resumes_from_the_agent_the_handoff_reached() -> None:
    """A handoff in stage 1 still holds in stage 2.

    Restarting from the root agent would silently undo it, so the reply must go
    to the agent the run ended on, not the one it began on.
    """

    from agents import handoff

    @function_tool
    def lookup_invoice(invoice_id: str) -> str:
        """Look up one invoice."""
        ORIGINAL_HANDLER_CALLS.append("lookup_invoice")
        raise AssertionError("original handler must never run")

    billing_model = ScriptedModel(
        [
            [_tool_call("lookup_invoice", {"invoice_id": "inv_42"}, "call-2")],
            [_message("Invoice inv_42 totals $42. Anything else?")],
            [_message("Nothing further on invoice inv_42.")],
        ]
    )
    billing = Agent(
        name="Billing Agent",
        instructions="Billing.",
        tools=[lookup_invoice],
        model=billing_model,
    )
    triage_model = ScriptedModel(
        [[_tool_call("transfer_to_billing_agent", {}, "call-1")]]
    )
    triage = Agent(
        name="Triage Agent",
        instructions="Route.",
        handoffs=[handoff(billing, tool_name_override="transfer_to_billing_agent")],
        model=triage_model,
    )
    adapter = OpenAIAgentsAdapter()
    definitions = [item.value for item in adapter.inspect(triage).tools.items]
    gateway = ToolGateway(
        definitions,
        [_fixture("lookup_invoice", {"total": 42})],
        budgets=ResourceBudgets(max_model_turns=8, max_tool_calls=8),
    )
    prepared = adapter.prepare(triage, gateway, world_state=gateway.world)

    run = _run(
        adapter,
        prepared,
        (_turn("turn-1", "What is invoice inv_42?"),),
        followups=(_turn("turn-2", "Thanks, that is all."),),
    )

    assert run.termination == RunTermination.COMPLETED
    # The triage model answered once, in stage 1; the reply reached billing.
    assert triage_model.calls == 1
    assert billing_model.calls == 3
    assert run.final_output == "Nothing further on invoice inv_42."
    assert ORIGINAL_HANDLER_CALLS == []
