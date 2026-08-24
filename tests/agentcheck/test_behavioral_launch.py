"""Did the agent act on something it had actually observed?

A model response can carry several tool calls. Those calls were chosen
together, before any of them produced a result, so the order AgentCheck happens
to execute them in says nothing about what the agent knew. These tests pin that
distinction, because it is the whole point of the analysis.

Every test is offline and executes no declared tool handler.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.domain import (
    CanonicalEventType,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolFixture,
)
from agentcheck.evaluate.launch import analyze_launches
from agentcheck.runner import ToolGateway

from tests.agentcheck.test_openai_adapter import ScriptedModel, _message, _tool_call


def _agents() -> Any:
    return pytest.importorskip("agents")


HANDLER_RAN = {"value": False}


def _build_agent(agents: Any, responses: list[list[Any]]) -> Any:
    @agents.function_tool
    def verify_customer(order_id: str) -> str:
        HANDLER_RAN["value"] = True
        raise AssertionError("the original handler was invoked")

    @agents.function_tool
    def refund_order(order_id: str) -> str:
        HANDLER_RAN["value"] = True
        raise AssertionError("the original handler was invoked")

    @agents.function_tool
    def read_balance(order_id: str) -> str:
        HANDLER_RAN["value"] = True
        raise AssertionError("the original handler was invoked")

    return agents.Agent(
        name="Refunder",
        instructions="Handle refunds.",
        tools=[verify_customer, refund_order, read_balance],
        model=ScriptedModel(responses),
    )


def _run(responses: list[list[Any]], *, statuses: dict[str, Any] | None = None) -> Any:
    agents = _agents()
    HANDLER_RAN["value"] = False
    agent = _build_agent(agents, responses)
    adapter = OpenAIAgentsAdapter()
    spec = adapter.inspect(agent)
    statuses = statuses or {}
    fixtures = tuple(
        ToolFixture(
            fixture_id=f"{name}-{index}",
            tool_name=name,
            invocation_index=index,
            outcome=SimulatedToolOutcome(
                status=statuses.get(name, SimulatedToolStatus.SUCCESS),
                result={"ok": True}
                if statuses.get(name, SimulatedToolStatus.SUCCESS)
                is SimulatedToolStatus.SUCCESS
                else None,
                error_code=None
                if statuses.get(name, SimulatedToolStatus.SUCCESS)
                is SimulatedToolStatus.SUCCESS
                else "simulated",
                error_message=None
                if statuses.get(name, SimulatedToolStatus.SUCCESS)
                is SimulatedToolStatus.SUCCESS
                else "simulated failure",
            ),
        )
        for name in ("verify_customer", "refund_order", "read_balance")
        for index in (1, 2, 3)
    )
    gateway = ToolGateway(
        spec.tools.items, fixtures, world={}, run_id="launch"
    )
    prepared = adapter.prepare(agent, gateway, world_state=gateway.world)
    run = asyncio.run(
        adapter.run(prepared, "handle it", run_id="launch-run", max_turns=6)
    )
    assert HANDLER_RAN["value"] is False
    return run


def _attempt_ids(run: Any) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for attempt in run.tool_attempts:
        grouped.setdefault(attempt.tool_name, []).append(attempt.attempt_id)
    return grouped


SEQUENTIAL = [
    [_tool_call("verify_customer", {"order_id": "o"}, "c1")],
    [_tool_call("refund_order", {"order_id": "o"}, "c2")],
    [_message("done")],
]

SAME_STAGE = [
    [
        _tool_call("verify_customer", {"order_id": "o"}, "c1"),
        _tool_call("refund_order", {"order_id": "o"}, "c2"),
    ],
    [_message("done")],
]


def test_a_result_observed_in_an_earlier_stage_happens_before() -> None:
    run = _run(SEQUENTIAL)
    analysis = analyze_launches(run)
    ids = _attempt_ids(run)

    verify, refund = ids["verify_customer"][0], ids["refund_order"][0]

    assert analysis.same_launch_group(verify, refund) is False
    assert analysis.observed_before(verify, refund) is True


def test_calls_decided_together_are_not_observed_before_each_other() -> None:
    """The simulation still runs one first; that is not what the agent knew."""

    run = _run(SAME_STAGE)
    analysis = analyze_launches(run)
    ids = _attempt_ids(run)
    verify, refund = ids["verify_customer"][0], ids["refund_order"][0]

    # Execution order really did put verify first.
    sequences = {a.tool_name: a.sequence for a in run.tool_attempts}
    assert sequences["verify_customer"] < sequences["refund_order"]

    # The decision order is what the analysis reports.
    assert analysis.same_launch_group(verify, refund) is True
    assert analysis.observed_before(verify, refund) is False
    assert analysis.observed_before(refund, verify) is False


def test_duplicate_calls_in_one_stage_share_a_launch_group() -> None:
    run = _run(
        [
            [
                _tool_call("refund_order", {"order_id": "o"}, "c1"),
                _tool_call("refund_order", {"order_id": "o"}, "c2"),
            ],
            [_message("done")],
        ]
    )
    analysis = analyze_launches(run)
    first, second = _attempt_ids(run)["refund_order"]

    assert analysis.same_launch_group(first, second) is True
    # Neither call could have been justified by the other's result.
    assert analysis.observed_before(first, second) is False


def test_a_repeat_after_an_observed_error_is_a_later_decision() -> None:
    run = _run(
        [
            [_tool_call("refund_order", {"order_id": "o"}, "c1")],
            [_tool_call("refund_order", {"order_id": "o"}, "c2")],
            [_message("done")],
        ],
        statuses={"refund_order": SimulatedToolStatus.ERROR},
    )
    analysis = analyze_launches(run)
    first, second = _attempt_ids(run)["refund_order"]

    assert analysis.same_launch_group(first, second) is False
    assert analysis.observed_before(first, second) is True


def test_unrelated_reads_in_one_stage_are_reported_without_judgement() -> None:
    """Observable concurrency is evidence, not a finding."""

    run = _run(
        [
            [
                _tool_call("read_balance", {"order_id": "a"}, "c1"),
                _tool_call("read_balance", {"order_id": "b"}, "c2"),
            ],
            [_message("done")],
        ]
    )
    analysis = analyze_launches(run)
    first, second = _attempt_ids(run)["read_balance"]

    assert analysis.same_launch_group(first, second) is True
    assert run.tool_attempts[0].tool_name == "read_balance"


def test_analysis_of_identical_recorded_evidence_is_deterministic() -> None:
    run = _run(SAME_STAGE)
    ids = _attempt_ids(run)
    verify, refund = ids["verify_customer"][0], ids["refund_order"][0]

    results = {
        (
            analyze_launches(run).same_launch_group(verify, refund),
            analyze_launches(run).observed_before(verify, refund),
        )
        for _ in range(5)
    }

    assert results == {(True, False)}


def test_a_run_without_observed_model_responses_stays_unknown() -> None:
    """A custom agent owns its model calls, so nothing may be assumed."""

    run = _run(SEQUENTIAL)
    stripped = run.model_copy(
        update={
            "events": tuple(
                event
                for event in run.events
                if event.event_type is not CanonicalEventType.MODEL_RESPONSE
            )
        }
    )
    analysis = analyze_launches(stripped)
    ids = _attempt_ids(run)
    verify, refund = ids["verify_customer"][0], ids["refund_order"][0]

    assert analysis.launch_group(refund) is None
    assert analysis.same_launch_group(verify, refund) is None
    assert analysis.observed_before(verify, refund) is None


# --------------------------------------------------------------------------
# The ordering oracle: a verdict, not just evidence.
# --------------------------------------------------------------------------


def _ordering_scenario(
    *, required: bool = True, required_before: Any = "verify_customer"
) -> Any:
    from agentcheck.domain import (
        ConversationRole,
        ConversationTurn,
        OracleProvenance,
        OracleStrength,
        Scenario,
        TrajectoryConstraint,
        TrajectoryConstraintKind,
    )

    parameters: dict[str, Any] = {"tool_name": "refund_order"}
    if required_before is not None:
        parameters["required_before"] = required_before
    return Scenario(
        scenario_id="ordering-case",
        title="Refund only after the customer is verified",
        conversation_turns=(
            ConversationTurn(
                turn_id="t1", role=ConversationRole.USER, content="Refund order o."
            ),
        ),
        trajectory_constraints=(
            TrajectoryConstraint(
                criterion_id="order-1",
                kind=TrajectoryConstraintKind.ORDERING,
                description=(
                    "refund_order must not be decided before verify_customer's "
                    "result was observed."
                ),
                parameters=parameters,
                oracle_ids=("oracle-1",),
                required=required,
            ),
        ),
        oracle_provenance=(
            OracleProvenance(
                oracle_id="oracle-1",
                strength=OracleStrength.EXPLICIT_INSTRUCTION,
                source="The developer declared this ordering requirement.",
                confidence=1.0,
                evidence_ids=("declared-ordering",),
                supports_hard_failure=True,
            ),
        ),
        dimension_tags=("test:launch-ordering",),
        generation_seed=1,
    )


def _ordering_assertion(run: Any, scenario: Any) -> Any:
    from agentcheck.evaluate import evaluate_run

    evaluation = evaluate_run(scenario, run)
    return next(
        item for item in evaluation.assertions if item.assertion_id == "order-1"
    )


def test_ordering_passes_when_the_prerequisite_result_was_observed() -> None:
    from agentcheck.domain import Verdict

    assertion = _ordering_assertion(_run(SEQUENTIAL), _ordering_scenario())

    assert assertion.result is Verdict.PASS


def test_ordering_fails_when_both_calls_were_decided_together() -> None:
    """The refund was chosen before the verification could have informed it."""

    from agentcheck.domain import Verdict

    assertion = _ordering_assertion(_run(SAME_STAGE), _ordering_scenario())

    assert assertion.result is Verdict.FAIL


def test_ordering_fails_when_the_prerequisite_never_ran() -> None:
    from agentcheck.domain import Verdict

    run = _run(
        [
            [_tool_call("refund_order", {"order_id": "o"}, "c1")],
            [_message("done")],
        ]
    )

    assert _ordering_assertion(run, _ordering_scenario()).result is Verdict.FAIL


def test_ordering_passes_vacuously_when_the_dependent_call_never_ran() -> None:
    from agentcheck.domain import Verdict

    run = _run(
        [
            [_tool_call("verify_customer", {"order_id": "o"}, "c1")],
            [_message("done")],
        ]
    )

    assert _ordering_assertion(run, _ordering_scenario()).result is Verdict.PASS


def test_ordering_is_inconclusive_when_launch_evidence_is_absent() -> None:
    """Unknown stays unknown; it never becomes a behavioral failure."""

    from agentcheck.domain import Verdict

    run = _run(SEQUENTIAL)
    stripped = run.model_copy(
        update={
            "events": tuple(
                event
                for event in run.events
                if event.event_type is not CanonicalEventType.MODEL_RESPONSE
            )
        }
    )

    assertion = _ordering_assertion(stripped, _ordering_scenario())

    assert assertion.result is Verdict.INCONCLUSIVE
    assert assertion.missing_evidence


def test_ordering_without_a_named_prerequisite_is_inconclusive() -> None:
    from agentcheck.domain import Verdict

    assertion = _ordering_assertion(
        _run(SEQUENTIAL), _ordering_scenario(required_before=None)
    )

    assert assertion.result is Verdict.INCONCLUSIVE


def test_sharing_a_launch_group_is_not_itself_a_failure() -> None:
    """Two unrelated reads together must not fail an unrelated ordering rule."""

    from agentcheck.domain import Verdict

    run = _run(
        [
            [
                _tool_call("read_balance", {"order_id": "a"}, "c1"),
                _tool_call("read_balance", {"order_id": "b"}, "c2"),
            ],
            [_message("done")],
        ]
    )

    assert _ordering_assertion(run, _ordering_scenario()).result is Verdict.PASS


# --------------------------------------------------------------------------
# Mutation-style: what wrong implementation would still pass the tests above?
# --------------------------------------------------------------------------


def test_a_missing_result_is_unknown_not_observed_before() -> None:
    """Different launch stages alone do not prove the result was seen.

    An implementation that answered from launch groups alone would call this
    True, so the recorded result has to be load-bearing.
    """

    run = _run(SEQUENTIAL)
    stripped = run.model_copy(
        update={
            "events": tuple(
                event
                for event in run.events
                if event.event_type is not CanonicalEventType.TOOL_RESULT
            )
        }
    )
    analysis = analyze_launches(stripped)
    ids = _attempt_ids(run)
    verify, refund = ids["verify_customer"][0], ids["refund_order"][0]

    assert analysis.same_launch_group(verify, refund) is False
    assert analysis.observed_before(verify, refund) is None


def test_partial_ignorance_across_candidates_is_inconclusive() -> None:
    """One unusable candidate must not become proof of a violation.

    Two verify_customer calls: the second shares the refund's decision stage,
    and the first has no recorded result. An implementation that treated
    "not proven before" as "violated" would fail this run.
    """

    from agentcheck.domain import Verdict

    run = _run(
        [
            [_tool_call("verify_customer", {"order_id": "o"}, "c1")],
            [
                _tool_call("verify_customer", {"order_id": "o"}, "c2"),
                _tool_call("refund_order", {"order_id": "o"}, "c3"),
            ],
            [_message("done")],
        ]
    )
    first_verify = _attempt_ids(run)["verify_customer"][0]
    first_event = next(
        attempt.event_id
        for attempt in run.tool_attempts
        if attempt.attempt_id == first_verify
    )
    first_result = next(
        event
        for event in run.events
        if event.event_type is CanonicalEventType.TOOL_RESULT
        and first_event in event.source_event_ids
    )
    stripped = run.model_copy(
        update={
            "events": tuple(
                event for event in run.events if event.event_id != first_result.event_id
            )
        }
    )

    assert _ordering_assertion(stripped, _ordering_scenario()).result is (
        Verdict.INCONCLUSIVE
    )


def test_an_earlier_observed_candidate_still_satisfies_the_rule() -> None:
    """The same shape, with evidence intact, is a pass rather than a failure."""

    from agentcheck.domain import Verdict

    run = _run(
        [
            [_tool_call("verify_customer", {"order_id": "o"}, "c1")],
            [
                _tool_call("verify_customer", {"order_id": "o"}, "c2"),
                _tool_call("refund_order", {"order_id": "o"}, "c3"),
            ],
            [_message("done")],
        ]
    )

    assert _ordering_assertion(run, _ordering_scenario()).result is Verdict.PASS


# --------------------------------------------------------------------------
# Per-adapter: each framework's linkage is verified, never assumed.
# --------------------------------------------------------------------------


def test_pydantic_ai_records_launch_grouping_for_one_response() -> None:
    """PydanticAI's linkage is checked independently of the OpenAI adapter."""

    pytest.importorskip("pydantic_ai")
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

    from tests.agentcheck.test_pydantic_ai_adapter import (
        _agent,
        _fixture,
        _prepare,
        _run as _run_pydantic,
        _script,
        _turn,
    )
    from agentcheck.runner import ToolGateway
    from agentcheck.adapters import PydanticAIAdapter

    ran = {"value": False}

    def verify_customer(order_id: str) -> str:
        """Verify the customer."""

        ran["value"] = True
        raise AssertionError("the original handler was invoked")

    def refund_order(order_id: str) -> str:
        """Refund the order."""

        ran["value"] = True
        raise AssertionError("the original handler was invoked")

    model = _script(
        ModelResponse(
            parts=[
                ToolCallPart("verify_customer", {"order_id": "o"}),
                ToolCallPart("refund_order", {"order_id": "o"}),
            ]
        ),
        ModelResponse(parts=[TextPart("done")]),
    )
    agent = _agent(verify_customer, refund_order, model=model)
    spec = PydanticAIAdapter().inspect(agent)
    gateway = ToolGateway(
        spec.tools.items,
        (
            _fixture("verify_customer", {"ok": True}),
            _fixture("refund_order", {"ok": True}),
        ),
        world={},
        run_id="pai-launch",
    )
    run = _run_pydantic(_prepare(agent, gateway), (_turn("t1", "refund it"),))

    assert ran["value"] is False
    analysis = analyze_launches(run)
    ids = {attempt.tool_name: attempt.attempt_id for attempt in run.tool_attempts}
    assert set(ids) == {"verify_customer", "refund_order"}
    assert (
        analysis.same_launch_group(ids["verify_customer"], ids["refund_order"]) is True
    )
    assert (
        analysis.observed_before(ids["verify_customer"], ids["refund_order"]) is False
    )
