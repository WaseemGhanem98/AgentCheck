"""0.3.0 release-readiness: the three most recent milestones composed together.

PR #45 (developer-declared risk), #46 (concurrent dispatch), and #47
(PydanticAI dependency injection) were each validated independently. This
file checks the actual release risk: a real target that exercises all three
at once, on a single PydanticAI agent with `deps_type` and a `RunContext`-
taking tool that is declared destructive, called twice in one launch group.

If any milestone silently discarded another's guarantee -- declared risk
lost across the context-aware reconstruction path, or the same-stage
duplicate oracle blind to a call that also carries dependency context --
this is where it would show up.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, ToolCallPart

from agentcheck.adapters.pydantic_ai import PydanticAIAdapter
from agentcheck.config import ToolRiskDeclaration
from agentcheck.domain import (
    ConversationRole,
    ConversationTurn,
    OracleProvenance,
    OracleStrength,
    RiskAuthority,
    Scenario,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolFixture,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
    Verdict,
)
from agentcheck.evaluate import evaluate_run
from agentcheck.runner import ToolGateway

from tests.agentcheck.test_pydantic_ai_adapter import (
    ORIGINAL_HANDLER_CALLS,
    _agent,
    _prepare,
    _run,
    _same_stage_response,
    _script,
    _text,
)


@dataclass
class Deps:
    token: str


def test_declared_risk_survives_the_context_aware_reconstruction_path() -> None:
    """A tool whose name reads as a harmless lookup, but is explicitly
    declared destructive, must keep that authority even though it also takes
    RunContext[Deps] -- neither milestone may silently override the other."""

    agent = _agent(deps_type=Deps)

    @agent.tool
    def find_record(ctx: RunContext[Deps], record_id: str) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    spec = PydanticAIAdapter().inspect(
        agent,
        declared_tool_risk={"find_record": ToolRiskDeclaration(destructive=True)},
    )
    tool = next(item.value for item in spec.tools.items if item.value.name == "find_record")
    assertion = next(item for item in spec.tool_risk.items if item.tool_name == "find_record")

    assert tool.destructive is True
    assert assertion.destructive.authority is RiskAuthority.DEVELOPER_DECLARED
    assert "ctx" not in tool.input_schema.get("properties", {})


def _same_stage_scenario() -> Scenario:
    return Scenario(
        scenario_id="di-risk-concurrency-case",
        title="find_record must not be decided twice in one stage",
        conversation_turns=(
            ConversationTurn(turn_id="t1", role=ConversationRole.USER, content="Find it."),
        ),
        trajectory_constraints=(
            TrajectoryConstraint(
                criterion_id="dup-1",
                kind=TrajectoryConstraintKind.NO_SAME_STAGE_DUPLICATE_ACTION,
                description="find_record must not be called twice with identical arguments in one stage.",
                parameters={"tool_name": "find_record"},
                oracle_ids=("oracle-dup-1",),
                required=True,
            ),
        ),
        oracle_provenance=(
            OracleProvenance(
                oracle_id="oracle-dup-1",
                strength=OracleStrength.EXPLICIT_INSTRUCTION,
                source="The developer declared this same-stage duplicate rule.",
                confidence=1.0,
                evidence_ids=("declared-no-same-stage-duplicate",),
                supports_hard_failure=True,
            ),
        ),
        dimension_tags=("test:di-risk-concurrency",),
        generation_seed=1,
    )


def test_same_stage_duplicate_oracle_fires_for_a_declared_destructive_context_aware_tool() -> None:
    """The concurrency oracle from PR #46 must still correctly fire on a
    tool whose risk was declared (not inferred) and which takes RunContext --
    composing all three recent milestones on one call."""

    agent = _agent(deps_type=Deps)

    @agent.tool
    def find_record(ctx: RunContext[Deps], record_id: str) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    agent.model = _script(_same_stage_response("find_record", {"record_id": "r1"}), _text("done"))
    gateway = ToolGateway(
        [
            item.value
            for item in PydanticAIAdapter()
            .inspect(agent, declared_tool_risk={"find_record": ToolRiskDeclaration(destructive=True)})
            .tools.items
        ],
        [
            ToolFixture(fixture_id="first", tool_name="find_record", invocation_index=1,
                        outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result={"call": 1})),
            ToolFixture(fixture_id="second", tool_name="find_record", invocation_index=2,
                        outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result={"call": 2})),
        ],
    )
    prepared = _prepare(agent, gateway)
    run = _run(prepared, "find it twice")

    assert ORIGINAL_HANDLER_CALLS == []
    assert [outcome.result for outcome in run.tool_outcomes] == [{"call": 1}, {"call": 2}]

    evaluation = evaluate_run(_same_stage_scenario(), run)
    assertion = next(item for item in evaluation.assertions if item.assertion_id == "dup-1")
    assert assertion.result is Verdict.FAIL


def test_a_single_declared_destructive_context_aware_call_does_not_false_fail() -> None:
    """Sanity check on the other side: one call, decided once, must not be
    mistaken for a same-stage duplicate merely because it is destructive and
    context-aware."""

    agent = _agent(deps_type=Deps)

    @agent.tool
    def find_record(ctx: RunContext[Deps], record_id: str) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    agent.model = _script(
        ModelResponse(parts=[ToolCallPart("find_record", {"record_id": "r1"})]), _text("done")
    )
    gateway = ToolGateway(
        [
            item.value
            for item in PydanticAIAdapter()
            .inspect(agent, declared_tool_risk={"find_record": ToolRiskDeclaration(destructive=True)})
            .tools.items
        ],
        [ToolFixture(fixture_id="f", tool_name="find_record",
                     outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result={"ok": True}))],
    )
    prepared = _prepare(agent, gateway)
    run = _run(prepared, "find it")

    assert ORIGINAL_HANDLER_CALLS == []
    evaluation = evaluate_run(_same_stage_scenario(), run)
    assertion = next(item for item in evaluation.assertions if item.assertion_id == "dup-1")
    assert assertion.result is Verdict.PASS
