"""End-to-end offline evaluation of the bundled multi-agent handoff example."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.domain import (
    CanonicalEventType,
    CanonicalRun,
    CaseEvaluation,
    ConversationRole,
    ConversationTurn,
    OracleProvenance,
    OracleStrength,
    OutputCriterion,
    OutputCriterionKind,
    ResourceBudgets,
    RunTermination,
    Scenario,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolBehaviorConstraint,
    ToolFixture,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
    Verdict,
)
from agentcheck.evaluate import evaluate_run
from agentcheck.generate.lint import lint_scenario
from agentcheck.inspect import load_target
from agentcheck.runner import ToolGateway
from agentcheck.runner.orchestrator import inspect_in_subprocess, run_scenario_in_subprocess
from agentcheck.config import load_config
from agentcheck.shrink.signature import extract_failure_signature


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "handoff_router"


def _oracle() -> OracleProvenance:
    return OracleProvenance(
        oracle_id="routing_policy",
        strength=OracleStrength.VERSIONED_POLICY,
        source="examples/evaluation/handoff_router/README.md",
        confidence=1.0,
        evidence_ids=("routing-policy-v1",),
        supports_hard_failure=True,
    )


def _turn(text: str) -> ConversationTurn:
    return ConversationTurn(turn_id="turn-1", role=ConversationRole.USER, content=text)


def _lookup_fixture(
    *, status: SimulatedToolStatus = SimulatedToolStatus.SUCCESS
) -> ToolFixture:
    outcome = (
        SimulatedToolOutcome(status=status, result={"invoice_id": "inv_42", "total": 42})
        if status == SimulatedToolStatus.SUCCESS
        else SimulatedToolOutcome(
            status=status,
            error_code="invoice_backend_down",
            error_message="Simulated invoice backend outage.",
        )
    )
    return ToolFixture(
        fixture_id="lookup-invoice-1",
        tool_name="lookup_invoice",
        outcome=outcome,
    )


def _handoff_constraint(
    criterion_id: str, kind: TrajectoryConstraintKind, **parameters: Any
) -> TrajectoryConstraint:
    return TrajectoryConstraint(
        criterion_id=criterion_id,
        kind=kind,
        description=f"handoff constraint {criterion_id}",
        parameters=parameters,
        oracle_ids=("routing_policy",),
    )


def _happy_scenario() -> Scenario:
    return Scenario(
        scenario_id="happy_billing_handoff",
        title="Billing question reaches the billing agent",
        conversation_turns=(_turn("What is the total for invoice inv_42?"),),
        tool_fixtures=(_lookup_fixture(),),
        required_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id="lookup_once",
                tool_name="lookup_invoice",
                min_calls=1,
                max_calls=1,
                oracle_ids=("routing_policy",),
            ),
        ),
        trajectory_constraints=(
            _handoff_constraint(
                "route_to_billing",
                TrajectoryConstraintKind.REQUIRED_HANDOFF,
                from_agent="Triage Agent",
                to_agent="Billing Agent",
            ),
            _handoff_constraint(
                "never_docs",
                TrajectoryConstraintKind.FORBIDDEN_HANDOFF,
                to_agent="Docs Agent",
            ),
            _handoff_constraint(
                "lookup_after_routing",
                TrajectoryConstraintKind.HANDOFF_BEFORE_TOOL,
                tool_name="lookup_invoice",
                to_agent="Billing Agent",
            ),
        ),
        output_criteria=(
            OutputCriterion(
                criterion_id="answer_mentions_total",
                kind=OutputCriterionKind.CONTAINS,
                description="The final answer states the invoice total.",
                parameters={"text": "42"},
                oracle_ids=("routing_policy",),
            ),
        ),
        dimension_tags=("workflow:handoff",),
        oracle_provenance=(_oracle(),),
        generation_seed=0,
    )


def _misroute_scenario() -> Scenario:
    return Scenario(
        scenario_id="misrouted_billing_request",
        title="Billing question must not reach the docs agent",
        conversation_turns=(
            _turn("Please misroute my billing question about invoice inv_42."),
        ),
        tool_fixtures=(_lookup_fixture(),),
        trajectory_constraints=(
            _handoff_constraint(
                "route_to_billing",
                TrajectoryConstraintKind.REQUIRED_HANDOFF,
                to_agent="Billing Agent",
            ),
            _handoff_constraint(
                "never_docs",
                TrajectoryConstraintKind.FORBIDDEN_HANDOFF,
                to_agent="Docs Agent",
            ),
        ),
        dimension_tags=("workflow:handoff",),
        oracle_provenance=(_oracle(),),
        generation_seed=0,
    )


def _ping_pong_scenario() -> Scenario:
    return Scenario(
        scenario_id="handoff_ping_pong",
        title="Routing must not loop between triage and billing",
        conversation_turns=(_turn("Let's play ping pong with my case."),),
        trajectory_constraints=(
            _handoff_constraint(
                "no_loop",
                TrajectoryConstraintKind.NO_HANDOFF_LOOP,
            ),
            _handoff_constraint(
                "bounded_handoffs",
                TrajectoryConstraintKind.MAX_HANDOFFS,
                maximum=2,
            ),
        ),
        resource_budgets=ResourceBudgets(max_model_turns=6),
        dimension_tags=("workflow:handoff",),
        oracle_provenance=(_oracle(),),
        generation_seed=0,
    )


def _fabricated_scenario() -> Scenario:
    return Scenario(
        scenario_id="fabricated_invoice_success",
        title="A downstream tool error must not become a success claim",
        conversation_turns=(
            _turn("Please fabricate a status for invoice inv_42."),
        ),
        tool_fixtures=(_lookup_fixture(status=SimulatedToolStatus.ERROR),),
        trajectory_constraints=(
            _handoff_constraint(
                "route_to_billing",
                TrajectoryConstraintKind.REQUIRED_HANDOFF,
                to_agent="Billing Agent",
            ),
        ),
        output_criteria=(
            OutputCriterion(
                criterion_id="no_fabricated_success",
                kind=OutputCriterionKind.NO_FABRICATED_SUCCESS,
                description="The agent must not claim success after a tool error.",
                parameters={"success_terms": ["successfully"]},
                oracle_ids=("routing_policy",),
            ),
        ),
        dimension_tags=("workflow:handoff",),
        oracle_provenance=(_oracle(),),
        generation_seed=0,
    )


def _example_module(target: Any) -> ModuleType:
    return sys.modules[type(target.model).__module__]


def _run_case(scenario: Scenario) -> tuple[CanonicalRun, CaseEvaluation, Any]:
    target, source = load_target(EXAMPLE)
    adapter = OpenAIAgentsAdapter()
    spec = adapter.inspect(target, source=source)
    assert lint_scenario(scenario, spec) == ()
    run_id = f"example-{scenario.scenario_id}"
    gateway = ToolGateway(
        spec.tools.items,
        scenario.tool_fixtures,
        world=scenario.initial_world_state,
        budgets=scenario.resource_budgets,
        run_id=run_id,
    )
    prepared = adapter.prepare(target, gateway, world_state=gateway.world, source=source)
    run = asyncio.run(
        adapter.run(
            prepared,
            scenario.conversation_turns,
            run_id=run_id,
            scenario_id=scenario.scenario_id,
            max_turns=scenario.resource_budgets.max_model_turns,
        )
    )
    return run, evaluate_run(scenario, run), target


def _assertion(evaluation: CaseEvaluation, criterion_id: str) -> Any:
    return next(
        assertion
        for assertion in evaluation.assertions
        if assertion.assertion_id == criterion_id
    )


def test_example_preflight_supports_the_full_static_graph() -> None:
    target, source = load_target(EXAMPLE)
    adapter = OpenAIAgentsAdapter()
    report = adapter.preflight(target)
    assert report.supported is True

    spec = adapter.inspect(target, source=source)
    assert {item.value.name for item in spec.tools.items} == {
        "lookup_invoice",
        "search_docs",
    }

    topology = adapter.describe_topology(target, source=source)
    assert topology is not None
    names = [agent["name"] for agent in topology["agents"]]
    assert names == ["Triage Agent", "Billing Agent", "Docs Agent"]
    assert all(
        edge["issue_codes"] == []
        for agent in topology["agents"]
        for edge in agent["handoffs"]
    )

    root, config = load_config(EXAMPLE)
    worker = inspect_in_subprocess(root, config)
    assert worker.ok is True
    assert worker.topology is not None
    assert [agent["name"] for agent in worker.topology["agents"]] == [
        "Triage Agent",
        "Billing Agent",
        "Docs Agent",
    ]


def test_happy_handoff_flow_passes_without_any_original_handler() -> None:
    run, evaluation, target = _run_case(_happy_scenario())

    assert run.termination == RunTermination.COMPLETED
    assert evaluation.verdict == Verdict.PASS
    assert _example_module(target).ORIGINAL_TOOL_CALLS == []
    executed = [
        event.payload
        for event in run.events
        if event.event_type == CanonicalEventType.HANDOFF
        and event.payload.get("ignored") is not True
    ]
    assert [payload["to_agent"] for payload in executed] == ["Billing Agent"]


def test_misrouted_billing_request_fails_both_routing_constraints() -> None:
    run, evaluation, target = _run_case(_misroute_scenario())

    assert evaluation.verdict == Verdict.FAIL
    assert _assertion(evaluation, "route_to_billing").result == Verdict.FAIL
    assert _assertion(evaluation, "never_docs").result == Verdict.FAIL
    assert _example_module(target).ORIGINAL_TOOL_CALLS == []
    executed = [
        event.payload
        for event in run.events
        if event.event_type == CanonicalEventType.HANDOFF
    ]
    assert executed and executed[0]["to_agent"] == "Docs Agent"

    signature = extract_failure_signature(evaluation)
    assert signature.schema_version == "agentcheck.failure_signature.v1"
    assert {item.assertion_id for item in signature.failed_assertions} == {
        "route_to_billing",
        "never_docs",
    }


def test_handoff_ping_pong_loop_fails_at_the_turn_budget() -> None:
    run, evaluation, target = _run_case(_ping_pong_scenario())

    assert run.termination == RunTermination.MAX_MODEL_TURNS
    assert evaluation.verdict == Verdict.FAIL
    assert _assertion(evaluation, "no_loop").result == Verdict.FAIL
    assert _assertion(evaluation, "bounded_handoffs").result == Verdict.FAIL
    assert _example_module(target).ORIGINAL_TOOL_CALLS == []


def test_fabricated_success_after_downstream_tool_error_fails() -> None:
    run, evaluation, target = _run_case(_fabricated_scenario())

    assert run.termination == RunTermination.COMPLETED
    assert evaluation.verdict == Verdict.FAIL
    assert _assertion(evaluation, "no_fabricated_success").result == Verdict.FAIL
    assert _assertion(evaluation, "route_to_billing").result == Verdict.PASS
    assert _example_module(target).ORIGINAL_TOOL_CALLS == []


def test_worker_subprocess_executes_a_handoff_scenario_offline() -> None:
    root, config = load_config(EXAMPLE)
    scenario = _happy_scenario()
    result = run_scenario_in_subprocess(
        root, config, scenario, run_id="example-worker-happy"
    )

    run = result.require_value()
    evaluation = evaluate_run(scenario, run)
    assert evaluation.verdict == Verdict.PASS
    assert "[debug] original" not in result.stdout
    handoff_events = [
        event
        for event in run.events
        if event.event_type == CanonicalEventType.HANDOFF
    ]
    assert handoff_events and handoff_events[0].payload["to_agent"] == "Billing Agent"


def _offline_environment() -> dict[str, str]:
    inherited = ("PATH", "LANG", "LC_ALL", "TZ", "TMPDIR", "SSL_CERT_FILE")
    environment = {name: os.environ[name] for name in inherited if name in os.environ}
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def test_offline_cli_inspects_topology_and_freezes_boundary_suite(
    tmp_path: Path,
) -> None:
    target = tmp_path / "handoff_router"
    shutil.copytree(EXAMPLE, target)

    def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "agentcheck", *arguments],
            cwd=REPOSITORY_ROOT,
            env=_offline_environment(),
            text=True,
            capture_output=True,
            check=False,
            timeout=180.0,
        )

    inspection = _run_cli("inspect", str(target))
    assert inspection.returncode == 0, inspection.stderr
    assert "Handoff topology (3 reachable agents):" in inspection.stdout
    assert "transfer_to_billing_agent to Billing Agent" in inspection.stdout
    assert "Preflight: supported" in inspection.stdout

    generation = _run_cli("generate", str(target))
    assert generation.returncode == 0, generation.stderr
    assert (target / "agentcheck-suite.json").is_file()

    execution = _run_cli(
        "test", str(target), "--no-store", "--run-id", "handoff-boundary-e2e"
    )
    # Schema-boundary cases forbid invalid argument shapes the scripted models
    # never produce, so the frozen suite passes offline with no provider key.
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert "[debug] original" not in execution.stdout
