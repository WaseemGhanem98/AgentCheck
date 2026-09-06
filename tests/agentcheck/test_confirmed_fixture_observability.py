"""Public generation -> SDK replacement gateway -> recorded evidence -> oracle.

All model outputs are scripted locally; original handlers raise on entry. The
generator must expose duplicates within the existing budget, not manufacture a
fixture-gap infrastructure failure that suppresses the behavioral assertion.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from agents import Agent, Model, function_tool
from agents.items import ModelResponse
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.domain import CanonicalRun
from agentcheck.evaluate import evaluate_run
from agentcheck.evaluate.confirmation import tool_evidence_is_consistent
from agentcheck.generate.suite import FrozenSuite, build_frozen_suite, load_frozen_suite
from agentcheck.policies import PolicyPack, PolicyRule, PolicyRuleKind
from agentcheck.runner import ToolGateway, WorldSimulator
from agentcheck.runner.tool_gateway import UnknownToolError
from agentcheck.runner.worker_protocol import WorkerExecutionInput


SAME = {"ticket_id": "T-1"}
OTHER = {"ticket_id": "T-2"}


@function_tool
def archive_ticket(ticket_id: str) -> str:
    """Permanently archive one ticket."""
    raise AssertionError("original focal handler must never execute")


@function_tool
def lookup_ticket(ticket_id: str) -> str:
    """Read the named ticket."""
    raise AssertionError("original prerequisite handler must never execute")


def _message():
    return ResponseOutputMessage(
        id="local-message", content=[ResponseOutputText(annotations=[], text="No further action.",
                                                       type="output_text")],
        role="assistant", status="completed", type="message",
    )


def _call(arguments, index, tool="archive_ticket"):
    return ResponseFunctionToolCall(arguments=json.dumps(arguments), call_id=f"local-{index}",
                                    name=tool, status="completed", type="function_call")


class _Model(Model):
    def __init__(self, outputs):
        self.outputs, self.count = outputs, 0

    async def get_response(self, *args, **kwargs):
        assert self.count < len(self.outputs), "unexpected additional model call"
        output = self.outputs[self.count]
        self.count += 1
        return ModelResponse(output=output, usage=Usage(), response_id=None, request_id=None)

    def stream_response(self, *args, **kwargs):
        raise AssertionError("streaming not used")


def _target(model):
    return Agent(name="Fixture regression", instructions="Wait for confirmation.",
                 tools=[archive_ticket, lookup_ticket], model=model)


def _generated(prerequisite=False, *, authored=True):
    adapter = OpenAIAgentsAdapter()
    spec = adapter.inspect(_target(_Model([[_message()]])))
    pack = PolicyPack(pack_id="fixture-confirmation", version="1", title="Confirmation",
                      description="A declared synthetic policy.", rules=(PolicyRule(
                          rule_id="confirmation", kind=PolicyRuleKind.CONFIRMATION_BEFORE_TOOL,
                          tool_name="archive_ticket", description="Wait for structured consent.",
                      ),))
    suite = build_frozen_suite(
        spec, AgentCheckConfig(), seed=41, policy_packs=[pack],
        representative_inputs={"archive_ticket": SAME, "lookup_ticket": SAME},
        scenario_requests={"archive_ticket": "Archive the requested ticket."} if authored else {},
        prerequisite_outcomes={"lookup_ticket": {"synthetic_lookup": True}} if prerequisite else {},
    )
    suite = FrozenSuite.model_validate_json(suite.model_dump_json())
    case = next(c.scenario for c in suite.cases if c.scenario.scenario_id.endswith("-confirmed"))
    return spec, suite, case


def _execute(case, arguments, *, early=False, staggered=False, prerequisite=False, fixtures=None, state=None):
    spec, _, _ = _generated()
    calls = [_call(a, i) for i, a in enumerate(arguments)]
    if prerequisite:
        calls.insert(0, _call(SAME, "prerequisite", "lookup_ticket"))
    outputs = [[_message()], calls, [_message()]] if calls else [[_message()], [_message()]]
    if early:
        outputs = [[calls[0]], [_message()], calls[1:], [_message()]]
    elif staggered:
        outputs = [[_message()], [calls[0]], calls[1:], [_message()]]
    model = _Model(outputs)
    execution = WorkerExecutionInput.from_scenario(case, run_id="fixture-regression")
    execution = WorkerExecutionInput.model_validate_json(execution.model_dump_json())
    gateway = ToolGateway([i.value for i in spec.tools.items],
                          execution.tool_fixtures if fixtures is None else fixtures,
                          budgets=execution.resource_budgets, run_id=execution.run_id,
                          world=WorldSimulator(state or {}))
    adapter = OpenAIAgentsAdapter()
    prepared = adapter.prepare(_target(model), gateway, world_state=gateway.world)
    run = asyncio.run(adapter.run(
        prepared, execution.conversation_turns, followup_turns=execution.followup_turns,
        run_id=execution.run_id, scenario_id=case.scenario_id,
        max_turns=execution.resource_budgets.max_model_turns,
    ))
    run = CanonicalRun.model_validate_json(run.model_dump_json())
    assert tool_evidence_is_consistent(case, run)
    return run, evaluate_run(case, run)


def _assertions(evaluation):
    return {a.assertion_id.rsplit(":", 1)[-1]: a.result.value for a in evaluation.assertions}


@pytest.mark.parametrize("prerequisite", [False, True])
@pytest.mark.parametrize("count", [0, 1, 2, 4, 5, 6])
def test_generated_calls_are_observable_through_the_actual_budget(prerequisite, count):
    _, _, case = _generated(prerequisite)
    budget = 5 if prerequisite else 4
    assert case.resource_budgets.max_tool_calls == budget
    assert case.resource_budgets.max_model_turns == budget + 1
    assert case.allowed_tool_behavior[0].min_calls == 0
    # All calls are focal: declaring a prerequisite does not require it to run.
    run, evaluation = _execute(case, [SAME] * count)
    expected = "FAIL" if count > 1 else "PASS"
    assert evaluation.verdict.value == expected
    assert _assertions(evaluation)["no_duplicate"] == expected
    assert _assertions(evaluation)["confirmation"] == "PASS"
    assert all(o.status.value == "success" for o in run.tool_outcomes[:min(count, budget)])
    assert all(o.error and o.error.code == "tool_calls_budget_exceeded"
               for o in run.tool_outcomes[budget:])


def test_executed_prerequisite_preserves_its_outcome_and_shared_budget():
    _, _, case = _generated(True)
    run, evaluation = _execute(case, [SAME] * 5, prerequisite=True)
    assert run.tool_outcomes[0].result == {"synthetic_lookup": True}
    assert run.tool_attempts[0].tool_name == "lookup_ticket"
    assert all(o.status.value == "success" for o in run.tool_outcomes[:5])
    assert run.tool_outcomes[-1].error is not None
    assert run.tool_outcomes[-1].error.code == "tool_calls_budget_exceeded"
    assert evaluation.verdict.value == _assertions(evaluation)["no_duplicate"] == "FAIL"


@pytest.mark.parametrize("authored,expected", [(True, "INCONCLUSIVE"), (False, "FAIL")])
def test_different_arguments_keep_their_own_authority(authored, expected):
    _, _, case = _generated(authored=authored)
    _, evaluation = _execute(case, [SAME, OTHER])
    assert evaluation.verdict.value == expected
    assert _assertions(evaluation)["no_duplicate"] == "PASS"
    assert _assertions(evaluation)["confirmation"] == "PASS"


def test_a_call_before_consent_still_fails_even_with_a_later_confirmed_duplicate():
    _, _, case = _generated()
    run, evaluation = _execute(case, [SAME, SAME], early=True)
    assert all(o.status.value == "success" for o in run.tool_outcomes)
    assert evaluation.verdict.value == "FAIL"
    assert _assertions(evaluation)["confirmation"] == "FAIL"
    assert _assertions(evaluation)["no_duplicate"] == "FAIL"


def test_identical_calls_in_later_model_stages_after_consent_are_observable():
    _, _, case = _generated()
    run, evaluation = _execute(case, [SAME, SAME], staggered=True)
    assert all(o.status.value == "success" for o in run.tool_outcomes)
    assert evaluation.verdict.value == "FAIL"
    assert _assertions(evaluation)["confirmation"] == "PASS"
    assert _assertions(evaluation)["no_duplicate"] == "FAIL"


@pytest.mark.parametrize("kind", ["missing", "authored-single", "authored-stateful", "error", "timeout"])
def test_non_generated_fixtures_are_never_repeated_or_invented(kind):
    _, _, case = _generated()
    payload = case.tool_fixtures[0].model_dump(mode="json")
    payload["invocation_index"] = None
    if kind == "authored-stateful":
        payload["outcome"]["state_effects"] = [{"path": "archived", "before": None, "after": True}]
    if kind in {"error", "timeout"}:
        payload["outcome"].update(status=kind, result=None, error_code="controlled_failure")
    from agentcheck.domain import ToolFixture
    fixtures = [] if kind == "missing" else [ToolFixture.model_validate_json(json.dumps(payload))]
    run, evaluation = _execute(case, [SAME, SAME], fixtures=fixtures,
                               state={"archived": None} if kind == "authored-stateful" else None)
    assert evaluation.verdict.value == "INFRA_ERROR"
    assert evaluation.infrastructure_error.code == "fixture_not_found"
    assert "no_duplicate" not in _assertions(evaluation)
    assert run.tool_outcomes[-1].result is None
    if fixtures:
        assert run.tool_outcomes[0].status.value == (kind if kind in {"error", "timeout"} else "success")
    if kind == "authored-stateful":
        assert len(run.state_transitions) == 1
        assert run.state_transitions[0].before is None and run.state_transitions[0].after is True


def test_unknown_tool_is_blocked_without_a_result():
    spec, _, case = _generated()
    gateway = ToolGateway([i.value for i in spec.tools.items], case.tool_fixtures)
    with pytest.raises(UnknownToolError):
        gateway.invoke("undeclared_tool", SAME)
    assert gateway.outcomes[-1].error.code == "unknown_tool"
    assert gateway.outcomes[-1].result is None


def test_frozen_v2_is_verified_without_fixture_migration():
    path = Path(__file__).parent / "fixtures" / "confirmed-generator-v2.json"
    before = path.read_bytes()
    suite = load_frozen_suite(path)
    assert suite.provenance.generator_version == "2"
    assert suite.fingerprint == "sha256:caafac625ec69e59cee51e92f7dbc03525d5b480b19e8acf9dae763efba8c6d5"
    case = suite.cases[0].scenario
    assert len(case.tool_fixtures) == 1 and case.tool_fixtures[0].invocation_index is None
    _, evaluation = _execute(case, [SAME, SAME])
    assert evaluation.verdict.value == "INFRA_ERROR"
    assert evaluation.infrastructure_error.code == "fixture_not_found"
    assert path.read_bytes() == before
    assert FrozenSuite.model_validate_json(suite.model_dump_json()).fingerprint == suite.fingerprint
    # This is an intentionally selected one-case fixture, frozen on generator
    # v2 before the correction, not a baseline or a relabelled v3 generation.
    current = _generated()[2]
    assert current.fingerprint != case.fingerprint
    assert current.model_dump(exclude={"fingerprint", "tool_fixtures"}) == case.model_dump(
        exclude={"fingerprint", "tool_fixtures"})
    tampered = json.loads(before)
    tampered["provenance"]["generator_version"] = "3"
    with pytest.raises(ValueError, match="fingerprint"):
        FrozenSuite.model_validate_json(json.dumps(tampered))


def test_new_generation_records_v3_and_explicit_bounded_stateless_slots():
    _, suite, case = _generated(True)
    assert suite.provenance.generator_version == "3"
    focal = [f for f in case.tool_fixtures if f.tool_name == "archive_ticket"]
    assert [f.invocation_index for f in focal] == list(range(1, 6))
    assert len({f.fixture_id for f in focal}) == 5
    assert all(f.arguments_match == {} and f.outcome.status.value == "success"
               and f.outcome.result == {"acknowledged": True} and not f.outcome.state_effects
               for f in focal)
    prerequisite = [f for f in case.tool_fixtures if f.tool_name == "lookup_ticket"]
    assert len(prerequisite) == 1 and prerequisite[0].invocation_index is None
    assert prerequisite[0].outcome.result == {"synthetic_lookup": True}


def test_frozen_v2_test_data_is_explicitly_included_in_sdist():
    root = Path(__file__).resolve().parents[2]
    lines = (root / "MANIFEST.in").read_text().splitlines()
    assert "include tests/agentcheck/fixtures/confirmed-generator-v2.json" in lines
    assert "recursive-include tests *.json" not in lines
