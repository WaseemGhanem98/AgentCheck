"""Public serialized argument contracts preserve JSON boolean/number types."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentcheck.config import AgentCheckConfig
from agentcheck.domain import (
    AgentSpec, CanonicalEvent, CanonicalEventType, CanonicalRun,
    ConversationRole, ConversationTurn, OracleProvenance, OracleStrength,
    RunTermination, Scenario, SimulatedToolOutcome, SimulatedToolStatus,
    ToolBehaviorConstraint, ToolFixture, Verdict,
)
from agentcheck.evaluate import evaluate_run
from agentcheck.generate import lint_scenario
from agentcheck.generate.suite import FrozenSuite, build_frozen_suite, load_frozen_suite
from agentcheck.runner import ToolCallBlockedError, ToolGateway
from agentcheck.schema_safety import offline_validator


NOW = datetime(2026, 9, 6, tzinfo=timezone.utc)
SCHEMA = {"type": "object"}
MISMATCHES = [
    (True, 1), (False, 0), (1, True), (0, False), (True, 1.0), (0.0, False),
    ({"x": True}, {"x": 1}), ([True], [1]),
    ({"x": [{"y": False}]}, {"x": [{"y": 0}]}),
    ([{"x": [1, True]}], [{"x": [1, 1]}]),
]
MATCHES = [
    (True, True), (False, False), (1, 1.0), (0.0, 0), (None, None),
    ("true", "true"), ({"x": 1}, {"x": 1.0}), ([True, 1], [True, 1.0]),
    ({"x": [], "y": {}}, {"y": {}, "x": []}),
]


def _spec() -> AgentSpec:
    source = {"kind": "developer_config", "locator": "owned-json-type-test"}

    def prop(value):
        return {"value": value, "source": source, "confidence": 1.0, "authoritative": True,
                "evidence": [{"evidence_id": "owned", "summary": "Inert authored schema."}]}

    return AgentSpec.model_validate_json(json.dumps({
        "spec_id": "json-type-test",
        "identity": {k: prop(v) for k, v in dict(name="Test", framework="custom",
                    framework_version=None, provider=None, model=None).items()},
        "interface": {k: prop(v) for k, v in dict(entrypoint="owned:inert", input_modalities=["text"],
                     output_modalities=["text"], input_schema=None, output_schema=None, interactive=False).items()},
        "instructions": {"system": prop("Follow the explicit JSON contract."), "developer": prop(None)},
        "tools": {"items": [prop({"name": "choose", "input_schema": SCHEMA, "replaceable": True})]},
        "runtime": {k: prop(None) for k in ("max_model_turns", "max_tool_calls", "timeout_seconds",
                                           "token_budget", "cost_budget_usd")},
        "observability": {"supported_event_types": prop([]), "usage_metrics": prop([]),
                          "provider_request_ids": prop(False), "source_event_links": prop(False)},
        "provenance": {"inspector": "owned-fixture", "inspector_version": "1", "inspected_at": NOW.isoformat(),
                       "target": "owned:inert", "sources": [source]},
    }))


def _fixture(expected=None, *, fixture_id="finite", invocation_index=None, result="controlled"):
    return ToolFixture(
        fixture_id=fixture_id, tool_name="choose", arguments_match=expected or {},
        invocation_index=invocation_index,
        outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result=result),
    )


def _scenario(expected, *, kind="allowed", weak=False, fixtures=None, max_calls=None):
    constraint = ToolBehaviorConstraint(
        criterion_id="typed", tool_name="choose", arguments_match=expected,
        min_calls=1 if kind == "required" else 0,
        max_calls=0 if kind == "forbidden" else max_calls, oracle_ids=("authored",),
    )
    return Scenario.model_validate_json(Scenario(
        scenario_id="typed-values", title="Typed JSON values",
        conversation_turns=(ConversationTurn(turn_id="request", role=ConversationRole.USER,
                                             content="Use only the explicitly supplied JSON value."),),
        oracle_provenance=(OracleProvenance(oracle_id="authored", strength=OracleStrength.EXPLICIT_INSTRUCTION,
                           source="Trusted explicit synthetic instruction.", confidence=0.0 if weak else 1.0,
                           supports_hard_failure=not weak, evidence_ids=("request",)),),
        tool_fixtures=(_fixture(),) if fixtures is None else fixtures,
        dimension_tags=("owned-json-types",), generation_seed=9,
        **{f"{kind}_tool_behavior": (constraint,)},
    ).model_dump_json())


def _replace(scenario, **updates):
    data = scenario.model_dump(mode="python")
    data.update(fingerprint="", **updates)
    return Scenario.model_validate(data)


def _execute(scenario, arguments, *, exact=False, schema=None):
    fixtures = scenario.tool_fixtures
    if exact:
        fixtures = [{**f.model_dump(mode="json"), "match_mode": "exact"} for f in fixtures]
    gateway = ToolGateway({"choose": {"input_schema": SCHEMA if schema is None else schema}},
                          fixtures, run_id="owned", now=lambda: NOW)
    for value in arguments:
        try:
            gateway.invoke("choose", value)
        except ToolCallBlockedError:
            pass
    events = [CanonicalEvent(event_id="request-event", run_id="owned", sequence=0, timestamp=NOW,
              event_type=CanonicalEventType.USER_TURN,
              payload={"turn_id": "request", "text": scenario.conversation_turns[0].content},
              metadata={"scenario_input": True})]
    events.extend(event.model_copy(update={"sequence": index}) for index, event in enumerate(gateway.events, 1))
    events.append(CanonicalEvent(event_id="final", run_id="owned", sequence=len(events), timestamp=NOW,
                  event_type=CanonicalEventType.FINAL_OUTPUT, payload={"text": "Finished."}))
    run = CanonicalRun.model_validate_json(CanonicalRun(
        run_id="owned", scenario_id=scenario.scenario_id, target_id="inert", started_at=NOW, ended_at=NOW,
        termination=RunTermination.COMPLETED, events=tuple(events), tool_attempts=gateway.attempts,
        tool_outcomes=gateway.outcomes, final_output="Finished.",
    ).model_dump_json())
    return run, evaluate_run(scenario, run)


@pytest.mark.parametrize("expected,actual", MISMATCHES)
def test_explicit_json_type_mismatch_is_failure_after_public_roundtrip(expected, actual):
    scenario = _scenario({"value": expected})
    assert offline_validator(SCHEMA).is_valid({"value": actual})
    run, evaluation = _execute(scenario, [{"value": actual}])
    assert json.dumps(run.tool_attempts[0].arguments, sort_keys=True) == json.dumps({"value": actual}, sort_keys=True)
    assert run.tool_outcomes[0].status.value == "success"
    assert evaluation.verdict is Verdict.FAIL
    mismatch = next(a for a in evaluation.assertions if a.assertion_id.endswith("unexpected_arguments"))
    assert mismatch.result is Verdict.FAIL and mismatch.confidence == 1.0


@pytest.mark.parametrize("expected,actual", MATCHES)
def test_equal_json_values_keep_existing_pass(expected, actual):
    _, evaluation = _execute(_scenario({"value": expected}), [{"value": actual}])
    assert evaluation.verdict is Verdict.PASS


@pytest.mark.parametrize("expected,actual", MISMATCHES)
def test_weak_json_type_mismatch_is_inconclusive_not_failure(expected, actual):
    _, evaluation = _execute(_scenario({"value": expected}, weak=True), [{"value": actual}])
    assert evaluation.verdict is Verdict.INCONCLUSIVE
    mismatch = next(a for a in evaluation.assertions if a.assertion_id.endswith("unexpected_arguments"))
    assert "authoritative high-confidence oracle evidence" in mismatch.missing_evidence


@pytest.mark.parametrize("exact", [False, True], ids=["subset", "exact"])
@pytest.mark.parametrize("expected,actual", MISMATCHES)
def test_wrong_typed_fixture_is_missing_not_an_invented_outcome(exact, expected, actual):
    scenario = _scenario({}, fixtures=(_fixture({"value": expected}),))
    run, evaluation = _execute(scenario, [{"value": actual}], exact=exact)
    assert run.tool_outcomes[0].error is not None
    assert run.tool_outcomes[0].error.code == "fixture_not_found"
    assert evaluation.verdict is Verdict.INFRA_ERROR


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("expected,actual", MATCHES)
def test_matching_typed_fixture_still_runs(exact, expected, actual):
    scenario = _scenario({}, fixtures=(_fixture({"value": expected}),))
    run, evaluation = _execute(scenario, [{"value": actual}], exact=exact)
    assert run.tool_outcomes[0].result == "controlled"
    assert evaluation.verdict is Verdict.PASS


@pytest.mark.parametrize("expected,actual,verdict", [
    ({}, {"anything": 1}, Verdict.PASS),
    ({"value": None}, {}, Verdict.FAIL),
    ({"value": 1}, {"value": "1"}, Verdict.FAIL),
    ({"value": "A"}, {"value": "a"}, Verdict.FAIL),
    ({"value": 1}, {"value": 1, "extra": True}, Verdict.PASS),
    ({"value": {"x": 1}}, {"value": {"x": 1, "extra": True}}, Verdict.FAIL),
    ({"value": [1, 2]}, {"value": [2, 1]}, Verdict.FAIL),
    ({"value": [1]}, {"value": [1, 2]}, Verdict.FAIL),
])
def test_existing_behavior_match_topology_is_preserved(expected, actual, verdict):
    _, evaluation = _execute(_scenario(expected), [actual])
    assert evaluation.verdict is verdict


def test_recursive_fixture_subset_does_not_make_behavior_recursive_subset():
    expected, actual = {"value": {"x": True}}, {"value": {"x": True, "extra": 2}}
    scenario = _scenario(expected, fixtures=(_fixture(expected),))
    run, evaluation = _execute(scenario, [actual])
    assert run.tool_outcomes[0].status.value == "success"
    assert evaluation.verdict is Verdict.FAIL
    run, evaluation = _execute(_scenario({}, fixtures=(_fixture(expected),)), [actual], exact=True)
    assert run.tool_outcomes[0].error.code == "fixture_not_found"
    assert evaluation.verdict is Verdict.INFRA_ERROR


def test_objects_inside_fixture_arrays_are_complete_values():
    scenario = _scenario({}, fixtures=(_fixture({"value": [{"x": True}]}),))
    _, evaluation = _execute(scenario, [{"value": [{"x": True, "extra": 2}]}])
    assert evaluation.verdict is Verdict.INFRA_ERROR


def test_typed_fixture_is_still_finite():
    scenario = _scenario({}, fixtures=(_fixture({"value": True}),))
    run, evaluation = _execute(scenario, [{"value": True}, {"value": True}])
    assert run.tool_outcomes[0].status.value == "success"
    assert run.tool_outcomes[1].error.code == "fixture_not_found"
    assert evaluation.verdict is Verdict.INFRA_ERROR


@pytest.mark.parametrize("kind,no_call", [("allowed", Verdict.PASS), ("required", Verdict.INCONCLUSIVE)])
def test_optional_and_required_no_call_semantics_are_unchanged(kind, no_call):
    _, evaluation = _execute(_scenario({"value": True}, kind=kind), [])
    assert evaluation.verdict is no_call


def test_boolean_forbidden_contract_does_not_forbid_number():
    scenario = _scenario({"value": True}, kind="forbidden")
    _, evaluation = _execute(scenario, [{"value": 1}])
    assert evaluation.verdict is Verdict.PASS
    _, evaluation = _execute(scenario, [{"value": True}])
    assert evaluation.verdict is Verdict.FAIL


def test_required_count_and_allowed_union_do_not_conflate_types():
    scenario = _scenario({"value": True}, kind="required", max_calls=1,
                         fixtures=(_fixture(invocation_index=1), _fixture(fixture_id="second", invocation_index=2)))
    allowed = ToolBehaviorConstraint(criterion_id="number-allowed", tool_name="choose",
              arguments_match={"value": 1}, min_calls=0, oracle_ids=("authored",))
    scenario = _replace(scenario, allowed_tool_behavior=(allowed,))
    _, evaluation = _execute(scenario, [{"value": True}, {"value": 1}])
    assert evaluation.verdict is Verdict.PASS
    _, evaluation = _execute(scenario, [{"value": 1}])
    assert evaluation.verdict is Verdict.FAIL


def test_conditional_confirmation_scope_is_type_sensitive():
    scenario = _scenario({"value": True}, kind="required")
    guarded = scenario.required_tool_behavior[0].model_copy(update={
        "min_calls": 0, "confirmation_required_before_call": True})
    allowed = ToolBehaviorConstraint(criterion_id="number-allowed", tool_name="choose",
              arguments_match={"value": 1}, min_calls=0, oracle_ids=("authored",))
    scenario = _replace(scenario, required_tool_behavior=(), allowed_tool_behavior=(guarded, allowed))
    _, evaluation = _execute(scenario, [{"value": 1}])
    assert evaluation.verdict is Verdict.PASS
    _, evaluation = _execute(scenario, [{"value": True}])
    assert evaluation.verdict is Verdict.FAIL


def test_fixture_gap_cannot_be_excused_by_wrong_typed_forbidden_contract():
    scenario = _scenario({"value": True}, kind="forbidden", fixtures=())
    _, evaluation = _execute(scenario, [{"value": 1}])
    assert evaluation.verdict is Verdict.INFRA_ERROR
    _, evaluation = _execute(scenario, [{"value": True}])
    assert evaluation.verdict is Verdict.FAIL


def test_boolean_schema_rejection_is_unchanged():
    schema = {"type": "object", "properties": {"value": {"type": "boolean"}}, "required": ["value"]}
    _, evaluation = _execute(_scenario({"value": True}), [{"value": 1}], schema=schema)
    assert evaluation.verdict is Verdict.FAIL


def _lint_candidate():
    return _scenario({"value": True}, kind="required", fixtures=(
        _fixture({"value": True}), _fixture({"value": 1}, fixture_id="number"),
    ))


def test_lint_disjoint_types_are_not_ambiguous_or_contradictory():
    scenario = _lint_candidate()
    forbidden = ToolBehaviorConstraint(criterion_id="number-forbidden", tool_name="choose",
                arguments_match={"value": 1}, min_calls=0, max_calls=0, oracle_ids=("authored",))
    scenario = _replace(scenario, forbidden_tool_behavior=(forbidden,))
    codes = {issue.code for issue in lint_scenario(scenario, _spec())}
    assert not codes


def test_lint_wrong_typed_fixture_does_not_satisfy_required_call():
    scenario = _scenario({"value": True}, kind="required", fixtures=(_fixture({"value": 1}),))
    assert "missing_fixture" in {issue.code for issue in lint_scenario(scenario, _spec())}


def test_lint_numeric_equivalence_remains_ambiguous():
    scenario = _scenario({"value": 1}, kind="required", fixtures=(
        _fixture({"value": 1}), _fixture({"value": 1.0}, fixture_id="float"),
    ))
    assert "ambiguous_fixture" in {issue.code for issue in lint_scenario(scenario, _spec())}


def test_lint_change_affects_suite_admission_at_existing_builder_boundary(monkeypatch):
    import agentcheck.generate.suite as generation
    candidate = _lint_candidate()
    # An inert candidate injection tests the real lint/selection/freeze pipeline,
    # not a claim that a stock generator already emits this exact fixture pair.
    monkeypatch.setattr(generation, "build_positive_path_cases", lambda *a, **kw: [SimpleNamespace(
        scenario=candidate, representative=True, tool_name="choose", shallow_parameters=(),
    )])
    suite = build_frozen_suite(_spec(), AgentCheckConfig(), seed=9)
    assert candidate.scenario_id in {case.scenario.scenario_id for case in suite.cases}
    assert candidate.scenario_id not in {case.scenario.scenario_id for case in suite.rejected}
    assert suite.provenance.generator_version == "4"


def test_frozen_v3_validates_without_rewriting_bytes_or_identity():
    path = Path(__file__).parent / "fixtures" / "json-arguments-generator-v3.json"
    before = path.read_bytes()
    stored = load_frozen_suite(path)
    assert stored.provenance.generator_version == "3"
    assert stored.fingerprint == "sha256:18ad61dd72dc7c5d345e4b85bdbec90415e07a019ae7bc88df5043b52e2407e8"
    assert FrozenSuite.model_validate_json(stored.model_dump_json()).fingerprint == stored.fingerprint
    assert path.read_bytes() == before
    current = build_frozen_suite(_spec(), AgentCheckConfig(), seed=9,
                                 representative_inputs={"choose": {"value": True}})
    assert current.provenance.generator_version == "4"
    assert stored.fingerprint != current.fingerprint
    assert [case.scenario.model_dump() for case in stored.cases] == [case.scenario.model_dump() for case in current.cases]
    assert [case.scenario.scenario_id for case in stored.rejected] == [
        case.scenario.scenario_id for case in current.rejected]
    tampered = json.loads(before)
    tampered["provenance"]["generator_version"] = "4"
    with pytest.raises(ValueError, match="fingerprint"):
        FrozenSuite.model_validate_json(json.dumps(tampered))


def test_frozen_v3_data_is_explicitly_packaged():
    lines = (Path(__file__).resolve().parents[2] / "MANIFEST.in").read_text().splitlines()
    assert "include tests/agentcheck/fixtures/json-arguments-generator-v3.json" in lines
    assert "recursive-include tests *.json" not in lines
