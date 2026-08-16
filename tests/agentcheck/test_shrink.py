from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.application import execute_suite, replay_suite, shrink_suite
from agentcheck.cli import main
from agentcheck.domain import (
    AgentSpec,
    AssertionResult,
    CaseEvaluation,
    Evidence,
    EvidenceKind,
    Scenario,
    Verdict,
    utc_now,
)
from agentcheck.errors import ConfigurationError
from agentcheck.evaluate import infrastructure_evaluation
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.inspect import load_target
from agentcheck.replay import (
    EnvironmentRequirements,
    ReplayManifest,
    SourceBinding,
    SpecBinding,
    encode_replay_manifest,
    load_replay_manifest,
)
from agentcheck.shrink import (
    REDUCTION_ORDER,
    FailureSignature,
    ShrinkResult,
    extract_failure_signature,
    measure_complexity,
    shrink_scenario,
    signatures_match,
    unsupported_failure_reason,
)
from agentcheck.shrink.candidates import reconstruct_scenario
from agentcheck.shrink.complexity import is_strictly_smaller
from agentcheck.shrink.result import ShrinkBudget, encode_shrink_result


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
SEED = 1729
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes required")


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "account_agent"
    shutil.copytree(EXAMPLE, target, symlinks=False)
    return target


def _failing_evaluation(
    *assertion_ids: str,
    scenario_id: str = "case-a",
    oracle_id: str = "oracle-a",
) -> CaseEvaluation:
    now = utc_now()
    evidence: list[Evidence] = []
    assertions: list[AssertionResult] = []
    for assertion_id in assertion_ids:
        evidence_id = f"ev-{assertion_id}"
        evidence.append(
            Evidence(
                evidence_id=evidence_id,
                kind=EvidenceKind.TOOL_ATTEMPT,
                summary="Deterministic failure evidence.",
                source_ids=("attempt-1",),
            )
        )
        assertions.append(
            AssertionResult(
                assertion_id=assertion_id,
                criterion="stable structured criterion",
                result=Verdict.FAIL,
                oracle_ids=(oracle_id,),
                supporting_evidence_ids=(evidence_id,),
                rationale="Required assertion failed.",
            )
        )
    return CaseEvaluation(
        evaluation_id="eval-case",
        scenario_id=scenario_id,
        run_id="run-case",
        verdict=Verdict.FAIL,
        assertions=tuple(assertions),
        evidence=tuple(evidence),
        started_at=now,
        completed_at=now,
        summary="fail",
    )


def _passing_evaluation(scenario_id: str = "case-a") -> CaseEvaluation:
    now = utc_now()
    assertion = AssertionResult(
        assertion_id=f"{scenario_id}:ok",
        criterion="passes",
        result=Verdict.PASS,
        oracle_ids=(f"{scenario_id}:oracle",),
        rationale="Required assertion passed.",
    )
    return CaseEvaluation(
        evaluation_id="eval-pass",
        scenario_id=scenario_id,
        run_id="run-pass",
        verdict=Verdict.PASS,
        assertions=(assertion,),
        evidence=(),
        started_at=now,
        completed_at=now,
        summary="pass",
    )


def _inconclusive_evaluation(scenario_id: str = "case-a") -> CaseEvaluation:
    now = utc_now()
    assertion = AssertionResult(
        assertion_id=f"{scenario_id}:maybe",
        criterion="unmeasured",
        result=Verdict.INCONCLUSIVE,
        oracle_ids=(f"{scenario_id}:oracle",),
        missing_evidence=("token usage",),
        rationale="Budget could not be measured.",
    )
    return CaseEvaluation(
        evaluation_id="eval-inc",
        scenario_id=scenario_id,
        run_id="run-inc",
        verdict=Verdict.INCONCLUSIVE,
        assertions=(assertion,),
        evidence=(),
        started_at=now,
        completed_at=now,
        summary="inconclusive",
    )


def _delete_scenario() -> Scenario:
    suite = build_account_support_suite(seed=SEED)
    return next(item for item in suite if item.scenario_id == "delete_without_confirmation")


def _pad_scenario(scenario: Scenario, *, secret_turn: bool = False) -> Scenario:
    payload = json.loads(scenario.model_dump_json())
    noise_turns = [
        {
            "turn_id": f"noise-{index}",
            "role": "user",
            "content": f"Unrelated AgentCheck control turn {index}; ignore it.",
            "metadata": {},
        }
        for index in range(1, 5)
    ]
    if secret_turn:
        noise_turns.append(
            {
                "turn_id": "noise-secret",
                "role": "user",
                "content": "sk-thisisafakesecretvalue12",
                "metadata": {},
            }
        )
    payload["conversation_turns"] = noise_turns + payload["conversation_turns"]
    payload["initial_world_state"]["noise"] = {"unused": True}
    payload["tool_fixtures"].append(
        {
            "fixture_id": f"{scenario.scenario_id}:lookup_account:noise",
            "tool_name": "lookup_account",
            "arguments_match": {"account_id": "acct_noise"},
            "priority": 0,
            "outcome": {
                "status": "success",
                "result": {"ok": True},
                "latency_ms": 0.0,
                "state_effects": [],
            },
        }
    )
    payload["injected_faults"] = [
        {
            "fault_id": f"{scenario.scenario_id}:fault-noise",
            "tool_name": "lookup_account",
            "fault_type": "timeout",
            "invocation_index": 9,
            "message": "unused injected fault",
        }
    ]
    payload["fingerprint"] = ""
    return Scenario.model_validate_json(json.dumps(payload))


def _account_spec() -> AgentSpec:
    loaded, source = load_target(EXAMPLE)
    return OpenAIAgentsAdapter().inspect(loaded, source=source)


def test_reduction_order_is_documented_and_stable() -> None:
    assert REDUCTION_ORDER[0] == "conversation_turns"
    assert REDUCTION_ORDER[1] == "tool_fixtures"
    assert REDUCTION_ORDER[2] == "injected_faults"
    assert "expected_postconditions" in REDUCTION_ORDER


def test_same_deterministic_failure_is_accepted() -> None:
    left = extract_failure_signature(_failing_evaluation("a:state", "a:tool"))
    right = extract_failure_signature(_failing_evaluation("a:tool", "a:state"))
    assert signatures_match(left, right)
    assert left.fingerprint == right.fingerprint


def test_different_fail_is_rejected() -> None:
    left = extract_failure_signature(_failing_evaluation("a:state"))
    right = extract_failure_signature(_failing_evaluation("a:other"))
    assert not signatures_match(left, right)


def test_pass_inconclusive_and_infra_are_not_shrinkable() -> None:
    scenario = _delete_scenario()
    assert unsupported_failure_reason(_passing_evaluation()) is not None
    assert unsupported_failure_reason(_inconclusive_evaluation()) is not None
    infra = infrastructure_evaluation(
        scenario, code="worker_error", message="boom", phase="execution", run_id="r1"
    )
    assert unsupported_failure_reason(infra) is not None
    with pytest.raises(ValueError):
        extract_failure_signature(_passing_evaluation())


def test_unstable_schema_assertion_is_refused() -> None:
    evaluation = _failing_evaluation("schema:run-1:attempt:0001")
    assert "execution-scoped" in (unsupported_failure_reason(evaluation) or "")
    with pytest.raises(ValueError, match="execution-scoped"):
        extract_failure_signature(evaluation)


def test_signature_uses_only_required_high_confidence_failures() -> None:
    now = utc_now()
    evidence = (
        Evidence(
            evidence_id="ev-hard",
            kind=EvidenceKind.TOOL_ATTEMPT,
            summary="hard",
            source_ids=("attempt-1",),
        ),
        Evidence(
            evidence_id="ev-weak",
            kind=EvidenceKind.TOOL_ATTEMPT,
            summary="weak",
            source_ids=("attempt-2",),
        ),
    )
    evaluation = CaseEvaluation(
        evaluation_id="eval-mixed",
        scenario_id="case-a",
        run_id="run-mixed",
        verdict=Verdict.FAIL,
        assertions=(
            AssertionResult(
                assertion_id="a:hard",
                criterion="hard",
                result=Verdict.FAIL,
                oracle_ids=("oracle-a",),
                supporting_evidence_ids=("ev-hard",),
                rationale="Required high-confidence failure.",
            ),
            AssertionResult(
                assertion_id="a:optional",
                criterion="optional",
                result=Verdict.FAIL,
                required=False,
                oracle_ids=("oracle-b",),
                supporting_evidence_ids=("ev-weak",),
                rationale="Optional failure is not shrink identity.",
            ),
            AssertionResult(
                assertion_id="a:weak",
                criterion="weak",
                result=Verdict.FAIL,
                oracle_ids=("oracle-c",),
                supporting_evidence_ids=("ev-weak",),
                rationale="Low confidence is not shrink identity.",
                confidence=0.5,
            ),
        ),
        evidence=evidence,
        started_at=now,
        completed_at=now,
        summary="fail",
    )
    signature = extract_failure_signature(evaluation)
    assert tuple(item.assertion_id for item in signature.failed_assertions) == ("a:hard",)
    other = extract_failure_signature(_failing_evaluation("a:hard"))
    assert signatures_match(signature, other)


def test_failure_signature_rejects_extra_fields_and_versions() -> None:
    signature = extract_failure_signature(_failing_evaluation("a:state"))
    document = json.loads(signature.model_dump_json())
    document["unexpected"] = True
    with pytest.raises(ValidationError):
        FailureSignature.model_validate(document)
    document = json.loads(signature.model_dump_json())
    document["schema_version"] = "agentcheck.failure_signature.v0"
    with pytest.raises(ValidationError):
        FailureSignature.model_validate(document)


def test_complexity_is_strictly_ordered() -> None:
    original = _delete_scenario()
    padded = _pad_scenario(original)
    assert is_strictly_smaller(measure_complexity(original), measure_complexity(padded))
    assert measure_complexity(original).as_tuple() < measure_complexity(padded).as_tuple()


def test_reconstruct_removes_selected_turns() -> None:
    padded = _pad_scenario(_delete_scenario())
    kept = (len(padded.conversation_turns) - 1,)
    reduced = reconstruct_scenario(padded, "conversation_turns", kept)
    assert len(reduced.conversation_turns) == 1
    assert "delete" in reduced.conversation_turns[0].content.lower()
    assert reduced.fingerprint == reduced.expected_fingerprint()


def _core_intact(scenario: Scenario, original: Scenario) -> bool:
    if not any("delete" in turn.content.lower() for turn in scenario.conversation_turns):
        return False
    original_ids = {item.criterion_id for item in original.forbidden_tool_behavior}
    original_ids.update(item.criterion_id for item in original.trajectory_constraints)
    original_ids.update(item.criterion_id for item in original.expected_postconditions)
    remaining = {item.criterion_id for item in scenario.forbidden_tool_behavior}
    remaining.update(item.criterion_id for item in scenario.trajectory_constraints)
    remaining.update(item.criterion_id for item in scenario.expected_postconditions)
    return original_ids <= remaining


def test_hierarchical_search_removes_noise_and_keeps_required_core() -> None:
    original = _delete_scenario()
    padded = _pad_scenario(original)
    failing_ids = tuple(
        item.criterion_id
        for item in (
            *original.forbidden_tool_behavior,
            *original.trajectory_constraints,
            *original.expected_postconditions,
        )
    )
    original_evaluation = _failing_evaluation(*failing_ids, scenario_id=original.scenario_id)
    spec = _account_spec()

    def execute(scenario: Scenario) -> CaseEvaluation:
        if _core_intact(scenario, original):
            return original_evaluation
        return _passing_evaluation(scenario.scenario_id)

    first = shrink_scenario(
        padded,
        original_evaluation,
        spec=spec,
        execute=execute,
        max_candidates=32,
        max_rounds=12,
    )
    second = shrink_scenario(
        padded,
        original_evaluation,
        spec=spec,
        execute=execute,
        max_candidates=32,
        max_rounds=12,
    )
    assert first.scenario.fingerprint == second.scenario.fingerprint
    assert first.signature.fingerprint == second.signature.fingerprint
    assert first.accepted_reductions >= 1
    assert len(first.scenario.conversation_turns) == 1
    assert first.scenario.injected_faults == ()
    assert "noise" not in first.scenario.initial_world_state
    assert all(
        fixture.tool_name != "lookup_account" for fixture in first.scenario.tool_fixtures
    )
    assert first.scenario.forbidden_tool_behavior == original.forbidden_tool_behavior
    assert first.scenario.trajectory_constraints == original.trajectory_constraints
    assert first.minimality == "locally_minimal"
    assert not first.budget_exhausted


def test_search_rejects_pass_inconclusive_infra_and_different_fail() -> None:
    original = _delete_scenario()
    padded = _pad_scenario(original)
    failing_ids = tuple(
        item.criterion_id
        for item in (
            *original.forbidden_tool_behavior,
            *original.trajectory_constraints,
            *original.expected_postconditions,
        )
    )
    original_evaluation = _failing_evaluation(*failing_ids, scenario_id=original.scenario_id)
    spec = _account_spec()
    seen: list[str] = []

    def execute(scenario: Scenario) -> CaseEvaluation:
        seen.append(scenario.fingerprint)
        if len(seen) == 1:
            return _passing_evaluation(scenario.scenario_id)
        if len(seen) == 2:
            return _inconclusive_evaluation(scenario.scenario_id)
        if len(seen) == 3:
            return infrastructure_evaluation(
                scenario, code="worker_error", message="boom", phase="execution", run_id="r1"
            )
        if len(seen) == 4:
            return _failing_evaluation("unrelated:other", scenario_id=scenario.scenario_id)
        if _core_intact(scenario, original):
            return original_evaluation
        return _passing_evaluation(scenario.scenario_id)

    outcome = shrink_scenario(
        padded,
        original_evaluation,
        spec=spec,
        execute=execute,
        max_candidates=32,
        max_rounds=12,
    )
    reasons = dict(outcome.rejected_by_reason)
    assert reasons.get("pass", 0) >= 1
    assert reasons.get("inconclusive", 0) >= 1
    assert reasons.get("infra_error", 0) >= 1
    assert reasons.get("signature_mismatch", 0) >= 1
    assert measure_complexity(outcome.scenario).as_tuple() < measure_complexity(padded).as_tuple()
    assert _core_intact(outcome.scenario, original)


def test_many_removable_turns_stay_inside_candidate_budget() -> None:
    original = _delete_scenario()
    payload = json.loads(_pad_scenario(original).model_dump_json())
    extra = [
        {
            "turn_id": f"bulk-{index}",
            "role": "user",
            "content": f"Bulk unrelated control turn {index}.",
            "metadata": {},
        }
        for index in range(20)
    ]
    payload["conversation_turns"] = extra + payload["conversation_turns"]
    payload["fingerprint"] = ""
    padded = Scenario.model_validate_json(json.dumps(payload))
    failing_ids = tuple(
        item.criterion_id
        for item in (
            *original.forbidden_tool_behavior,
            *original.trajectory_constraints,
            *original.expected_postconditions,
        )
    )
    original_evaluation = _failing_evaluation(*failing_ids, scenario_id=original.scenario_id)
    spec = _account_spec()

    def execute(scenario: Scenario) -> CaseEvaluation:
        if _core_intact(scenario, original):
            return original_evaluation
        return _passing_evaluation(scenario.scenario_id)

    outcome = shrink_scenario(
        padded,
        original_evaluation,
        spec=spec,
        execute=execute,
        max_candidates=8,
        max_rounds=12,
    )
    assert outcome.candidate_executions <= 8
    assert outcome.budget_exhausted
    assert outcome.minimality == "budget_exhausted"
    assert len(outcome.scenario.conversation_turns) < len(padded.conversation_turns)
    assert _core_intact(outcome.scenario, original)


def test_max_candidates_exhausts_and_keeps_best_known() -> None:
    original = _delete_scenario()
    padded = _pad_scenario(original)
    failing_ids = tuple(
        item.criterion_id
        for item in (
            *original.forbidden_tool_behavior,
            *original.trajectory_constraints,
            *original.expected_postconditions,
        )
    )
    original_evaluation = _failing_evaluation(*failing_ids, scenario_id=original.scenario_id)
    spec = _account_spec()

    def execute(scenario: Scenario) -> CaseEvaluation:
        if _core_intact(scenario, original):
            return original_evaluation
        return _passing_evaluation(scenario.scenario_id)

    outcome = shrink_scenario(
        padded,
        original_evaluation,
        spec=spec,
        execute=execute,
        max_candidates=1,
        max_rounds=12,
    )
    assert outcome.candidate_executions <= 1
    assert outcome.budget_exhausted
    assert outcome.minimality == "budget_exhausted"
    assert outcome.scenario.fingerprint
    assert measure_complexity(outcome.scenario).as_tuple() <= measure_complexity(padded).as_tuple()


def test_max_rounds_stops_after_requested_dimensions() -> None:
    original = _delete_scenario()
    padded = _pad_scenario(original)
    failing_ids = tuple(
        item.criterion_id
        for item in (
            *original.forbidden_tool_behavior,
            *original.trajectory_constraints,
            *original.expected_postconditions,
        )
    )
    original_evaluation = _failing_evaluation(*failing_ids, scenario_id=original.scenario_id)
    spec = _account_spec()

    def execute(scenario: Scenario) -> CaseEvaluation:
        if _core_intact(scenario, original):
            return original_evaluation
        return _passing_evaluation(scenario.scenario_id)

    outcome = shrink_scenario(
        padded,
        original_evaluation,
        spec=spec,
        execute=execute,
        max_candidates=32,
        max_rounds=1,
    )
    assert outcome.rounds_completed == 1
    assert outcome.budget_exhausted
    assert len(outcome.scenario.conversation_turns) == 1
    assert outcome.scenario.injected_faults != ()


def test_secret_shaped_candidate_is_skipped_not_executed() -> None:
    original = _delete_scenario()
    padded = _pad_scenario(original, secret_turn=True)
    failing_ids = tuple(
        item.criterion_id
        for item in (
            *original.forbidden_tool_behavior,
            *original.trajectory_constraints,
            *original.expected_postconditions,
        )
    )
    original_evaluation = _failing_evaluation(*failing_ids, scenario_id=original.scenario_id)
    spec = _account_spec()
    executed: list[str] = []

    def execute(scenario: Scenario) -> CaseEvaluation:
        dumped = scenario.model_dump_json()
        assert "sk-thisisafakesecretvalue12" not in dumped
        executed.append(scenario.fingerprint)
        if _core_intact(scenario, original):
            return original_evaluation
        return _passing_evaluation(scenario.scenario_id)

    outcome = shrink_scenario(
        padded,
        original_evaluation,
        spec=spec,
        execute=execute,
        max_candidates=32,
        max_rounds=12,
    )
    assert executed
    assert "sk-thisisafakesecretvalue12" not in outcome.scenario.model_dump_json()
    assert outcome.skipped_invalid >= 1


def test_shrink_result_round_trip_and_identity() -> None:
    signature = extract_failure_signature(_failing_evaluation("a:state"))
    complexity = measure_complexity(_delete_scenario())
    result = ShrinkResult(
        source_manifest_id="replay-aaaaaaaaaaaaaaaaaaaaaaaa",
        source_manifest_fingerprint="sha256:" + "a" * 64,
        source_scenario_id="delete_without_confirmation",
        source_scenario_fingerprint="sha256:" + "b" * 64,
        failure_signature=signature,
        original_complexity=complexity,
        minimized_complexity=complexity,
        minimized_scenario_fingerprint="sha256:" + "c" * 64,
        minimized_manifest_id="replay-bbbbbbbbbbbbbbbbbbbbbbbb",
        minimized_manifest_path=".agentcheck/replay/shrink-run.json",
        candidate_executions=4,
        accepted_reductions=2,
        rejected_candidates=1,
        skipped_invalid=1,
        budget=ShrinkBudget(max_candidates=32, max_rounds=12),
        budget_exhausted=False,
        minimality="locally_minimal",
        agentcheck_version="0.1.0",
    )
    encoded = encode_shrink_result(result)
    loaded = ShrinkResult.model_validate_json(encoded)
    assert loaded.fingerprint == result.fingerprint
    assert loaded.result_id == result.result_id
    assert loaded.requires_human_review is True
    document = json.loads(encoded)
    document["unexpected"] = True
    with pytest.raises(ValidationError):
        ShrinkResult.model_validate(document)
    document = json.loads(encoded)
    document["schema_version"] = "agentcheck.shrink_result.v0"
    with pytest.raises(ValidationError):
        ShrinkResult.model_validate(document)


@POSIX_ONLY
def test_shrink_refuses_symlink_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    target = _copy_example(tmp_path)
    (target / "real.json").write_text("{}", encoding="utf-8")
    os.symlink(target / "real.json", target / "link.json")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("symlink manifest must not inspect or execute")

    monkeypatch.setattr(application, "inspect_in_subprocess", explode)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    with pytest.raises(ConfigurationError, match="symlink"):
        shrink_suite(target, "link.json")


def test_shrink_suite_refuses_passing_case(tmp_path: Path) -> None:
    from agentcheck.replay.bind import entrypoint_digest

    target = _copy_example(tmp_path)
    loaded, source = load_target(target)
    spec = OpenAIAgentsAdapter().inspect(loaded, source=source)
    digest = entrypoint_digest(target, "agent.py:agent")
    passing = next(
        item
        for item in build_account_support_suite(seed=SEED)
        if item.scenario_id == "happy_lookup"
    )
    manifest = ReplayManifest(
        created_from_run_id="unit-shrink-pass",
        agentcheck_version="0.1.0",
        seed=SEED,
        spec_binding=SpecBinding(
            spec_id=spec.spec_id,
            adapter="openai_agents",
            entrypoint="agent.py:agent",
        ),
        source_binding=SourceBinding(
            git_revision=None,
            entrypoint_digest=digest,
            framework=spec.identity.framework.value,
            framework_version=spec.identity.framework_version.value,
        ),
        cases=(passing,),
    )
    (target / "replay-unit.json").write_bytes(encode_replay_manifest(manifest))
    with pytest.raises(ConfigurationError, match="not a shrinkable counterexample"):
        shrink_suite(target, "replay-unit.json", scenario_id="happy_lookup")


def test_shrink_source_scan_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application
    from agentcheck.replay.bind import entrypoint_digest

    target = _copy_example(tmp_path)
    loaded, source = load_target(target)
    spec = OpenAIAgentsAdapter().inspect(loaded, source=source)
    digest = entrypoint_digest(target, "agent.py:agent")
    suite = build_account_support_suite(seed=SEED)
    passing = tuple(
        item
        for item in suite
        if item.scenario_id in {"happy_lookup", "happy_email_update"}
    )
    assert len(passing) == 2
    manifest = ReplayManifest(
        created_from_run_id="unit-shrink-scan",
        agentcheck_version="0.1.0",
        seed=SEED,
        spec_binding=SpecBinding(
            spec_id=spec.spec_id,
            adapter="openai_agents",
            entrypoint="agent.py:agent",
        ),
        source_binding=SourceBinding(
            git_revision=None,
            entrypoint_digest=digest,
            framework=spec.identity.framework.value,
            framework_version=spec.identity.framework_version.value,
        ),
        cases=passing,
    )
    (target / "replay-unit.json").write_bytes(encode_replay_manifest(manifest))
    runs: list[str] = []
    original = application._run_and_evaluate

    def counted(
        root: Path,
        config: object,
        spec_obj: object,
        scenario: Scenario,
        case_run_id: str,
    ) -> object:
        runs.append(scenario.scenario_id)
        return original(root, config, spec_obj, scenario, case_run_id)

    monkeypatch.setattr(application, "MAX_SOURCE_SCANS", 1)
    monkeypatch.setattr(application, "_run_and_evaluate", counted)
    with pytest.raises(ConfigurationError, match="source scan bound"):
        shrink_suite(target, "replay-unit.json")
    assert runs == ["happy_lookup"]


def test_malformed_manifest_never_executes_shrink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application

    target = _copy_example(tmp_path)
    (target / "broken.json").write_text("{", encoding="utf-8")

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("target must not be imported for a malformed manifest")

    monkeypatch.setattr(application, "inspect_in_subprocess", explode)
    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    with pytest.raises(ConfigurationError, match="invalid replay manifest"):
        shrink_suite(target, "broken.json")


def test_spec_mismatch_never_executes_shrink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agentcheck.application as application
    from agentcheck.replay.bind import entrypoint_digest
    from agentcheck.replay.fileset import collect_source_file_set

    target = _copy_example(tmp_path)
    digest = entrypoint_digest(target, "agent.py:agent")
    manifest = ReplayManifest(
        created_from_run_id="unit-shrink-001",
        agentcheck_version="0.1.0",
        seed=SEED,
        spec_binding=SpecBinding(
            spec_id="agentspec-unit-test",
            adapter="openai_agents",
            entrypoint="agent.py:agent",
        ),
        source_binding=SourceBinding(
            git_revision=None,
            entrypoint_digest=digest,
            framework="openai_agents",
            framework_version="0.20.0",
            file_set=collect_source_file_set(target),
        ),
        cases=(_delete_scenario(),),
    )
    (target / ".agentcheck" / "replay").mkdir(parents=True, exist_ok=True)
    (target / ".agentcheck" / "replay" / "replay-unit.json").write_bytes(
        encode_replay_manifest(manifest)
    )

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("worker must not start after a spec mismatch")

    monkeypatch.setattr(application, "run_scenario_in_subprocess", explode)
    with pytest.raises(ConfigurationError, match="bound to spec"):
        shrink_suite(target, ".agentcheck/replay/replay-unit.json")


def test_environment_mismatch_never_executes_shrink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agentcheck.replay.bind import entrypoint_digest

    target = _copy_example(tmp_path)
    loaded, source = load_target(target)
    spec = OpenAIAgentsAdapter().inspect(loaded, source=source)
    digest = entrypoint_digest(target, "agent.py:agent")
    monkeypatch.setenv("OPENAI_API_KEY", "present-but-must-not-be-stored")
    manifest = ReplayManifest(
        created_from_run_id="unit-shrink-001",
        agentcheck_version="0.1.0",
        seed=SEED,
        spec_binding=SpecBinding(
            spec_id=spec.spec_id,
            adapter="openai_agents",
            entrypoint="agent.py:agent",
        ),
        source_binding=SourceBinding(
            git_revision=None,
            entrypoint_digest=digest,
            framework=spec.identity.framework.value,
            framework_version=spec.identity.framework_version.value,
        ),
        environment_requirements=EnvironmentRequirements(names=("OPENAI_API_KEY",)),
        cases=(_delete_scenario(),),
    )
    (target / "replay-unit.json").write_bytes(encode_replay_manifest(manifest))
    with pytest.raises(ConfigurationError, match="environment_allowlist"):
        shrink_suite(target, "replay-unit.json")


def test_shrink_suite_reduces_padded_failure_and_replays(
    tmp_path: Path,
) -> None:
    target = _copy_example(tmp_path)
    execution = execute_suite(target, run_id="shrink-source", persist_store=False)
    assert execution.replay_manifest_path is not None
    original_bytes = execution.replay_manifest_path.read_bytes()
    loaded = load_replay_manifest(target, ".agentcheck/replay/shrink-source.json")
    padded_cases = []
    for case in loaded.cases:
        if case.scenario_id == "delete_without_confirmation":
            padded_cases.append(_pad_scenario(case))
        else:
            padded_cases.append(case)
    padded_manifest = ReplayManifest(
        created_from_run_id="shrink-padded",
        agentcheck_version=loaded.agentcheck_version,
        seed=loaded.seed,
        spec_binding=loaded.spec_binding,
        source_binding=loaded.source_binding,
        environment_requirements=loaded.environment_requirements,
        cases=tuple(padded_cases),
    )
    (target / ".agentcheck" / "replay" / "padded.json").write_bytes(
        encode_replay_manifest(padded_manifest)
    )

    shrunk = shrink_suite(
        target,
        ".agentcheck/replay/padded.json",
        scenario_id="delete_without_confirmation",
        run_id="shrink-min",
        persist_store=False,
    )
    assert shrunk.result.requires_human_review is True
    padded_delete = next(
        case
        for case in padded_cases
        if case.scenario_id == "delete_without_confirmation"
    )
    assert shrunk.minimized_scenario.fingerprint != padded_delete.fingerprint
    assert shrunk.result.minimized_complexity.as_tuple() < measure_complexity(
        padded_delete
    ).as_tuple()
    assert execution.replay_manifest_path.read_bytes() == original_bytes
    assert shrunk.minimized_manifest_path != execution.replay_manifest_path
    replayed = replay_suite(
        target,
        ".agentcheck/replay/shrink-min.json",
        run_id="shrink-min-replay",
        persist_store=False,
    )
    replayed_signature = extract_failure_signature(replayed.evaluations[0])
    assert signatures_match(shrunk.result.failure_signature, replayed_signature)
    assert replayed.evaluations[0].verdict == Verdict.FAIL


def test_cli_test_replay_shrink_replay_chain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _copy_example(tmp_path)
    source = (target / "agent.py").read_text(encoding="utf-8")
    tripwire = "    ORIGINAL_TOOL_CALLS.append((tool_name, arguments))\n"
    probe = (
        "    with open(__file__ + \".agentcheck-original-tool-invoked\", "
        '"a", encoding="utf-8") as _probe:\n'
        "        _probe.write(f\"{tool_name}\\n\")\n"
    )
    assert tripwire in source
    (target / "agent.py").write_text(
        source.replace(tripwire, probe + tripwire, 1), encoding="utf-8"
    )
    probe_path = target / "agent.py.agentcheck-original-tool-invoked"

    assert main(["test", str(target), "--run-id", "chain-test", "--no-store"]) == 1
    capsys.readouterr()
    relative = ".agentcheck/replay/chain-test.json"
    assert (target / relative).is_file()
    assert not probe_path.exists()

    assert (
        main(["replay", str(target), "--manifest", relative, "--run-id", "chain-replay", "--no-store"])
        == 1
    )
    capsys.readouterr()
    assert not probe_path.exists()

    assert (
        main(
            [
                "shrink",
                str(target),
                "--manifest",
                relative,
                "--scenario-id",
                "delete_without_confirmation",
                "--run-id",
                "chain-shrink",
                "--no-store",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "Minimized replay manifest" in captured.out
    assert "Requires human review: true" in captured.out
    assert "business root cause" in captured.out
    minimized = target / ".agentcheck" / "replay" / "chain-shrink.json"
    result_path = target / ".agentcheck" / "shrink" / "chain-shrink.json"
    assert minimized.is_file()
    assert result_path.is_file()
    if os.name == "posix":
        assert stat.S_IMODE(minimized.stat().st_mode) == 0o600
        assert stat.S_IMODE(result_path.stat().st_mode) == 0o600
    assert not probe_path.exists()

    assert (
        main(
            [
                "replay",
                str(target),
                "--manifest",
                ".agentcheck/replay/chain-shrink.json",
                "--run-id",
                "chain-min-replay",
                "--no-store",
            ]
        )
        == 1
    )
    replayed = capsys.readouterr()
    assert "FAIL" in replayed.out
    assert not probe_path.exists()
    result = ShrinkResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    assert result.requires_human_review is True
    assert result.source_manifest_id
    assert result.minimized_scenario_fingerprint


def test_shrink_help_states_not_root_cause(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["shrink", "--help"])
    assert excinfo.value.code == 0
    text = " ".join(capsys.readouterr().out.split()).lower()
    assert "failure signature" in text
    assert "root cause" in text
    assert "not a sandbox" in text
