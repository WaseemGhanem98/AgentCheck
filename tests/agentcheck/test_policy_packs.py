from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import agentcheck.application as application
from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.cli import main
from agentcheck.config import AgentCheckConfig, load_config
from agentcheck.domain import (
    CanonicalEvent,
    CanonicalEventType,
    CanonicalRun,
    OracleProvenance,
    OracleStrength,
    RunTermination,
    Scenario,
    SourceKind,
    ToolAttempt,
    ToolOutcome,
    ToolOutcomeStatus,
    UsageMetrics,
    Verdict,
    utc_now,
)
from agentcheck.errors import ConfigurationError
from agentcheck.evaluate import evaluate_run
from agentcheck.generate import build_frozen_suite, encode_frozen_suite, lint_scenario
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.inspect import load_target
from agentcheck.policies import (
    BUILTIN_POLICY_PACKS,
    CONFIRM_BEFORE_DESTRUCTIVE_V1,
    NO_FABRICATED_SUCCESS_V1,
    POLICY_PACK_CONTRACT_VERSION,
    PolicyPack,
    PolicyPackRegistry,
    PolicyRule,
    PolicyRuleKind,
    apply_policy_packs,
    attach_declared_policies,
    policy_oracle,
    resolve_policy_packs,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
SEED = 1729
EXPECTED_FAILURES = {
    "delete_without_confirmation",
    "ambiguous_account_selection",
    "destructive_ambiguous_timeout",
    "fabricated_email_success",
    "duplicate_email_update",
}


def _example_spec() -> Any:
    target, source = load_target(EXAMPLE)
    return OpenAIAgentsAdapter().inspect(target, source=source)


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "account_agent"
    shutil.copytree(
        EXAMPLE,
        target,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    return target


def _parent(scenario_id: str) -> Scenario:
    return next(
        scenario
        for scenario in build_account_support_suite(seed=SEED)
        if scenario.scenario_id == scenario_id
    )


def _without_confirmation_gate(scenario: Scenario) -> Scenario:
    data = json.loads(scenario.model_dump_json())
    for turn in data["conversation_turns"]:
        metadata = turn.get("metadata") or {}
        metadata.pop("explicit_confirmation", None)
        turn["metadata"] = metadata
    for item in data["required_tool_behavior"]:
        item["confirmation_required_before_call"] = False
    data["trajectory_constraints"] = [
        item
        for item in data.get("trajectory_constraints") or []
        if item.get("kind") != "confirmation_before_tool"
    ]
    data["fingerprint"] = ""
    return Scenario.model_validate_json(json.dumps(data))


def _delete_run(scenario: Scenario, *, confirmation: bool = False) -> CanonicalRun:
    now = utc_now()
    final_state = json.loads(json.dumps(scenario.initial_world_state))
    final_state["accounts"]["acct_123"]["exists"] = False
    return CanonicalRun(
        run_id="run-policy",
        scenario_id=scenario.scenario_id,
        target_id="target",
        started_at=now,
        ended_at=utc_now(),
        termination=RunTermination.COMPLETED,
        events=(
            CanonicalEvent(
                event_id="event-1",
                run_id="run-policy",
                sequence=1,
                event_type=CanonicalEventType.USER_TURN,
                timestamp=now,
                metadata={"explicit_confirmation": confirmation},
            ),
            CanonicalEvent(
                event_id="event-2",
                run_id="run-policy",
                sequence=2,
                event_type=CanonicalEventType.TOOL_ATTEMPT,
                timestamp=now,
            ),
            CanonicalEvent(
                event_id="event-3",
                run_id="run-policy",
                sequence=3,
                event_type=CanonicalEventType.TOOL_RESULT,
                timestamp=now,
            ),
        ),
        tool_attempts=(
            ToolAttempt(
                attempt_id="attempt-1",
                event_id="event-2",
                tool_name="delete_account",
                arguments={"account_id": "acct_123"},
                sequence=2,
                timestamp=now,
                state_changing=True,
                destructive=True,
            ),
        ),
        tool_outcomes=(
            ToolOutcome(
                outcome_id="outcome-1",
                attempt_id="attempt-1",
                event_id="event-3",
                tool_name="delete_account",
                status=ToolOutcomeStatus.SUCCESS,
            ),
        ),
        initial_world_state=scenario.initial_world_state,
        final_world_state=final_state,
        final_output="Deleted.",
        usage=UsageMetrics(),
        latency_ms=5,
    )


def test_builtin_packs_are_versioned_forbid_extra_and_deterministic() -> None:
    first = {pack.pack_id: pack.model_dump(mode="json") for pack in BUILTIN_POLICY_PACKS}
    second = {
        pack.pack_id: PolicyPack.model_validate_json(pack.model_dump_json()).model_dump(
            mode="json"
        )
        for pack in BUILTIN_POLICY_PACKS
    }
    assert first == second
    assert {pack.pack_id for pack in BUILTIN_POLICY_PACKS} == {
        "confirm_before_destructive_v1",
        "no_retry_after_ambiguous_timeout_v1",
        "no_fabricated_success_v1",
        "no_duplicate_side_effect_v1",
    }
    for pack in BUILTIN_POLICY_PACKS:
        assert pack.schema_version == POLICY_PACK_CONTRACT_VERSION
        restored = PolicyPack.model_validate_json(pack.model_dump_json())
        assert restored == pack
    with pytest.raises(ValidationError):
        PolicyPack.model_validate(
            {**CONFIRM_BEFORE_DESTRUCTIVE_V1.model_dump(mode="json"), "unexpected": True}
        )
    with pytest.raises(ValidationError):
        PolicyRule.model_validate(
            {
                "rule_id": "x",
                "kind": PolicyRuleKind.CONFIRMATION_BEFORE_TOOL.value,
                "description": "needs a tool",
            }
        )


def test_unknown_and_malformed_packs_fail_closed(tmp_path: Path) -> None:
    root = _copy_example(tmp_path)
    registry = PolicyPackRegistry()
    with pytest.raises(ConfigurationError, match="unknown policy pack"):
        registry.resolve("not_a_real_pack_v1", root=root)
    with pytest.raises(ConfigurationError, match="unknown policy pack"):
        resolve_policy_packs(root, AgentCheckConfig(policy_packs=("missing_v1",)))

    escaped = tmp_path / "escape.json"
    escaped.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="safe relative path"):
        registry.resolve("../escape.json", root=root)

    destination = root / "packs"
    destination.mkdir()
    hostile = destination / "hostile.json"
    hostile.write_text(
        json.dumps({"schema_version": "agentcheck.policy_pack.v1", "unexpected": True}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="invalid policy pack"):
        registry.resolve("packs/hostile.json", root=root)

    valid = destination / "local.json"
    valid.write_text(
        json.dumps(NO_FABRICATED_SUCCESS_V1.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    loaded = registry.resolve("packs/local.json", root=root)
    assert loaded.pack_id == NO_FABRICATED_SUCCESS_V1.pack_id


def test_attached_pack_preserves_absent_context_and_withheld_oracle_strength() -> None:
    parent = _without_confirmation_gate(_parent("confirmed_delete"))
    run = _delete_run(parent)
    baseline = evaluate_run(parent, run)
    assert baseline.verdict == Verdict.PASS

    declared = apply_policy_packs(
        parent, (CONFIRM_BEFORE_DESTRUCTIVE_V1,), declared=True
    )
    undeclared = apply_policy_packs(
        parent, (CONFIRM_BEFORE_DESTRUCTIVE_V1,), declared=False
    )
    assert declared.fingerprint != parent.fingerprint
    assert undeclared.fingerprint != declared.fingerprint
    declared_oracle = next(
        oracle
        for oracle in declared.oracle_provenance
        if oracle.oracle_id.startswith("policy:")
    )
    undeclared_oracle = next(
        oracle
        for oracle in undeclared.oracle_provenance
        if oracle.oracle_id.startswith("policy:")
    )
    assert declared_oracle.supports_hard_failure is True
    assert declared_oracle.strength is OracleStrength.VERSIONED_POLICY
    assert declared_oracle.confidence == 1.0
    assert undeclared_oracle.supports_hard_failure is False
    assert undeclared_oracle.confidence < 0.8

    # Attaching a rule cannot manufacture deliberately withheld scenario context.
    assert evaluate_run(declared, _delete_run(declared)).verdict == Verdict.INCONCLUSIVE
    assert evaluate_run(undeclared, _delete_run(undeclared)).verdict == Verdict.INCONCLUSIVE
    # Preserve the strong/weak policy-oracle control using an authored refusal,
    # rather than relying on absent context to imply withholding.
    refusal = _parent("delete_without_confirmation").model_copy(update={"trajectory_constraints": ()})
    declared_refusal = apply_policy_packs(refusal, (CONFIRM_BEFORE_DESTRUCTIVE_V1,), declared=True)
    undeclared_refusal = apply_policy_packs(refusal, (CONFIRM_BEFORE_DESTRUCTIVE_V1,), declared=False)
    policy_id = next(c.criterion_id for c in declared_refusal.trajectory_constraints if ":policy:" in c.criterion_id)
    assert next(a for a in evaluate_run(declared_refusal, _delete_run(declared_refusal)).assertions if a.assertion_id == policy_id).result == Verdict.FAIL
    inconclusive = evaluate_run(undeclared_refusal, _delete_run(undeclared_refusal))
    assert next(a for a in inconclusive.assertions if a.assertion_id == policy_id).result == Verdict.INCONCLUSIVE
    assert any(
        "authoritative high-confidence oracle evidence" in assertion.missing_evidence
        for assertion in inconclusive.assertions
        if assertion.result == Verdict.INCONCLUSIVE
    )


def test_pack_cannot_grant_llm_oracle_authority() -> None:
    with pytest.raises(ValidationError):
        PolicyRule.model_validate(
            {
                "rule_id": "llm",
                "kind": PolicyRuleKind.NO_FABRICATED_SUCCESS.value,
                "description": "hostile",
                "oracle_strength": "llm_inference",
                "supports_hard_failure": True,
            }
        )
    oracle = policy_oracle(
        CONFIRM_BEFORE_DESTRUCTIVE_V1,
        CONFIRM_BEFORE_DESTRUCTIVE_V1.rules[0],
        declared=True,
    )
    assert oracle.strength is not OracleStrength.LLM_INFERENCE
    with pytest.raises(ValidationError):
        OracleProvenance(
            oracle_id="hostile",
            strength=OracleStrength.LLM_INFERENCE,
            source="hostile pack",
            confidence=1.0,
            evidence_ids=("hostile",),
            supports_hard_failure=True,
        )


def test_apply_is_deterministic_and_idempotent() -> None:
    parent = _parent("delete_without_confirmation")
    packs = (CONFIRM_BEFORE_DESTRUCTIVE_V1, NO_FABRICATED_SUCCESS_V1)
    first = apply_policy_packs(parent, packs, declared=True)
    second = apply_policy_packs(parent, packs, declared=True)
    again = apply_policy_packs(first, packs, declared=True)
    assert first == second
    assert first.fingerprint == again.fingerprint
    spec = _example_spec()
    assert not lint_scenario(first, spec)


def test_config_policy_packs_are_optional_and_keep_v1(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("agent = object()\n", encoding="utf-8")
    (tmp_path / "agentcheck.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.config.v1",
                "entrypoint": "agent.py:agent",
            }
        ),
        encoding="utf-8",
    )
    _, config = load_config(tmp_path)
    assert config.policy_packs is None
    assert AgentCheckConfig().policy_packs is None
    loaded = AgentCheckConfig(policy_packs=("confirm_before_destructive_v1",))
    assert loaded.policy_packs == ("confirm_before_destructive_v1",)
    with pytest.raises(ValueError, match="safe relative path"):
        AgentCheckConfig(policy_packs=("../escape.json",))
    with pytest.raises(ValidationError):
        AgentCheckConfig.model_validate(
            {
                "schema_version": "agentcheck.config.v1",
                "policy_packs": ["confirm_before_destructive_v1"],
                "unexpected": True,
            }
        )


def test_inspect_reports_declared_packs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _copy_example(tmp_path)
    document = json.loads((target / "agentcheck.json").read_text(encoding="utf-8"))
    document["policy_packs"] = ["confirm_before_destructive_v1"]
    (target / "agentcheck.json").write_text(json.dumps(document), encoding="utf-8")

    assert main(["inspect", str(target)]) == 0
    output = capsys.readouterr().out
    assert "Declared policy packs:" in output
    assert "confirm_before_destructive_v1" in output

    _, _, result = application.inspect_target(target)
    spec = result.require_value()
    assert spec.policies.items
    property_ = spec.policies.items[0]
    assert property_.source.kind is SourceKind.DECLARED_POLICY
    assert property_.authoritative is True
    assert property_.inferred is False
    attached = attach_declared_policies(spec, (CONFIRM_BEFORE_DESTRUCTIVE_V1,))
    assert attached.spec_id == spec.spec_id


def test_unknown_pack_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = _copy_example(tmp_path)
    assert main(["generate", str(target), "--policy-pack", "missing_v1"]) == 2
    assert "unknown policy pack" in capsys.readouterr().err


def test_generate_with_packs_changes_identity_and_records_provenance(
    tmp_path: Path,
) -> None:
    spec = _example_spec()
    baseline = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    packed = build_frozen_suite(
        spec,
        AgentCheckConfig(),
        seed=SEED,
        policy_packs=(CONFIRM_BEFORE_DESTRUCTIVE_V1,),
    )
    assert packed.fingerprint != baseline.fingerprint
    assert packed.provenance.policy_packs == ("confirm_before_destructive_v1",)
    dumped = json.loads(encode_frozen_suite(baseline))
    assert "policy_packs" not in dumped["provenance"]
    oracles = [
        oracle.oracle_id
        for case in packed.cases
        for oracle in case.scenario.oracle_provenance
        if oracle.oracle_id.startswith("policy:confirm_before_destructive_v1")
    ]
    assert oracles
    assert all(
        oracle.supports_hard_failure
        for case in packed.cases
        for oracle in case.scenario.oracle_provenance
        if oracle.oracle_id.startswith("policy:confirm_before_destructive_v1")
    )


def test_cli_generate_policy_pack_and_test_never_invokes_original_handlers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _copy_example(tmp_path)
    source_path = target / "agent.py"
    source = source_path.read_text(encoding="utf-8")
    tripwire = "    ORIGINAL_TOOL_CALLS.append((tool_name, arguments))\n"
    probe = """    with open(
        __file__ + ".agentcheck-original-tool-invoked", "a", encoding="utf-8"
    ) as _tool_probe:
        _tool_probe.write(f"{tool_name}\\n")
"""
    assert source.count(tripwire) == 1
    source_path.write_text(source.replace(tripwire, probe + tripwire, 1), encoding="utf-8")
    probe_path = Path(f"{source_path}.agentcheck-original-tool-invoked")

    assert (
        main(
            [
                "generate",
                str(target),
                "--seed",
                "1729",
                "--policy-pack",
                "confirm_before_destructive_v1",
            ]
        )
        == 0
    )
    assert "Frozen suite written." in capsys.readouterr().out
    execution = application.execute_suite(target, run_id="policy-e2e")
    assert not probe_path.exists()
    assert all(
        evaluation.verdict != Verdict.INFRA_ERROR for evaluation in execution.evaluations
    )
    failed = {
        evaluation.scenario_id
        for evaluation in execution.evaluations
        if evaluation.verdict == Verdict.FAIL and evaluation.scenario_id in EXPECTED_FAILURES
    }
    assert EXPECTED_FAILURES <= failed


def test_phase1_verdicts_unchanged_without_declared_packs(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    execution = application.execute_suite(target, run_id="policy-default")
    failed = {
        evaluation.scenario_id
        for evaluation in execution.evaluations
        if evaluation.verdict == Verdict.FAIL
    }
    assert failed == EXPECTED_FAILURES
    assert all(
        evaluation.verdict != Verdict.INCONCLUSIVE
        for evaluation in execution.evaluations
    )
