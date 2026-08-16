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
from agentcheck.config import AgentCheckConfig
from agentcheck.domain import (
    CanonicalEvent,
    CanonicalEventType,
    CanonicalRun,
    ConversationRole,
    RunTermination,
    Scenario,
    ToolAttempt,
    ToolOutcome,
    ToolOutcomeStatus,
    UsageMetrics,
    Verdict,
    utc_now,
)
from agentcheck.evaluate import evaluate_run
from agentcheck.generate import (
    DEFAULT_SUITE_FILENAME,
    MUTATION_CONTRACT_VERSION,
    CaseLineage,
    CaseOrigin,
    FrozenSuite,
    MutationKind,
    build_frozen_suite,
    build_workflow_mutations,
    encode_frozen_suite,
    inherited_hard_failure_oracles,
    lint_scenario,
    load_frozen_suite,
    mutate_scenario,
    unsupported_mutation_reasons,
)
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.inspect import load_target


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
SEED = 1729
EXPECTED_BUILTIN_FAILURES = {
    "delete_without_confirmation",
    "ambiguous_account_selection",
    "destructive_ambiguous_timeout",
    "fabricated_email_success",
    "duplicate_email_update",
}


def _example_spec() -> Any:
    target, source = load_target(EXAMPLE)
    return OpenAIAgentsAdapter().inspect(target, source=source)


def _parent(scenario_id: str) -> Scenario:
    return next(
        scenario
        for scenario in build_account_support_suite(seed=SEED)
        if scenario.scenario_id == scenario_id
    )


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "account_agent"
    shutil.copytree(
        EXAMPLE,
        target,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    return target


def _event(
    run_id: str, sequence: int, event_type: CanonicalEventType, **metadata: object
) -> CanonicalEvent:
    payload = metadata.pop("payload", {})
    return CanonicalEvent(
        event_id=f"event-{sequence}",
        run_id=run_id,
        sequence=sequence,
        event_type=event_type,
        timestamp=utc_now(),
        payload=payload if isinstance(payload, dict) else {},
        metadata={key: value for key, value in metadata.items() if key != "payload"},
    )


def _run(
    scenario: Scenario,
    *,
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
    output: str = "ok",
    confirmation: bool = False,
    extra_attempts: tuple[ToolAttempt, ...] = (),
    extra_outcomes: tuple[ToolOutcome, ...] = (),
    final_state: dict[str, Any] | None = None,
) -> CanonicalRun:
    now = utc_now()
    events = [
        _event(
            "run-mut",
            1,
            CanonicalEventType.USER_TURN,
            explicit_confirmation=True if confirmation else False,
        )
    ]
    attempts: list[ToolAttempt] = []
    outcomes: list[ToolOutcome] = []
    sequence = 2
    if tool_name is not None:
        attempt = ToolAttempt(
            attempt_id="attempt-1",
            event_id=f"event-{sequence}",
            tool_name=tool_name,
            arguments=arguments or {},
            sequence=sequence,
            timestamp=now,
            state_changing=True,
        )
        events.append(_event("run-mut", sequence, CanonicalEventType.TOOL_ATTEMPT))
        sequence += 1
        outcome = ToolOutcome(
            outcome_id="outcome-1",
            attempt_id="attempt-1",
            event_id=f"event-{sequence}",
            tool_name=tool_name,
            status=ToolOutcomeStatus.SUCCESS,
        )
        events.append(_event("run-mut", sequence, CanonicalEventType.TOOL_RESULT))
        attempts.append(attempt)
        outcomes.append(outcome)
        sequence += 1
    attempts.extend(extra_attempts)
    outcomes.extend(extra_outcomes)
    for extra in extra_attempts:
        events.append(
            CanonicalEvent(
                event_id=extra.event_id,
                run_id="run-mut",
                sequence=extra.sequence,
                event_type=CanonicalEventType.TOOL_ATTEMPT,
                timestamp=now,
                payload={},
            )
        )
    for extra in extra_outcomes:
        events.append(
            CanonicalEvent(
                event_id=extra.event_id,
                run_id="run-mut",
                sequence=max(event.sequence for event in events) + 1,
                event_type=CanonicalEventType.TOOL_RESULT,
                timestamp=now,
                payload={},
            )
        )
    events.sort(key=lambda event: event.sequence)
    return CanonicalRun(
        run_id="run-mut",
        scenario_id=scenario.scenario_id,
        target_id="target",
        started_at=now,
        ended_at=utc_now(),
        termination=RunTermination.COMPLETED,
        events=tuple(events),
        tool_attempts=tuple(attempts),
        tool_outcomes=tuple(outcomes),
        initial_world_state=scenario.initial_world_state,
        final_world_state=final_state or scenario.initial_world_state,
        final_output=output,
        usage=UsageMetrics(),
        latency_ms=5,
    )


def test_each_supported_kind_lints_and_differs_from_its_parent() -> None:
    spec = _example_spec()
    parents = {
        MutationKind.WITHHOLD_CONFIRMATION: _parent("confirmed_delete"),
        MutationKind.DUPLICATE_REQUEST: _parent("happy_lookup"),
        MutationKind.AMBIGUOUS_IDENTIFIER: _parent("happy_lookup"),
        MutationKind.REORDER_DIALOGUE: _parent("confirmed_delete"),
        MutationKind.INTERLEAVE_UNRELATED_TURN: _parent("happy_lookup"),
    }
    fingerprints: set[str] = {parent.fingerprint for parent in parents.values()}
    for kind, parent in parents.items():
        generated = mutate_scenario(parent, kind, seed=SEED)
        assert generated is not None, kind
        mutant = generated.scenario
        assert not lint_scenario(mutant, spec)
        assert mutant.fingerprint != parent.fingerprint
        assert mutant.fingerprint not in fingerprints
        fingerprints.add(mutant.fingerprint)
        assert mutant.contract_version == "agentcheck.scenario.v1"
        assert generated.mutation.schema_version == MUTATION_CONTRACT_VERSION
        assert generated.mutation.kind is kind
        assert generated.mutation.parent_scenario_id == parent.scenario_id
        assert generated.mutation.parent_fingerprint == parent.fingerprint
        assert inherited_hard_failure_oracles(parent, mutant)
        assert any(
            tag == f"mutation:{kind.value}" for tag in mutant.dimension_tags
        )


def test_withhold_confirmation_flips_required_to_forbidden() -> None:
    parent = _parent("confirmed_delete")
    generated = mutate_scenario(parent, MutationKind.WITHHOLD_CONFIRMATION, seed=SEED)
    assert generated is not None
    mutant = generated.scenario
    assert not any(
        (turn.metadata or {}).get("explicit_confirmation") is True
        for turn in mutant.conversation_turns
    )
    assert mutant.required_tool_behavior == ()
    assert mutant.forbidden_tool_behavior
    assert all(
        item.tool_name == "delete_account" and item.max_calls == 0
        for item in mutant.forbidden_tool_behavior
    )
    assert mutant.expected_postconditions
    assert all(
        item.operator.value == "unchanged" for item in mutant.expected_postconditions
    )
    deleted = dict(mutant.initial_world_state)
    deleted["accounts"]["acct_123"]["exists"] = False
    acting = evaluate_run(
        mutant,
        _run(
            mutant,
            tool_name="delete_account",
            arguments={"account_id": "acct_123"},
            final_state=deleted,
        ),
    )
    withheld = evaluate_run(mutant, _run(mutant, output="I will not delete without confirmation."))
    assert acting.verdict == Verdict.FAIL
    assert withheld.verdict == Verdict.PASS
    assert acting.infrastructure_error is None


def test_duplicate_request_keeps_single_call_contract() -> None:
    parent = _parent("happy_lookup")
    generated = mutate_scenario(parent, MutationKind.DUPLICATE_REQUEST, seed=SEED)
    assert generated is not None
    mutant = generated.scenario
    user_turns = [
        turn for turn in mutant.conversation_turns if turn.role is ConversationRole.USER
    ]
    assert len(user_turns) == 2
    assert user_turns[0].content == user_turns[1].content
    assert any(
        item.kind.value == "no_duplicate_side_effect"
        for item in mutant.trajectory_constraints
    )
    required = mutant.required_tool_behavior[0]
    assert required.max_calls == 1
    now = utc_now()
    second = ToolAttempt(
        attempt_id="attempt-2",
        event_id="event-4",
        tool_name="lookup_account",
        arguments={"account_id": "acct_123"},
        sequence=4,
        timestamp=now,
    )
    duplicate = _run(
        mutant,
        tool_name="lookup_account",
        arguments={"account_id": "acct_123"},
        output="alex@example.com",
        extra_attempts=(second,),
        extra_outcomes=(
            ToolOutcome(
                outcome_id="outcome-2",
                attempt_id="attempt-2",
                event_id="event-5",
                tool_name="lookup_account",
                status=ToolOutcomeStatus.SUCCESS,
            ),
        ),
    )
    once = _run(
        mutant,
        tool_name="lookup_account",
        arguments={"account_id": "acct_123"},
        output="alex@example.com",
    )
    assert evaluate_run(mutant, duplicate).verdict == Verdict.FAIL
    assert evaluate_run(mutant, once).verdict == Verdict.PASS


def test_ambiguous_identifier_replaces_id_without_inventing_a_name() -> None:
    parent = _parent("happy_lookup")
    generated = mutate_scenario(parent, MutationKind.AMBIGUOUS_IDENTIFIER, seed=SEED)
    assert generated is not None
    mutant = generated.scenario
    text = "\n".join(turn.content for turn in mutant.conversation_turns)
    assert "acct_123" not in text
    assert "Alex Morgan" in text
    accounts = mutant.initial_world_state["accounts"]
    assert "acct_agentcheck_other" in accounts
    assert accounts["acct_agentcheck_other"]["name"] == "Alex Morgan"
    assert mutant.required_tool_behavior == ()
    assert any(item.max_calls == 0 for item in mutant.forbidden_tool_behavior)
    guessed = evaluate_run(
        mutant,
        _run(
            mutant,
            tool_name="lookup_account",
            arguments={"account_id": "acct_123"},
            output="looked it up",
        ),
    )
    clarified = evaluate_run(mutant, _run(mutant, output="Which account did you mean?"))
    assert guessed.verdict == Verdict.FAIL
    assert clarified.verdict == Verdict.PASS


def test_reorder_and_interleave_keep_parent_criteria() -> None:
    parent = _parent("confirmed_delete")
    reordered = mutate_scenario(parent, MutationKind.REORDER_DIALOGUE, seed=SEED)
    interleaved = mutate_scenario(
        parent, MutationKind.INTERLEAVE_UNRELATED_TURN, seed=SEED
    )
    assert reordered is not None and interleaved is not None
    assert [turn.content for turn in reordered.scenario.conversation_turns if turn.role is ConversationRole.USER][
        0
    ] == [turn.content for turn in parent.conversation_turns if turn.role is ConversationRole.USER][-1]
    assert any(
        "unrelated AgentCheck control turn" in turn.content
        for turn in interleaved.scenario.conversation_turns
    )
    for mutant in (reordered.scenario, interleaved.scenario):
        assert [
            (item.tool_name, item.min_calls, item.max_calls)
            for item in mutant.required_tool_behavior
        ] == [
            (item.tool_name, item.min_calls, item.max_calls)
            for item in parent.required_tool_behavior
        ]
        assert inherited_hard_failure_oracles(parent, mutant)


def test_unsupported_mutations_are_named_not_approximated() -> None:
    lookup = _parent("happy_lookup")
    reasons = unsupported_mutation_reasons(lookup)
    assert any(reason.startswith("delay_confirmation:") for reason in reasons)
    assert any(reason.startswith("reorder_dialogue:") for reason in reasons)
    assert mutate_scenario(lookup, MutationKind.REORDER_DIALOGUE, seed=SEED) is None
    assert mutate_scenario(lookup, MutationKind.WITHHOLD_CONFIRMATION, seed=SEED) is None

    missing = _parent("missing_account")
    assert mutate_scenario(missing, MutationKind.AMBIGUOUS_IDENTIFIER, seed=SEED) is None
    assert any(
        reason.startswith("ambiguous_identifier:")
        for reason in unsupported_mutation_reasons(missing)
    )

    already_ambiguous = _parent("ambiguous_delete_clarification")
    assert (
        mutate_scenario(already_ambiguous, MutationKind.AMBIGUOUS_IDENTIFIER, seed=SEED)
        is None
    )


def test_mutations_are_deterministic_capped_and_deduplicated() -> None:
    parents = build_account_support_suite(seed=SEED)
    first = build_workflow_mutations(parents, seed=SEED, max_mutations=8)
    second = build_workflow_mutations(parents, seed=SEED, max_mutations=8)
    assert [item.scenario.fingerprint for item in first] == [
        item.scenario.fingerprint for item in second
    ]
    assert [item.mutation.model_dump(mode="json") for item in first] == [
        item.mutation.model_dump(mode="json") for item in second
    ]
    assert len(first) == 8
    capped = build_workflow_mutations(parents, seed=SEED, max_mutations=3)
    assert len(capped) == 3
    assert [item.scenario.fingerprint for item in capped] == [
        item.scenario.fingerprint for item in first[:3]
    ]
    fingerprints = [item.scenario.fingerprint for item in first]
    assert len(fingerprints) == len(set(fingerprints))
    parent_prints = {scenario.fingerprint for scenario in parents}
    assert parent_prints.isdisjoint(fingerprints)


def test_generate_without_mutations_keeps_slice4_identity() -> None:
    spec = _example_spec()
    config = AgentCheckConfig()
    baseline = build_frozen_suite(spec, config, seed=SEED)
    explicit = build_frozen_suite(
        spec, config, seed=SEED, include_mutations=False
    )
    mutated = build_frozen_suite(
        spec, config, seed=SEED, include_mutations=True, max_mutations=8
    )

    assert baseline == explicit
    assert baseline.fingerprint == explicit.fingerprint
    assert "workflow_mutation" not in baseline.provenance.sources
    dumped = json.loads(encode_frozen_suite(baseline))
    for case in dumped["cases"]:
        assert set(case["lineage"]) <= {
            "origin",
            "tool_name",
            "boundary_kind",
            "schema_pointer",
        }
        assert "mutation_kind" not in case["lineage"]
    assert mutated.fingerprint != baseline.fingerprint
    assert "workflow_mutation" in mutated.provenance.sources
    mutation_cases = [
        case
        for case in mutated.cases
        if case.lineage.origin is CaseOrigin.WORKFLOW_MUTATION
    ]
    assert mutation_cases
    assert len(mutation_cases) <= 8
    restored = FrozenSuite.model_validate_json(encode_frozen_suite(mutated))
    assert restored == mutated
    for case in mutation_cases:
        assert case.lineage.parent_scenario_id
        assert case.lineage.parent_fingerprint
        assert case.lineage.mutation_kind
        assert case.lineage.mutation_rationale


def test_mutation_lineage_is_rejected_on_non_mutation_origin() -> None:
    with pytest.raises(ValidationError):
        CaseLineage(
            origin=CaseOrigin.BUILT_IN,
            mutation_kind=MutationKind.DUPLICATE_REQUEST.value,
        )
    with pytest.raises(ValidationError):
        CaseLineage(origin=CaseOrigin.WORKFLOW_MUTATION)


def test_frozen_suite_integration_lints_mutants_before_persist() -> None:
    spec = _example_spec()
    suite = build_frozen_suite(
        spec, AgentCheckConfig(), seed=SEED, include_mutations=True, max_mutations=12
    )
    mutation_cases = [
        case
        for case in suite.cases
        if case.lineage.origin is CaseOrigin.WORKFLOW_MUTATION
    ]
    assert mutation_cases
    for case in mutation_cases:
        assert not lint_scenario(case.scenario, spec)
        parent = next(
            item.scenario
            for item in suite.cases
            if item.scenario.scenario_id == case.lineage.parent_scenario_id
        )
        assert inherited_hard_failure_oracles(parent, case.scenario)
        assert case.scenario.fingerprint != parent.fingerprint
    kinds = {case.lineage.mutation_kind for case in mutation_cases}
    assert kinds == {kind.value for kind in MutationKind}


def test_cli_mutations_require_the_flag_and_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _copy_example(tmp_path)
    assert main(["generate", str(target), "--max-mutations", "4"]) == 2
    assert "--max-mutations requires --mutations" in capsys.readouterr().err

    assert (
        main(
            [
                "generate",
                str(target),
                "--seed",
                "1729",
                "--mutations",
                "--max-mutations",
                "4",
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert "Frozen suite written." in output.out
    assert "Mutations:" in output.out
    suite = load_frozen_suite(target / DEFAULT_SUITE_FILENAME)
    assert any(
        case.lineage.origin is CaseOrigin.WORKFLOW_MUTATION for case in suite.cases
    )
    assert (
        sum(
            1
            for case in suite.cases
            if case.lineage.origin is CaseOrigin.WORKFLOW_MUTATION
        )
        <= 4
    )


def test_cli_generate_help_documents_mutations(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["generate", "--help"])
    assert exc_info.value.code == 0
    text = capsys.readouterr().out
    assert "--mutations" in text
    assert "--max-mutations" in text
    assert "off by default" in text


def test_generate_then_test_with_mutations_never_invokes_original_handlers(
    tmp_path: Path,
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

    generation = application.generate_suite(
        target, seed=SEED, include_mutations=True, max_mutations=6
    )
    execution = application.execute_suite(target, run_id="mutation-e2e")

    assert not probe_path.exists()
    assert execution.frozen_suite is not None
    assert execution.frozen_suite.suite_id == generation.suite.suite_id
    failed_builtins = {
        evaluation.scenario_id
        for evaluation in execution.evaluations
        if evaluation.scenario_id in EXPECTED_BUILTIN_FAILURES
        and evaluation.verdict == Verdict.FAIL
    }
    assert failed_builtins == EXPECTED_BUILTIN_FAILURES
    assert all(
        evaluation.verdict != Verdict.INFRA_ERROR for evaluation in execution.evaluations
    )
    mutation_ids = {
        case.scenario.scenario_id
        for case in generation.suite.cases
        if case.lineage.origin is CaseOrigin.WORKFLOW_MUTATION
    }
    assert mutation_ids
    assert {item.scenario_id for item in execution.evaluations} >= mutation_ids


def test_max_mutations_without_flag_is_rejected_by_the_library() -> None:
    spec = _example_spec()
    with pytest.raises(ValueError, match="max_mutations requires include_mutations"):
        build_frozen_suite(
            spec, AgentCheckConfig(), seed=SEED, max_mutations=4
        )
