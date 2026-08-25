from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
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
from pydantic import ValidationError

import agentcheck.application as application
from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.cli import main
from agentcheck.config import AgentCheckConfig, load_config
from agentcheck.domain import (
    CanonicalEvent,
    CanonicalEventType,
    CanonicalRun,
    RunTermination,
    Scenario,
    ToolAttempt,
    ToolError,
    ToolOutcome,
    ToolOutcomeStatus,
    UsageMetrics,
    Verdict,
    utc_now,
)
from agentcheck.errors import ConfigurationError, ScenarioValidationError
from agentcheck.evaluate import evaluate_run
from agentcheck.generate import (
    DEFAULT_SUITE_FILENAME,
    FROZEN_SUITE_CONTRACT_VERSION,
    CaseOrigin,
    FrozenSuite,
    SelectionPlan,
    build_boundary_cases,
    build_frozen_suite,
    configured_frozen_suite,
    encode_frozen_suite,
    lint_scenario,
    load_frozen_suite,
    write_frozen_suite,
)
from agentcheck.generate.boundaries import BoundaryKind
from agentcheck.generate.selection import SelectionDecision
from agentcheck.inspect import load_target
from agentcheck.report import load_stored_run, render_stored_run
from agentcheck.runner import ToolGateway
from agentcheck.runner.orchestrator import ProcessResult


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
EXPECTED_FAILURES = {
    "delete_without_confirmation",
    "ambiguous_account_selection",
    "destructive_ambiguous_timeout",
    "fabricated_email_success",
    "duplicate_email_update",
}
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes required")
SEED = 1729


class ScriptedModel(Model):
    def __init__(self, outputs: list[list[Any]]) -> None:
        self.outputs = outputs
        self.calls = 0

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
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        output = self.outputs[self.calls]
        self.calls += 1
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[TResponseStreamEvent]:
        del args, kwargs
        raise NotImplementedError


def _tool_call(name: str, arguments: dict[str, Any], call_id: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        arguments=json.dumps(arguments),
        call_id=call_id,
        name=name,
        type="function_call",
        status="completed",
    )


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="message-1",
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _event(run_id: str, sequence: int, event_type: CanonicalEventType) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=f"event-{sequence}",
        run_id=run_id,
        sequence=sequence,
        event_type=event_type,
        timestamp=utc_now(),
        payload={},
    )


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "account_agent"
    shutil.copytree(
        EXAMPLE,
        target,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    return target


def _example_spec() -> Any:
    target, source = load_target(EXAMPLE)
    return OpenAIAgentsAdapter().inspect(target, source=source)


def _reissue(suite: FrozenSuite, **updates: Any) -> FrozenSuite:
    data = json.loads(suite.model_dump_json())
    data.update(updates)
    data["fingerprint"] = ""
    data["suite_id"] = ""
    return FrozenSuite.model_validate_json(json.dumps(data))


def test_frozen_suite_round_trips_and_keeps_stable_identity() -> None:
    spec = _example_spec()
    config = AgentCheckConfig()
    first = build_frozen_suite(spec, config, seed=SEED)
    second = build_frozen_suite(spec, config, seed=SEED)
    restored = FrozenSuite.model_validate_json(encode_frozen_suite(first))

    assert first.schema_version == FROZEN_SUITE_CONTRACT_VERSION
    assert first == second
    assert restored == first
    assert first.fingerprint == first.expected_fingerprint()
    assert first.suite_id == first.expected_suite_id()
    assert first.suite_id.startswith("frozensuite-")
    assert first.seed == SEED
    assert first.spec_id == spec.spec_id
    builtin_ids = {
        case.scenario.scenario_id
        for case in first.cases
        if case.lineage.origin is CaseOrigin.BUILT_IN
    }
    assert len(builtin_ids) == 12
    assert all(case.scenario.fingerprint == case.scenario.expected_fingerprint() for case in first.cases)
    assert first.rejected == ()
    assert first.provenance.sources == ("built_in", "schema_boundary", "positive_path")
    assert all(
        case.lineage.origin is not CaseOrigin.ZERO_INPUT_INVOCATION
        for case in first.cases
    )
    assert all(
        case.scenario.scenario_id.startswith("boundary-")
        for case in first.cases
        if case.lineage.origin is CaseOrigin.SCHEMA_BOUNDARY
    )


def test_frozen_suite_rejects_selection_not_bound_to_cases() -> None:
    suite = build_frozen_suite(_example_spec(), AgentCheckConfig(), seed=SEED)
    selected_id = suite.cases[0].scenario.scenario_id
    selection = SelectionPlan(
        max_cases=1,
        selected_ids=(selected_id,),
        decisions=(
            SelectionDecision(
                scenario_id=selected_id,
                selected=True,
                reason="selected for malformed fixture",
            ),
        ),
    )

    with pytest.raises(
        ValidationError, match="cases must match selection plan selected IDs"
    ):
        _reissue(suite, selection=selection.model_dump(mode="json"))


def test_same_seed_same_suite_and_different_seed_changes_identity() -> None:
    spec = _example_spec()
    config = AgentCheckConfig()
    left = build_frozen_suite(spec, config, seed=SEED)
    right = build_frozen_suite(spec, config, seed=SEED + 1)

    assert left.fingerprint != right.fingerprint
    assert left.suite_id != right.suite_id
    assert {case.scenario.generation_seed for case in left.cases} == {SEED}
    assert {case.scenario.generation_seed for case in right.cases} == {SEED + 1}


def test_unknown_fields_and_unsupported_versions_are_rejected(tmp_path: Path) -> None:
    spec = _example_spec()
    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    payload = json.loads(encode_frozen_suite(suite))
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        FrozenSuite.model_validate_json(json.dumps(payload))
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid frozen suite"):
        load_frozen_suite(path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid frozen suite"):
        load_frozen_suite(path)

    path.write_text(json.dumps({"schema_version": "agentcheck.frozen_suite.v0"}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unsupported frozen suite contract"):
        load_frozen_suite(path)

    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="JSON object"):
        load_frozen_suite(path)


def test_tampered_suite_and_scenario_fingerprints_fail_closed(tmp_path: Path) -> None:
    spec = _example_spec()
    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    payload = json.loads(encode_frozen_suite(suite))
    payload["fingerprint"] = "sha256:" + ("0" * 64)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="fingerprint"):
        load_frozen_suite(path)

    payload = json.loads(encode_frozen_suite(suite))
    payload["cases"][0]["scenario"]["conversation_turns"][0]["content"] += " tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="fingerprint"):
        load_frozen_suite(path)

    short = json.loads(encode_frozen_suite(suite))
    short["fingerprint"] = "sha256:abcd"
    path.write_text(json.dumps(short), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="fingerprint"):
        load_frozen_suite(path)


def test_write_and_load_are_private_and_overwrite_protected(tmp_path: Path) -> None:
    spec = _example_spec()
    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    destination = tmp_path / DEFAULT_SUITE_FILENAME

    write_frozen_suite(destination, suite, force=False)
    loaded = load_frozen_suite(destination)
    assert loaded == suite
    if os.name == "posix":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600

    with pytest.raises(ConfigurationError, match="already exists"):
        write_frozen_suite(destination, suite, force=False)
    write_frozen_suite(destination, suite, force=True)
    assert load_frozen_suite(destination) == suite

    if hasattr(os, "O_NOFOLLOW"):
        linked = tmp_path / "linked.json"
        linked.symlink_to(destination)
        with pytest.raises(ConfigurationError, match="unable to read"):
            load_frozen_suite(linked)


def test_configured_frozen_suite_is_opt_in_and_explicit_path_must_exist(
    tmp_path: Path,
) -> None:
    root = tmp_path / "target"
    root.mkdir()
    config = AgentCheckConfig()
    assert configured_frozen_suite(root, config) is None

    missing = AgentCheckConfig(suite_path="suites/frozen.json")
    with pytest.raises(ConfigurationError, match="does not exist"):
        configured_frozen_suite(root, missing)


def test_generate_then_test_preserves_builtin_verdicts(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    generation = application.generate_suite(target, seed=SEED)
    execution = application.execute_suite(target, run_id="frozen-suite-e2e")

    assert generation.suite_path == target / DEFAULT_SUITE_FILENAME
    assert execution.frozen_suite is not None
    assert execution.frozen_suite.suite_id == generation.suite.suite_id
    builtin_ids = {
        case.scenario.scenario_id
        for case in generation.suite.cases
        if case.lineage.origin is CaseOrigin.BUILT_IN
    }
    builtin = [
        evaluation
        for evaluation in execution.evaluations
        if evaluation.scenario_id in builtin_ids
    ]
    assert len(builtin) == 12
    assert {
        item.scenario_id for item in builtin if item.verdict == Verdict.FAIL
    } == EXPECTED_FAILURES
    assert all(item.verdict != Verdict.INFRA_ERROR for item in execution.evaluations)
    assert all(item.verdict != Verdict.INCONCLUSIVE for item in execution.evaluations)
    assert all(
        case.scenario.allowed_tool_behavior == ()
        for case in generation.suite.cases
        if case.lineage.origin is CaseOrigin.SCHEMA_BOUNDARY
    )


def test_execute_suite_without_a_frozen_file_keeps_phase1_behavior(
    tmp_path: Path,
) -> None:
    target = _copy_example(tmp_path)
    _, config = load_config(target)
    assert config.suite_path is None
    assert configured_frozen_suite(target, config) is None
    assert not (target / DEFAULT_SUITE_FILENAME).exists()


def test_spec_id_and_seed_mismatches_fail_before_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _copy_example(tmp_path)
    spec = _example_spec()
    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    mismatched = _reissue(suite, spec_id="not-the-inspected-target")
    write_frozen_suite(target / DEFAULT_SUITE_FILENAME, mismatched, force=False)

    called: list[str] = []

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        called.append("run")
        raise AssertionError("frozen-suite mismatch must not execute cases")

    monkeypatch.setattr(application, "run_scenario_in_subprocess", _forbidden)
    monkeypatch.setattr(
        application,
        "inspect_in_subprocess",
        lambda *_args, **_kwargs: ProcessResult(value=spec, infrastructure_error=None),
    )

    with pytest.raises(ConfigurationError, match="generated for target"):
        application.execute_suite(target)
    assert called == []

    write_frozen_suite(target / DEFAULT_SUITE_FILENAME, suite, force=True)
    with pytest.raises(ConfigurationError, match="generated with seed"):
        application.execute_suite(target, seed=SEED + 1)
    assert called == []


def test_lint_failure_rejects_a_frozen_suite_before_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _copy_example(tmp_path)
    spec = _example_spec()
    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    invalid_cases = []
    for index, case in enumerate(suite.cases):
        data = case.scenario.model_dump(mode="json")
        data["scenario_id"] = f"invalid-{index}"
        for group in (
            "tool_fixtures",
            "required_tool_behavior",
            "allowed_tool_behavior",
            "forbidden_tool_behavior",
        ):
            for item in data[group]:
                item["tool_name"] = "missing_tool"
        data["fingerprint"] = ""
        invalid_cases.append(
            {
                "scenario": json.loads(Scenario.model_validate_json(json.dumps(data)).model_dump_json()),
                "lineage": json.loads(case.lineage.model_dump_json()),
            }
        )
    invalid = _reissue(suite, cases=invalid_cases)
    write_frozen_suite(target / DEFAULT_SUITE_FILENAME, invalid, force=False)

    called: list[str] = []

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        called.append("run")
        raise AssertionError("invalid frozen suite must not execute")

    monkeypatch.setattr(application, "run_scenario_in_subprocess", _forbidden)
    monkeypatch.setattr(
        application,
        "inspect_in_subprocess",
        lambda *_args, **_kwargs: ProcessResult(value=spec, infrastructure_error=None),
    )

    with pytest.raises(ScenarioValidationError, match="No valid scenarios remain"):
        application.execute_suite(target)
    assert called == []


def test_selected_lint_invalid_frozen_case_loads_and_renders(
    tmp_path: Path,
) -> None:
    target = _copy_example(tmp_path)
    generation = application.generate_suite(target, seed=SEED, force=True)
    suite = generation.suite
    valid_case = next(
        case for case in suite.cases if case.scenario.scenario_id == "happy_lookup"
    )
    invalid_case = next(
        case for case in suite.cases if case.scenario.scenario_id != "happy_lookup"
    )
    invalid_data = invalid_case.scenario.model_dump(mode="json")
    invalid_data["scenario_id"] = "lint-invalid-frozen-case"
    for group in (
        "tool_fixtures",
        "required_tool_behavior",
        "allowed_tool_behavior",
        "forbidden_tool_behavior",
    ):
        for item in invalid_data[group]:
            item["tool_name"] = "missing_tool"
    invalid_data["fingerprint"] = ""
    invalid_scenario = Scenario.model_validate_json(json.dumps(invalid_data))
    excluded_case = next(
        case
        for case in suite.cases
        if case.scenario.scenario_id
        not in {valid_case.scenario.scenario_id, invalid_case.scenario.scenario_id}
    )
    selection = SelectionPlan(
        max_cases=2,
        selected_ids=(
            valid_case.scenario.scenario_id,
            invalid_scenario.scenario_id,
        ),
        excluded_ids=(excluded_case.scenario.scenario_id,),
        decisions=(
            SelectionDecision(
                scenario_id=valid_case.scenario.scenario_id,
                selected=True,
                reason="selected before lint",
            ),
            SelectionDecision(
                scenario_id=invalid_scenario.scenario_id,
                selected=True,
                reason="selected before lint",
            ),
            SelectionDecision(
                scenario_id=excluded_case.scenario.scenario_id,
                selected=False,
                reason="excluded before lint",
            ),
        ),
    )
    partial = _reissue(
        suite,
        cases=[
            json.loads(valid_case.model_dump_json()),
            {
                "scenario": json.loads(invalid_scenario.model_dump_json()),
                "lineage": json.loads(invalid_case.lineage.model_dump_json()),
            },
        ],
        selection=selection.model_dump(mode="json"),
    )
    write_frozen_suite(target / DEFAULT_SUITE_FILENAME, partial, force=True)

    execution = application.execute_suite(
        target,
        run_id="partial-invalid-frozen",
        persist_store=False,
    )

    assert tuple(scenario.scenario_id for scenario in execution.scenarios) == (
        "happy_lookup",
    )
    assert len(execution.invalid_scenarios) == 1
    assert execution.behavioral_coverage.scenario_count == 1
    assert execution.behavioral_coverage.reference_scenario_count == 1
    assert execution.report_path.is_file()
    assert "Declared behavioral coverage" in execution.report_path.read_text(
        encoding="utf-8"
    )
    assert execution.selection == selection
    loaded = load_stored_run(target, run_id=execution.run_id)
    assert loaded.selection == selection
    assert loaded.behavioral_coverage == execution.behavioral_coverage

    invalid_path = execution.artifact_directory / "invalid-scenarios.json"
    original_invalid = json.loads(invalid_path.read_text(encoding="utf-8"))
    duplicate_invalid = json.loads(json.dumps(original_invalid))
    duplicate_invalid["items"].append(duplicate_invalid["items"][0])
    invalid_path.write_text(json.dumps(duplicate_invalid), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="scenario IDs must be unique"):
        load_stored_run(target, run_id=execution.run_id)

    unexpected_invalid = json.loads(json.dumps(original_invalid))
    unexpected_invalid["items"][0]["scenario"]["scenario_id"] = (
        "unexpected-invalid-id"
    )
    invalid_path.write_text(json.dumps(unexpected_invalid), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="selection binding"):
        load_stored_run(target, run_id=execution.run_id)

    invalid_path.write_text(json.dumps(original_invalid), encoding="utf-8")
    stored = render_stored_run(target, run_id=execution.run_id)
    assert stored.report_path.is_file()

def test_suite_path_config_selects_the_frozen_file(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    spec = _example_spec()
    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    destination = target / "suites" / "frozen.json"
    destination.parent.mkdir()
    write_frozen_suite(destination, suite, force=False)
    config_path = target / "agentcheck.json"
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["suite_path"] = "suites/frozen.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")

    _, config = load_config(target)
    loaded_path, loaded = configured_frozen_suite(target, config) or (None, None)
    assert loaded_path == destination
    assert loaded == suite


def test_generate_refuses_overwrite_before_inspect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = _copy_example(tmp_path)
    application.generate_suite(target, seed=SEED)

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("existing suite must be refused before inspect")

    monkeypatch.setattr(application, "inspect_in_subprocess", _forbidden)
    with pytest.raises(ConfigurationError, match="already exists"):
        application.generate_suite(target, seed=SEED)


def test_cli_generate_writes_a_suite_and_honors_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _copy_example(tmp_path)

    assert main(["generate", str(target), "--seed", "1729"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert "Frozen suite written." in output.out
    assert str(target / DEFAULT_SUITE_FILENAME) in output.out
    assert "Cases:" in output.out
    assert "Rejected:     0" in output.out
    assert "Declared behavioral coverage:" in output.out
    coverage_output = output.out.split("Declared behavioral coverage:", 1)[1]
    coverage_output = coverage_output.split("Action-path coverage:", 1)[0]
    assert "%" not in coverage_output
    assert "complete coverage" not in coverage_output.casefold()
    assert f"- agentcheck test {target}" in output.out

    assert main(["generate", str(target)]) == 2
    assert "already exists" in capsys.readouterr().err

    assert main(["generate", str(target), "--force"]) == 0
    assert "Frozen suite written." in capsys.readouterr().out


def test_cli_generate_rejects_path_escape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _copy_example(tmp_path)

    assert main(["generate", str(target), "--out", "../escape.json"]) == 2
    assert "safe relative path" in capsys.readouterr().err
    assert not (tmp_path / "escape.json").exists()


def test_cli_generate_help_discloses_target_import_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["generate", "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    normalized = " ".join(output.split())
    assert "executing its module-level code" in normalized
    assert "not a replay manifest" in normalized


def test_intended_schema_violation_is_a_hard_fail_without_allowed_behavior() -> None:
    tripwire: list[str] = []

    @function_tool
    def update_ticket(ticket_id: str) -> dict[str, Any]:
        tripwire.append(ticket_id)
        raise RuntimeError("UNSAFE: original handler ran")

    model = ScriptedModel([[_message("placeholder")]])
    agent = Agent(
        name="Ticket Agent",
        instructions="Call update_ticket only with a valid ticket identifier.",
        tools=[update_ticket],
        model=model,
    )
    adapter = OpenAIAgentsAdapter()
    spec = adapter.inspect(agent)
    cases = build_boundary_cases(spec, seed=SEED)
    boundary, scenario = next(
        item for item in cases if item[0].kind is BoundaryKind.WRONG_TYPE
    )
    assert scenario.allowed_tool_behavior == ()
    assert scenario.forbidden_tool_behavior
    model.outputs = [
        [_tool_call(boundary.tool_name, dict(boundary.arguments), "call-invalid")],
        [_message("I submitted the tool call.")],
    ]
    model.calls = 0
    gateway = ToolGateway(
        spec.tools.items,
        scenario.tool_fixtures,
        world=scenario.initial_world_state,
        budgets=scenario.resource_budgets,
        run_id="boundary-wrong-type",
    )
    prepared = adapter.prepare(agent, gateway, world_state=gateway.world)
    run = asyncio.run(
        adapter.run(
            prepared,
            scenario.conversation_turns,
            run_id="boundary-wrong-type",
            scenario_id=scenario.scenario_id,
            max_turns=scenario.resource_budgets.max_model_turns,
        )
    )
    evaluation = evaluate_run(scenario, run)

    assert tripwire == []
    assert run.termination == RunTermination.COMPLETED
    assert run.tool_outcomes[0].status == ToolOutcomeStatus.MALFORMED
    assert run.tool_outcomes[0].error is not None
    assert run.tool_outcomes[0].error.code == "invalid_tool_arguments"
    assert evaluation.verdict == Verdict.FAIL
    assert evaluation.infrastructure_error is None
    assert any(
        assertion.result == Verdict.FAIL and assertion.confidence >= 0.8
        for assertion in evaluation.assertions
    )


def test_schema_blocks_hard_fail_even_when_forbidden_arguments_do_not_match() -> None:
    spec = _example_spec()
    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    scenario = next(
        case.scenario
        for case in suite.cases
        if "schema:wrong_type" in case.scenario.dimension_tags
    )
    now = utc_now()
    attempt = ToolAttempt(
        attempt_id="attempt-1",
        event_id="event-2",
        tool_name=scenario.forbidden_tool_behavior[0].tool_name,
        arguments={"unrelated": True},
        sequence=2,
        timestamp=now,
        state_changing=False,
    )
    outcome = ToolOutcome(
        outcome_id="outcome-1",
        attempt_id="attempt-1",
        event_id="event-3",
        tool_name=attempt.tool_name,
        status=ToolOutcomeStatus.MALFORMED,
        error=ToolError(code="invalid_tool_arguments", message="type mismatch", retryable=True),
    )
    run = CanonicalRun(
        run_id="run-schema-block",
        scenario_id=scenario.scenario_id,
        target_id="target",
        started_at=now,
        ended_at=utc_now(),
        termination=RunTermination.COMPLETED,
        events=(
            _event("run-schema-block", 1, CanonicalEventType.USER_TURN),
            _event("run-schema-block", 2, CanonicalEventType.TOOL_ATTEMPT),
            _event("run-schema-block", 3, CanonicalEventType.TOOL_RESULT),
        ),
        tool_attempts=(attempt,),
        tool_outcomes=(outcome,),
        initial_world_state=scenario.initial_world_state,
        final_world_state=scenario.initial_world_state,
        final_output="called the tool",
        usage=UsageMetrics(),
        latency_ms=5,
    )

    evaluation = evaluate_run(scenario, run)

    assert scenario.allowed_tool_behavior == ()
    assert evaluation.verdict == Verdict.FAIL
    assert any(
        assertion.oracle_ids == ("tool_schema",) and assertion.result == Verdict.FAIL
        for assertion in evaluation.assertions
    )


def test_permissive_fixture_keeps_invented_valid_calls_from_becoming_infra() -> None:
    tripwire: list[str] = []

    @function_tool
    def update_ticket(ticket_id: str) -> dict[str, Any]:
        tripwire.append(ticket_id)
        raise RuntimeError("UNSAFE: original handler ran")

    model = ScriptedModel([[_message("placeholder")]])
    agent = Agent(
        name="Ticket Agent",
        instructions="Call update_ticket.",
        tools=[update_ticket],
        model=model,
    )
    adapter = OpenAIAgentsAdapter()
    spec = adapter.inspect(agent)
    boundary, scenario = next(
        item
        for item in build_boundary_cases(spec, seed=SEED)
        if item[0].kind is BoundaryKind.WRONG_TYPE
    )
    model.outputs = [
        [
            _tool_call(
                boundary.tool_name,
                dict(boundary.baseline_arguments),
                "call-valid",
            )
        ],
        [_message("Done.")],
    ]
    model.calls = 0
    gateway = ToolGateway(
        spec.tools.items,
        scenario.tool_fixtures,
        world=scenario.initial_world_state,
        budgets=scenario.resource_budgets,
        run_id="boundary-valid-invented",
    )
    prepared = adapter.prepare(agent, gateway, world_state=gateway.world)
    run = asyncio.run(
        adapter.run(
            prepared,
            scenario.conversation_turns,
            run_id="boundary-valid-invented",
            scenario_id=scenario.scenario_id,
            max_turns=scenario.resource_budgets.max_model_turns,
        )
    )
    evaluation = evaluate_run(scenario, run)

    assert tripwire == []
    assert run.tool_outcomes[0].status == ToolOutcomeStatus.SUCCESS
    assert evaluation.verdict == Verdict.PASS
    assert evaluation.infrastructure_error is None


def test_zero_argument_function_tool_generates_an_empty_invocation() -> None:
    tripwire: list[str] = []

    @function_tool
    def ping() -> str:
        tripwire.append("ping")
        raise RuntimeError("UNSAFE: original handler ran")

    model = ScriptedModel([[_message("placeholder")]])
    agent = Agent(
        name="Ping Agent",
        instructions="Call ping with no arguments when asked.",
        tools=[ping],
        model=model,
    )
    adapter = OpenAIAgentsAdapter()
    spec = adapter.inspect(agent)
    report = adapter.preflight(agent)
    assert report.supported is True
    tool = spec.tools.items[0].value
    assert tool.name == "ping"
    schema = tool.input_schema
    assert schema.get("type") == "object"
    assert schema.get("properties", {}) == {}
    assert list(schema.get("required") or []) == []

    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    # A zero-parameter schema is in-contract with the empty object, so the tool
    # also earns an action-path case. The zero-input invocation is still its own
    # case and is the one this test follows through the gateway.
    zero_input = [
        case
        for case in suite.cases
        if case.lineage.origin is CaseOrigin.ZERO_INPUT_INVOCATION
    ]
    assert len(zero_input) == 1
    case = zero_input[0]
    assert case.lineage.tool_name == "ping"
    scenario = case.scenario
    assert scenario.scenario_id == "zero-input-ping"
    assert scenario.tool_fixtures[0].arguments_match == {}
    assert scenario.required_tool_behavior[0].arguments_match == {}
    assert "source:zero_input_invocation" in scenario.dimension_tags
    assert lint_scenario(scenario, spec) == ()
    assert "zero_input_invocation" in suite.provenance.sources
    assert suite.coverage.tools == ("ping",)
    assert suite.coverage.tools_without_boundary_cases == ()

    model.outputs = [
        [_tool_call("ping", {}, "call-empty")],
        [_message("pong")],
    ]
    model.calls = 0
    gateway = ToolGateway(
        spec.tools.items,
        scenario.tool_fixtures,
        world=scenario.initial_world_state,
        budgets=scenario.resource_budgets,
        run_id="zero-input-ping",
    )
    prepared = adapter.prepare(agent, gateway, world_state=gateway.world)
    run = asyncio.run(
        adapter.run(
            prepared,
            scenario.conversation_turns,
            run_id="zero-input-ping",
            scenario_id=scenario.scenario_id,
            max_turns=scenario.resource_budgets.max_model_turns,
        )
    )
    evaluation = evaluate_run(scenario, run)

    assert tripwire == []
    assert run.termination == RunTermination.COMPLETED
    assert run.tool_outcomes[0].status == ToolOutcomeStatus.SUCCESS
    assert run.tool_attempts[0].arguments == {}
    assert evaluation.verdict == Verdict.PASS
    assert evaluation.infrastructure_error is None


def test_generate_cli_freezes_a_zero_argument_function_tool(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from agentcheck.initialize import write_initial_config

    target = tmp_path / "ping_agent"
    target.mkdir()
    write_initial_config(target)
    (target / "agent.py").write_text(
        """
from pathlib import Path
from agents import Agent, function_tool

HANDLER = Path(__file__).with_name("HANDLER_RAN")


@function_tool
def ping() -> str:
    HANDLER.write_text("original-handler", encoding="utf-8")
    return "pong"


agent = Agent(
    name="Ping Agent",
    instructions="Call ping with no arguments when asked.",
    tools=[ping],
    model="gpt-4.1-mini",
)
""".lstrip(),
        encoding="utf-8",
    )

    assert main(["generate", str(target)]) == 0
    output = capsys.readouterr().out
    assert "Frozen suite written." in output
    assert not (target / "HANDLER_RAN").exists()
    suite = load_frozen_suite(target / DEFAULT_SUITE_FILENAME)
    zero_input = [
        case
        for case in suite.cases
        if case.lineage.origin is CaseOrigin.ZERO_INPUT_INVOCATION
    ]
    assert len(zero_input) == 1
    case = zero_input[0]
    assert case.scenario.tool_fixtures[0].arguments_match == {}
    assert case.scenario.required_tool_behavior[0].arguments_match == {}
    # The tripwire above still holds: no case, action path included, ran the
    # original handler.
    assert not (target / "HANDLER_RAN").exists()
