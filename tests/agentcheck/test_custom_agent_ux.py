"""Two things AgentCheck cannot do for a custom agent, and what it says instead.

Milestone B made custom agents executable and left two claims that were true
only by omission. A custom agent owns its own model calls, so AgentCheck can
neither substitute a controlled model into it nor count the model turns it
used. Both were quietly survivable -- ``controlled_model`` was accepted and
noted in metadata, and the model-turn budget compared an unobserved zero
against the limit and passed.

Neither is survivable now, and these tests are the reason it stays that way.
The first half proves a controlled-model request cannot reach a run at all; the
second proves a model-turn budget that cannot be measured is never reported as
having been met. The rest is the ordinary developer path -- config, entrypoint
errors, inspect, generate, test -- run end to end against a temporary target.

Nothing here contacts a provider.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from agentcheck.adapters import CustomAgentAdapter, UnsupportedTargetError
from agentcheck.application import execute_suite, generate_suite, inspect_target
from agentcheck.config import (
    ADAPTERS_WITHOUT_CONTROLLED_MODEL,
    AgentCheckConfig,
    load_config,
)
from agentcheck.domain import (
    CanonicalEvent,
    CanonicalEventType,
    CanonicalRun,
    ConversationRole,
    ConversationTurn,
    RunTermination,
    Scenario,
    ToolDefinition,
    Verdict,
    utc_now,
)
from agentcheck.domain.scenario import (
    OracleProvenance,
    OracleStrength,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
)
from agentcheck.errors import ConfigurationError
from agentcheck.evaluate import evaluate_run
from agentcheck.evaluate.engine import MODEL_TURNS_UNOBSERVABLE, _observed_model_turns
from agentcheck.initialize import write_initial_config
from agentcheck.runner import ToolGateway, run_scenario_in_subprocess


# ---------------------------------------------------------------------------
# A minimal target on disk, which several sections drive through the real CLI
# entry points rather than through the adapter directly.
# ---------------------------------------------------------------------------

CUSTOM_TARGET = '''
from typing import Any, Sequence

from agentcheck import ToolRuntime, TurnResult
from agentcheck.domain import ToolDefinition

LOOKUP = ToolDefinition(
    name="lookup_account",
    description="Read one account.",
    input_schema={
        "type": "object",
        "properties": {"account_id": {"type": "string"}},
        "required": ["account_id"],
        "additionalProperties": False,
    },
)


class Support:
    name = "Support"
    instructions = "Answer account questions."
    tools: Sequence[ToolDefinition] = (LOOKUP,)

    def start(self, message: str, tools: ToolRuntime) -> TurnResult:
        outcome = tools.call("lookup_account", {"account_id": "A-1"})
        return TurnResult(output="Account is " + str(outcome.result), state={})

    def resume(self, state: Any, message: str, tools: ToolRuntime) -> TurnResult:
        return TurnResult(output="Nothing further.", state=state)


agent = Support()
'''

LOOKUP = ToolDefinition(
    name="lookup_account",
    input_schema={
        "type": "object",
        "properties": {"account_id": {"type": "string"}},
        "required": ["account_id"],
        "additionalProperties": False,
    },
)


class MinimalAgent:
    name = "Support"
    tools: Sequence[ToolDefinition] = (LOOKUP,)

    def start(self, message: str, tools: Any) -> Any:
        from agentcheck import TurnResult

        return TurnResult(output="done", state=None)

    def resume(self, state: Any, message: str, tools: Any) -> Any:
        from agentcheck import TurnResult

        return TurnResult(output="done", state=state)


def _write_target(root: Path, source: str = CUSTOM_TARGET) -> AgentCheckConfig:
    (root / "agent.py").write_text(source, encoding="utf-8")
    config = AgentCheckConfig(adapter="custom", entrypoint="agent.py:agent")
    (root / "agentcheck.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )
    return config


# ---------------------------------------------------------------------------
# Controlled model: refused, not noted
# ---------------------------------------------------------------------------


def test_a_custom_target_without_a_controlled_model_request_still_runs(
    tmp_path: Path,
) -> None:
    """The default path is untouched; only the explicit request is refused."""

    config = _write_target(tmp_path)

    assert config.controlled_model is False
    result = run_scenario_in_subprocess(tmp_path, config, _scenario(), "cm-default")

    run = result.require_value()
    assert run.termination == RunTermination.COMPLETED
    assert [attempt.tool_name for attempt in run.tool_attempts] == ["lookup_account"]


def test_config_refuses_controlled_model_for_a_custom_target() -> None:
    """Caught at the configuration, before a worker is ever launched."""

    with pytest.raises(ValueError) as raised:
        AgentCheckConfig(adapter="custom", controlled_model=True)

    message = str(raised.value)
    assert "controlled_model" in message
    assert "custom" in message


def test_a_config_file_requesting_controlled_model_fails_to_load(
    tmp_path: Path,
) -> None:
    """The developer is told which line of agentcheck.json is wrong."""

    _write_target(tmp_path)
    (tmp_path / "agentcheck.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.config.v1",
                "adapter": "custom",
                "entrypoint": "agent.py:agent",
                "controlled_model": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as raised:
        load_config(tmp_path)

    assert "controlled_model" in str(raised.value)


def test_the_adapter_refuses_a_controlled_model_request_independently() -> None:
    """A config built in code skips the validator; the adapter still refuses.

    Two checks rather than one because they guard different entrances, and the
    dangerous outcome -- a run that looks offline and is not -- is the same
    through either.
    """

    adapter = CustomAgentAdapter()
    agent = MinimalAgent()
    spec = adapter.inspect(agent)
    gateway = ToolGateway(tuple(item.value for item in spec.tools.items), ())

    with pytest.raises(UnsupportedTargetError) as raised:
        adapter.prepare(agent, gateway, controlled_model=True)

    assert "controlled_model_unsupported" in str(raised.value)
    assert "config.controlled_model" in str(raised.value)


def test_no_prepared_custom_target_can_carry_a_controlled_model_request() -> None:
    """There is no 'requested but unsupported' state left to misread.

    Milestone B recorded the request in metadata and continued. A caveat beside
    a completed run reads as a note about a substitution that happened; the
    state itself had to go, not just its wording.
    """

    adapter = CustomAgentAdapter()
    agent = MinimalAgent()
    spec = adapter.inspect(agent)
    gateway = ToolGateway(tuple(item.value for item in spec.tools.items), ())

    prepared = adapter.prepare(agent, gateway)

    leaked = [key for key in prepared.metadata if "controlled_model" in key]
    assert leaked == [], f"a controlled-model marker survived: {leaked}"


def test_controlled_model_stays_available_to_the_adapters_that_implement_it() -> None:
    """The restriction names one adapter, and does not spread to the others."""

    assert ADAPTERS_WITHOUT_CONTROLLED_MODEL == frozenset({"custom"})
    assert AgentCheckConfig(adapter="openai_agents", controlled_model=True)
    assert AgentCheckConfig(adapter="pydantic_ai", controlled_model=True)


# ---------------------------------------------------------------------------
# Model turns: not measured, and never reported as met
# ---------------------------------------------------------------------------


def _oracle() -> OracleProvenance:
    # `supports_hard_failure` so a declared constraint can reach FAIL. Without
    # it the evaluator downgrades every failure to INCONCLUSIVE, which would
    # hide whether the model-turn comparison happened at all -- the one thing
    # these tests are here to tell apart.
    return OracleProvenance(
        oracle_id="oracle-1",
        strength=OracleStrength.TOOL_CONTRACT,
        source="agent.py",
        confidence=1.0,
        evidence_ids=("evidence-1",),
        supports_hard_failure=True,
    )


def _scenario(
    *,
    trajectory_constraints: Sequence[TrajectoryConstraint] = (),
) -> Scenario:
    from agentcheck.domain import SimulatedToolOutcome, SimulatedToolStatus, ToolFixture

    return Scenario(
        scenario_id="custom_lookup",
        title="Look up an account",
        conversation_turns=(
            ConversationTurn(
                turn_id="t0", role=ConversationRole.USER, content="Check account A-1"
            ),
        ),
        tool_fixtures=(
            ToolFixture(
                fixture_id="f1",
                tool_name="lookup_account",
                outcome=SimulatedToolOutcome(
                    status=SimulatedToolStatus.SUCCESS, result={"status": "active"}
                ),
            ),
        ),
        trajectory_constraints=tuple(trajectory_constraints),
        dimension_tags=("custom",),
        oracle_provenance=(_oracle(),),
        generation_seed=1,
    )


def _run(
    *,
    model_turns_observable: bool | None,
    model_requests: int = 0,
    run_id: str = "run-1",
) -> CanonicalRun:
    """A canonical run with a chosen model-turn observability declaration."""

    events = tuple(
        CanonicalEvent(
            event_id=f"{run_id}:event:{index:04d}",
            run_id=run_id,
            sequence=index,
            event_type=CanonicalEventType.MODEL_REQUEST,
            timestamp=utc_now(),
        )
        for index in range(model_requests)
    )
    metadata: dict[str, Any] = {"framework": "custom"}
    if model_turns_observable is not None:
        metadata["model_turns_observable"] = model_turns_observable
    now = utc_now()
    return CanonicalRun(
        run_id=run_id,
        scenario_id="custom_lookup",
        target_id="agentspec-x",
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
        events=events,
        final_output="done",
        metadata=metadata,
    )


def test_an_unobservable_model_turn_count_is_none_not_zero() -> None:
    """The distinction the whole fix rests on, at its smallest."""

    assert _observed_model_turns(_run(model_turns_observable=False)) is None
    assert _observed_model_turns(_run(model_turns_observable=True)) == 0
    assert _observed_model_turns(_run(model_turns_observable=None)) == 0
    assert (
        _observed_model_turns(
            _run(model_turns_observable=True, model_requests=3)
        )
        == 3
    )


def test_a_custom_run_never_passes_the_model_turn_budget_vacuously() -> None:
    """The report has to say the budget was not measured, and it does."""

    evaluation = evaluate_run(_scenario(), _run(model_turns_observable=False))

    observability = next(
        assertion
        for assertion in evaluation.assertions
        if assertion.assertion_id == "model_turn_observability"
    )
    assert observability.result == Verdict.INCONCLUSIVE
    assert observability.missing_evidence == (MODEL_TURNS_UNOBSERVABLE,)
    assert "not measured" in observability.rationale

    budgets = next(
        assertion
        for assertion in evaluation.assertions
        if assertion.assertion_id == "resource_budgets"
    )
    evidence = next(
        item
        for item in evaluation.evidence
        if item.evidence_id in budgets.supporting_evidence_ids
    )
    assert evidence.data["model_turns"] is None, "an unobserved count became a number"
    assert evidence.data["model_turns_observable"] is False


def test_the_observability_note_does_not_by_itself_sink_the_case() -> None:
    """Honest about one budget, not paralysed about the rest.

    Everything else in the scenario -- tool calls, wall clock, declared token
    and cost budgets -- was measured. Turning the whole case INCONCLUSIVE would
    trade one overclaim for another, and would make custom targets unusable.
    """

    evaluation = evaluate_run(_scenario(), _run(model_turns_observable=False))

    observability = next(
        assertion
        for assertion in evaluation.assertions
        if assertion.assertion_id == "model_turn_observability"
    )
    assert observability.required is False
    assert evaluation.verdict == Verdict.PASS


def test_a_declared_model_turn_constraint_is_inconclusive_not_passed() -> None:
    """A scenario that explicitly asks the question gets a blocking non-answer."""

    constraint = TrajectoryConstraint(
        criterion_id="max_model_turns",
        description="The agent answers within the model-turn budget",
        kind=TrajectoryConstraintKind.MAX_MODEL_TURNS,
        parameters={"maximum": 4},
        oracle_ids=("oracle-1",),
    )

    evaluation = evaluate_run(
        _scenario(trajectory_constraints=(constraint,)),
        _run(model_turns_observable=False),
    )

    declared = next(
        assertion
        for assertion in evaluation.assertions
        if assertion.assertion_id == "max_model_turns"
    )
    assert declared.result == Verdict.INCONCLUSIVE
    assert declared.missing_evidence == (MODEL_TURNS_UNOBSERVABLE,)
    assert declared.required is True
    assert evaluation.verdict == Verdict.INCONCLUSIVE, (
        "a criterion the developer asked for and AgentCheck could not evaluate "
        "must not leave the case looking verified"
    )


def test_an_observing_adapter_still_evaluates_model_turns_normally() -> None:
    """The existing adapters are unaffected: absence of the key means observable."""

    constraint = TrajectoryConstraint(
        criterion_id="max_model_turns",
        description="The agent answers within the model-turn budget",
        kind=TrajectoryConstraintKind.MAX_MODEL_TURNS,
        parameters={"maximum": 2},
        oracle_ids=("oracle-1",),
    )
    scenario = _scenario(trajectory_constraints=(constraint,))

    within = evaluate_run(
        scenario, _run(model_turns_observable=None, model_requests=2)
    )
    over = evaluate_run(
        scenario, _run(model_turns_observable=None, model_requests=5, run_id="run-2")
    )

    assert _assertion(within, "max_model_turns").result == Verdict.PASS
    assert _assertion(over, "max_model_turns").result == Verdict.FAIL
    for evaluation in (within, over):
        assert not [
            assertion
            for assertion in evaluation.assertions
            if assertion.assertion_id == "model_turn_observability"
        ], "an observing adapter must not get the not-measurable note"


def test_an_observing_adapter_still_fails_an_exceeded_budget() -> None:
    evaluation = evaluate_run(
        _scenario(), _run(model_turns_observable=None, model_requests=99)
    )

    budgets = _assertion(evaluation, "resource_budgets")
    assert budgets.result == Verdict.FAIL
    evidence = next(
        item
        for item in evaluation.evidence
        if item.evidence_id in budgets.supporting_evidence_ids
    )
    assert evidence.data["model_turns"] == 99
    assert "max_model_turns" in evidence.data["exceeded"]


def _assertion(evaluation: Any, assertion_id: str) -> Any:
    return next(
        item for item in evaluation.assertions if item.assertion_id == assertion_id
    )


def test_no_fake_model_events_are_emitted_for_a_custom_run(tmp_path: Path) -> None:
    """The count stays unobservable because nothing invented an observation."""

    config = _write_target(tmp_path)

    result = run_scenario_in_subprocess(tmp_path, config, _scenario(), "no-fakes")

    run = result.require_value()
    kinds = {event.event_type for event in run.events}
    assert CanonicalEventType.MODEL_REQUEST not in kinds
    assert CanonicalEventType.MODEL_RESPONSE not in kinds
    assert run.metadata["model_turns_observable"] is False
    assert _observed_model_turns(run) is None


def test_the_agent_turn_budget_is_still_enforced_and_named_as_its_own_thing(
    tmp_path: Path,
) -> None:
    """Enforcement did not go away with the claim to have measured model turns.

    The runtime bounds the conversation by the turns it drives. That bound is
    real, and a run that hits it terminates -- which is where a reader should
    see it, rather than in a model-turn count nobody measured.
    """

    from agentcheck.domain import ResourceBudgets

    config = _write_target(tmp_path)
    scenario = _scenario().model_copy(
        update={
            "followup_turns": (
                ConversationTurn(
                    turn_id="t1", role=ConversationRole.USER, content="And again?"
                ),
            ),
            "resource_budgets": ResourceBudgets(max_model_turns=1),
            "fingerprint": "",
        }
    )

    result = run_scenario_in_subprocess(tmp_path, config, scenario, "turn-budget")

    run = result.require_value()
    assert run.termination == RunTermination.MAX_MODEL_TURNS
    assert run.metadata["followups_undelivered"] == 1
    assert "turn budget" in (run.termination_reason or "")
    assert _observed_model_turns(run) is None, (
        "enforcing a turn budget must not be mistaken for observing model turns"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_a_custom_config_round_trips_through_the_ordinary_loader(
    tmp_path: Path,
) -> None:
    _write_target(tmp_path)

    root, config = load_config(tmp_path)

    assert root == tmp_path.resolve()
    assert config.adapter == "custom"
    assert config.entrypoint == "agent.py:agent"
    assert AgentCheckConfig().adapter == "openai_agents", "the default is unchanged"


def test_init_writes_a_custom_config_through_the_existing_command(
    tmp_path: Path,
) -> None:
    """`init` already accepted every adapter in the config Literal; it still does."""

    (tmp_path / "agent.py").write_text(CUSTOM_TARGET, encoding="utf-8")

    config_path = write_initial_config(tmp_path, adapter="custom")

    written = json.loads(config_path.read_text(encoding="utf-8"))
    assert written["adapter"] == "custom"
    assert load_config(tmp_path)[1].adapter == "custom"


def test_an_unknown_adapter_value_is_still_refused() -> None:
    with pytest.raises(ValueError):
        AgentCheckConfig(adapter="my_framework")


def test_an_invalid_entrypoint_is_refused_before_any_import(tmp_path: Path) -> None:
    _write_target(tmp_path)
    (tmp_path / "agentcheck.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.config.v1",
                "adapter": "custom",
                "entrypoint": "../outside.py:agent",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        load_config(tmp_path)


def test_a_missing_entrypoint_source_is_named(tmp_path: Path) -> None:
    (tmp_path / "agentcheck.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.config.v1",
                "adapter": "custom",
                "entrypoint": "agent.py:agent",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as raised:
        load_config(tmp_path)

    assert "entrypoint source does not exist" in str(raised.value)
    assert "agent.py" in str(raised.value)


def test_the_printed_integration_skeleton_is_a_working_custom_agent(
    tmp_path: Path,
) -> None:
    """`init --adapter custom` prints this; a skeleton that does not run is worse
    than none, so the printed text is the thing under test."""

    from agentcheck.cli import _CUSTOM_INTEGRATION_SKELETON

    (tmp_path / "agent.py").write_text(_CUSTOM_INTEGRATION_SKELETON, encoding="utf-8")
    write_initial_config(tmp_path, adapter="custom")

    _root, _config, result = inspect_target(str(tmp_path))

    spec = result.require_value()
    assert result.preflight_issues == ()
    assert [item.value.name for item in spec.tools.items] == ["lookup_account"]


# ---------------------------------------------------------------------------
# Entrypoint and contract errors, as a developer meets them
# ---------------------------------------------------------------------------


def _issue_codes(root: Path) -> set[str]:
    _root, _config, result = inspect_target(str(root))
    if result.infrastructure_error is not None:
        return {result.infrastructure_error.code}
    return {issue.code for issue in result.preflight_issues}


def test_a_missing_symbol_names_the_symbol(tmp_path: Path) -> None:
    _write_target(tmp_path, "other = 1\n")

    _root, _config, result = inspect_target(str(tmp_path))

    assert result.infrastructure_error is not None
    assert result.infrastructure_error.code == "target_attribute_missing"
    assert "agent" in result.infrastructure_error.message


def test_an_object_that_is_not_a_custom_agent_gets_one_diagnosis(
    tmp_path: Path,
) -> None:
    """Three missing members invite the developer to fix the wrong object.

    A config pointing at a dict, a module, or another framework's agent is a
    wrong-object mistake, not three separate omissions, and the message says
    which adapter such a target probably wants instead.
    """

    _write_target(tmp_path, 'agent = {"not": "an agent"}\n')

    _root, _config, result = inspect_target(str(tmp_path))

    assert [issue.code for issue in result.preflight_issues] == ["not_a_custom_agent"]
    message = result.preflight_issues[0].message
    assert "dict" in message
    assert "start(message, tools)" in message
    assert "PydanticAI" in message


def test_a_partial_implementation_gets_the_specific_gap(tmp_path: Path) -> None:
    """Something that is clearly trying to be a custom agent is told what is missing."""

    _write_target(tmp_path, CUSTOM_TARGET.replace("    def resume(", "    def cont("))

    assert _issue_codes(tmp_path) == {"missing_resume"}


def test_missing_tools_is_reported_without_running_the_agent(tmp_path: Path) -> None:
    _write_target(
        tmp_path,
        CUSTOM_TARGET.replace(
            "    tools: Sequence[ToolDefinition] = (LOOKUP,)\n", ""
        ),
    )

    codes = _issue_codes(tmp_path)

    assert "missing_tools_declaration" in codes
    assert "not_a_custom_agent" not in codes, (
        "start/resume are present, so this is an incomplete agent, not the wrong object"
    )


def test_a_malformed_tool_declaration_is_reported(tmp_path: Path) -> None:
    _write_target(
        tmp_path,
        CUSTOM_TARGET.replace(
            "    tools: Sequence[ToolDefinition] = (LOOKUP,)",
            "    tools = (LOOKUP, lambda account_id: None)",
        ),
    )

    assert "invalid_tool_definition" in _issue_codes(tmp_path)


def test_a_duplicate_tool_name_is_reported(tmp_path: Path) -> None:
    _write_target(
        tmp_path,
        CUSTOM_TARGET.replace(
            "    tools: Sequence[ToolDefinition] = (LOOKUP,)",
            "    tools: Sequence[ToolDefinition] = (LOOKUP, LOOKUP)",
        ),
    )

    assert "duplicate_tool_name" in _issue_codes(tmp_path)


def test_an_unsafe_tool_schema_is_reported(tmp_path: Path) -> None:
    _write_target(
        tmp_path,
        CUSTOM_TARGET.replace(
            '        "additionalProperties": False,\n    },\n)',
            '        "additionalProperties": False,\n    },\n)\n'
            'LOOKUP = LOOKUP.model_copy(update={"input_schema": '
            '{"$ref": "https://example.invalid/s.json"}})',
        ),
    )

    assert "invalid_tool_schema" in _issue_codes(tmp_path)


def test_a_mismatched_async_turn_method_is_reported(tmp_path: Path) -> None:
    """Only `start` turned async, `resume` left sync -- a contract mismatch."""

    _write_target(
        tmp_path,
        CUSTOM_TARGET.replace(
            "    def start(self, message", "    async def start(self, message"
        ),
    )

    assert "mismatched_turn_method_concurrency" in _issue_codes(tmp_path)


def test_no_contract_error_requires_running_the_target(tmp_path: Path) -> None:
    """Inspection stays structural: a turn method that explodes is never called."""

    _write_target(
        tmp_path,
        CUSTOM_TARGET.replace(
            '        outcome = tools.call("lookup_account", {"account_id": "A-1"})',
            '        raise AssertionError("inspection executed a turn")',
        ),
    )

    _root, _config, result = inspect_target(str(tmp_path))

    spec = result.require_value()
    assert result.preflight_issues == ()
    assert [item.value.name for item in spec.tools.items] == ["lookup_account"]


# ---------------------------------------------------------------------------
# The whole developer path, offline
# ---------------------------------------------------------------------------


def test_inspect_generate_and_test_all_reach_the_custom_adapter(
    tmp_path: Path,
) -> None:
    """No custom-specific CLI branch: config, registry, generator, runner, evaluator.

    One test for the whole path on purpose. Each stage feeding the next is the
    property worth pinning; a per-stage test could pass while the seam between
    two of them did not exist.
    """

    _write_target(tmp_path)

    _root, _config, inspected = inspect_target(str(tmp_path))
    spec = inspected.require_value()
    assert spec.identity.framework.value == "custom"
    assert inspected.preflight_issues == ()

    # The built-in suite is domain-specific, so a custom target reaches cases
    # the same way any other unfamiliar target does: schema-boundary cases
    # frozen from its own declared tool schemas.
    generated = generate_suite(str(tmp_path))
    assert generated.suite.scenarios, "no scenarios were derived for a custom target"
    assert generated.suite_path.is_file()

    execution = execute_suite(str(tmp_path))
    assert execution.evaluations
    assert all(evaluation.assertions for evaluation in execution.evaluations)
    assert all(
        run.metadata.get("framework") == "custom" for run in execution.runs
    ), "every scenario ran through the custom adapter"
    assert all(
        run.termination == RunTermination.COMPLETED for run in execution.runs
    )


def test_every_custom_case_report_states_the_model_turn_limitation(
    tmp_path: Path,
) -> None:
    """The honesty fix survives the real generator and runner, not just a unit test."""

    _write_target(tmp_path)
    generate_suite(str(tmp_path))

    execution = execute_suite(str(tmp_path))

    for evaluation in execution.evaluations:
        ids = [assertion.assertion_id for assertion in evaluation.assertions]
        assert "model_turn_observability" in ids, (
            f"case {evaluation.scenario_id} reported budgets without saying model "
            "turns were not measured"
        )


def test_the_replay_manifest_binds_a_custom_target(tmp_path: Path) -> None:
    """SpecBinding accepts the value, so a custom run stays replayable."""

    from agentcheck.replay.manifest import SpecBinding

    binding = SpecBinding(
        spec_id="agentspec-x", adapter="custom", entrypoint="agent.py:agent"
    )
    restored = SpecBinding.model_validate_json(binding.model_dump_json())

    assert restored.adapter == "custom"
    assert restored == binding
