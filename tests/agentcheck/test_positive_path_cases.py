"""Action-path cases so a valid tool call is possible, not merely forbidden.

Boundary cases only ever ask for an invalid call, so a well-behaved agent
passes an entire generated suite by never calling a tool. That leaves the
trajectory policies vacuous: a duplicate side effect or a fabricated success
can only be observed once a call actually happens. Validated against the real
customer-service target, every generated case was a negative one and the model
correctly declined all of them.

These cases supply a complete, in-contract argument set and ask for the tool's
declared action. Whether that warrants a call stays the agent's decision, which
is the only way the resulting trajectory says anything about the agent.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, function_tool
from pydantic import BaseModel

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.generate.boundaries import build_positive_path_cases
from agentcheck.generate.suite import CaseOrigin, build_frozen_suite
from agentcheck.schema_safety import offline_validator


@function_tool
def update_seat(confirmation_number: str, new_seat: str) -> str:
    """Update the seat for a given confirmation number."""
    raise AssertionError("original handler must never run")


@function_tool
def lookup_account(account_id: str) -> str:
    """Look up an account."""
    raise AssertionError("original handler must never run")


def _spec(*tools: Any, output_type: Any = None):
    agent = Agent(
        name="Target",
        instructions="Assist the customer.",
        tools=list(tools),
        output_type=output_type,
        model="gpt-4.1-mini",
    )
    return OpenAIAgentsAdapter().inspect(agent)


def _positive(spec, seed: int = 1729):
    """Scenarios only; build_positive_path_cases now also reports coverage."""

    return tuple(case.scenario for case in build_positive_path_cases(spec, seed=seed))


def test_positive_case_is_generated_from_a_valid_tool_schema() -> None:
    cases = _positive(_spec(update_seat))

    assert len(cases) == 1
    scenario = cases[0]
    assert scenario.conversation_turns
    assert "source:positive_path" in scenario.dimension_tags


def test_every_required_argument_is_supplied_and_schema_valid() -> None:
    """The request must be actionable: nothing required may be missing."""

    spec = _spec(update_seat)
    definition = next(item.value for item in spec.tools.items)
    arguments = _positive(spec)[0].allowed_tool_behavior[0].arguments_match

    required = set(definition.input_schema.get("required") or ())
    assert required and required.issubset(arguments)
    # Proved against the tool's own declared schema, not merely assumed.
    assert offline_validator(definition.input_schema).is_valid(dict(arguments))


def test_generated_values_respect_declared_constraints() -> None:
    """Enums and bounds come from the shared baseline builder, so they hold."""

    @function_tool
    def book(cabin: str, seats: int) -> str:
        """Book seats in a cabin."""
        raise AssertionError("original handler must never run")

    spec = _spec(book)
    definition = next(item.value for item in spec.tools.items)
    arguments = _positive(spec)[0].allowed_tool_behavior[0].arguments_match

    assert offline_validator(definition.input_schema).is_valid(dict(arguments))
    assert isinstance(arguments["seats"], int)
    assert isinstance(arguments["cabin"], str)


def test_the_request_asks_for_the_action_without_naming_the_tool_call() -> None:
    """The model must decide; a scripted instruction would author the result."""

    content = _positive(_spec(update_seat))[0].conversation_turns[0].content

    # Phrased from the tool's own declared description.
    assert "Update the seat" in content
    # Supplies the values rather than commanding a call.
    assert "confirmation_number is" in content
    assert "Use the update_seat tool with exactly these arguments" not in content


def test_declining_to_call_is_not_a_defect() -> None:
    """No required_tool_behavior: a schema cannot prove the agent must act."""

    scenario = _positive(_spec(update_seat))[0]

    assert scenario.required_tool_behavior == ()
    assert scenario.forbidden_tool_behavior == ()
    # The evaluable assertion holds vacuously when no call is made.
    assert scenario.trajectory_constraints
    assert scenario.trajectory_constraints[0].kind.value == "no_duplicate_side_effect"


def test_one_case_per_tool_with_a_constructible_baseline() -> None:
    """The documented rule, so suite size stays predictable."""

    cases = _positive(_spec(update_seat, lookup_account))

    assert len(cases) == 2
    tools = {tag.split(":", 1)[1] for c in cases for tag in c.dimension_tags if tag.startswith("tool:")}
    assert tools == {"update_seat", "lookup_account"}


def test_tools_without_a_constructible_baseline_are_omitted() -> None:
    """Fail conservatively: no invented semantics for an unreadable contract."""

    @function_tool
    def opaque() -> str:
        """Takes nothing."""
        raise AssertionError("original handler must never run")

    # A zero-parameter tool exposes no parameters, so no baseline exists.
    assert _positive(_spec(opaque)) == ()


def test_generation_is_deterministic() -> None:
    spec = _spec(update_seat)

    first, second = _positive(spec), _positive(spec)

    assert [c.fingerprint for c in first] == [c.fingerprint for c in second]
    assert [c.scenario_id for c in first] == [c.scenario_id for c in second]


def test_negative_boundary_coverage_is_preserved() -> None:
    """Positive cases add coverage; they must not replace the negative ones."""

    suite = build_frozen_suite(_spec(update_seat), AgentCheckConfig(), seed=1729)
    origins = [case.lineage.origin for case in suite.cases]

    assert CaseOrigin.SCHEMA_BOUNDARY in origins
    assert CaseOrigin.POSITIVE_PATH in origins
    assert origins.count(CaseOrigin.POSITIVE_PATH) == 1
    assert suite.rejected == ()


def test_suite_identity_reflects_the_added_case() -> None:
    """A suite with an action path is a different suite, and stays stable."""

    spec = _spec(update_seat)
    suite = build_frozen_suite(spec, AgentCheckConfig(), seed=1729)

    assert "positive_path" in suite.provenance.sources
    assert suite.fingerprint == build_frozen_suite(spec, AgentCheckConfig(), seed=1729).fingerprint


def test_frozen_serialization_preserves_the_positive_case() -> None:
    """Replay reads the artifact, so the case must survive a round trip."""

    suite = build_frozen_suite(_spec(update_seat), AgentCheckConfig(), seed=1729)
    dumped = suite.model_dump(mode="json")

    positive = [c for c in dumped["cases"] if c["lineage"]["origin"] == "positive_path"]
    assert len(positive) == 1
    scenario = positive[0]["scenario"]
    assert scenario["allowed_tool_behavior"][0]["arguments_match"]
    assert scenario["trajectory_constraints"][0]["kind"] == "no_duplicate_side_effect"
    assert scenario["tool_fixtures"][0]["tool_name"] == "update_seat"


def test_configured_wall_clock_applies_to_positive_cases() -> None:
    """PR #19 budget propagation must cover the new origin automatically."""

    suite = build_frozen_suite(
        _spec(update_seat),
        AgentCheckConfig(scenario_wall_clock_seconds=180),
        seed=1729,
    )
    positive = [c for c in suite.cases if c.lineage.origin is CaseOrigin.POSITIVE_PATH]

    assert positive
    assert all(c.scenario.resource_budgets.wall_clock_seconds == 180 for c in positive)


def test_derived_policies_attach_to_the_positive_case() -> None:
    """The point of the action path: policies stop being vacuous."""

    from agentcheck.policies import derive_tool_risk_pack

    spec = _spec(update_seat)
    pack = derive_tool_risk_pack(spec)
    assert pack is not None
    suite = build_frozen_suite(
        spec, AgentCheckConfig(), seed=1729, policy_packs=(pack,)
    )
    positive = next(
        c for c in suite.cases if c.lineage.origin is CaseOrigin.POSITIVE_PATH
    )

    kinds = {t.kind.value for t in positive.scenario.trajectory_constraints}
    assert "no_duplicate_side_effect" in kinds
    assert "no_fabricated_success" in {
        o.kind.value for o in positive.scenario.output_criteria
    }


def test_generation_never_executes_a_tool_handler() -> None:
    """Every handler above raises; generation reads declarations only."""

    cases = _positive(_spec(update_seat, lookup_account))

    assert len(cases) == 2


class _Plan(BaseModel):
    steps: list[str]


def test_output_schema_targets_are_unaffected() -> None:
    """A tool-less structured-output target still derives no action path."""

    assert _positive(_spec(output_type=_Plan)) == ()
