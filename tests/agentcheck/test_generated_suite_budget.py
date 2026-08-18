"""The configured wall clock must reach generated suites, not only the built-in one.

`scenario_wall_clock_seconds` was honoured by the built-in suite but generated
cases were built with the `ResourceBudgets` default, so they stayed at 30s. The
built-in suite only matches targets declaring its specific tool set, which means
the budget was unavailable for exactly the third-party targets that need it: a
real reasoning model is slower than the scripted one these defaults assume, and
cases timed out as `worker_timeout` despite the config asking for more.
"""

from __future__ import annotations

from agents import Agent, function_tool
from pydantic import BaseModel

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.generate.suite import build_frozen_suite


class Verdict(BaseModel):
    approved: bool


@function_tool
def update_seat(confirmation_number: str, new_seat: str) -> str:
    """Update the seat on a booking."""
    raise AssertionError("original handler must never run")


def _spec():
    agent = Agent(
        name="Seat Agent",
        instructions="Help with seating.",
        tools=[update_seat],
        output_type=Verdict,
        model="gpt-4.1-mini",
    )
    return OpenAIAgentsAdapter().inspect(agent)


def _suite(wall_clock: float | None):
    config = (
        AgentCheckConfig()
        if wall_clock is None
        else AgentCheckConfig(scenario_wall_clock_seconds=wall_clock)
    )
    return build_frozen_suite(_spec(), config, seed=1729)


def test_generated_boundary_cases_receive_the_configured_budget() -> None:
    suite = _suite(180)
    boundary = [c for c in suite.cases if c.lineage.origin.value == "schema_boundary"]

    assert boundary, "expected boundary cases for a target declaring a tool"
    assert all(
        case.scenario.resource_budgets.wall_clock_seconds == 180 for case in boundary
    )


def test_generated_output_schema_cases_receive_the_configured_budget() -> None:
    suite = _suite(180)
    output_schema = [c for c in suite.cases if c.lineage.origin.value == "output_schema"]

    assert output_schema, "expected an output-schema case for a declared output_type"
    assert all(
        case.scenario.resource_budgets.wall_clock_seconds == 180
        for case in output_schema
    )


def test_unset_config_preserves_the_existing_default() -> None:
    """The default path must not move: only an explicit request changes it."""

    assert AgentCheckConfig().scenario_wall_clock_seconds is None

    suite = _suite(None)
    assert suite.cases
    assert all(
        case.scenario.resource_budgets.wall_clock_seconds == 30.0
        for case in suite.cases
    )


def test_frozen_suite_serializes_the_effective_budget() -> None:
    """The artifact must carry the budget the run will actually be held to.

    Worker timeout and evaluated budget both read this value, so a suite that
    serialized the default while the worker used something else would let the
    two disagree.
    """

    dumped = _suite(180).model_dump(mode="json")
    budgets = {
        case["scenario"]["resource_budgets"]["wall_clock_seconds"]
        for case in dumped["cases"]
    }

    assert budgets == {180}


def test_changing_the_budget_changes_the_suite_identity() -> None:
    """A different budget is a different suite, and the default stays stable."""

    default = _suite(None)
    raised = _suite(180)

    assert default.fingerprint != raised.fingerprint
    assert default.fingerprint == _suite(None).fingerprint
    assert raised.fingerprint == _suite(180).fingerprint
    # Re-budgeting must not add or drop coverage.
    assert len(default.cases) == len(raised.cases)
