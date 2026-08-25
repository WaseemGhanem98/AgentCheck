"""The fault family must be bounded by the fault cap, not the positive-path cap.

`build_outcome_variant_cases` breaks on `_MAX_POSITIVE_SCENARIOS_PER_SPEC` (24)
at the top of its per-tool loop, while `MAX_FAULT_VARIANT_SCENARIOS` (64) is
only consulted deeper inside. The outer break always fires first, so the
documented bound is unreachable and the fault family silently stops at 24.

Measured on tau-bench (Arize-ai/phoenix @ 08922f3): 16 tools, 6 classified
state-changing. Generation grants a fault family to exactly five of them and
stops at 26 scenarios; `modify_user_address` -- state-changing, with a
constructible baseline, and last alphabetically -- receives none.
`docs/fault-testing.md` says generation "stops at MAX_FAULT_VARIANT_SCENARIOS
across the whole spec". It stops at 24.

The existing bound test in `test_zero_argument_actions.py` asserts
`len(scenarios) <= MAX_FAULT_VARIANT_SCENARIOS` and passes for the wrong
reason: 24 is trivially under 64. These tests are written so that cannot happen
again -- they pin the cap that actually binds, from both directions.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, function_tool

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.generate.boundaries import (
    MAX_FAULT_VARIANT_SCENARIOS,
    build_outcome_variant_cases,
)

SEED = 1729


def _spec(*tools: Any):
    return OpenAIAgentsAdapter().inspect(
        Agent(name="T", instructions="Assist.", tools=list(tools), model="gpt-4.1-mini")
    )


def _state_changing_tools(count: int) -> list[Any]:
    """`update_thing_NN` -- the classifier reads `update` as state-changing."""

    made: list[Any] = []
    for index in range(count):
        def _fn(record_id: str) -> str:
            raise AssertionError("original handler must never run")

        _fn.__name__ = f"update_thing_{index:02d}"
        _fn.__doc__ = f"Update thing {index}."
        made.append(function_tool(_fn))
    return made


def _covered_tools(scenarios: tuple[Any, ...]) -> set[str]:
    return {
        tag[len("tool:") :]
        for scenario in scenarios
        for tag in scenario.dimension_tags
        if tag.startswith("tool:")
    }


# --- the defect -------------------------------------------------------------


def test_generation_does_not_stop_at_the_positive_path_cap() -> None:
    """The regression. Generation used to stop at 24 scenarios.

    Ten state-changing tools yield five fault cases each -- 50 scenarios, well
    inside the fault cap of 64 and well past the positive-path cap of 24.
    """
    spec = _spec(*_state_changing_tools(10))

    scenarios = build_outcome_variant_cases(spec, seed=SEED)

    # Ten state-changing tools yield five variants each. Anything near 24 means
    # the positive-path cap is still the binding limit; `> 24` alone would pass
    # on 25 and prove nothing.
    assert len(scenarios) >= 45, (
        f"generation stopped at {len(scenarios)} scenarios; the positive-path "
        "cap is still bounding the fault family"
    )


def test_every_state_changing_tool_inside_the_cap_gets_a_family() -> None:
    """No tool should lose its fault family while the fault cap has room."""
    tools = _state_changing_tools(10)
    spec = _spec(*tools)

    covered = _covered_tools(build_outcome_variant_cases(spec, seed=SEED))

    assert covered == {f"update_thing_{i:02d}" for i in range(10)}


# --- the real cap must still bind -------------------------------------------


def test_the_fault_cap_still_bounds_generation() -> None:
    """Enough tools to exceed 64 must be cut at 64, not left unbounded."""
    spec = _spec(*_state_changing_tools(40))

    scenarios = build_outcome_variant_cases(spec, seed=SEED)

    assert len(scenarios) <= MAX_FAULT_VARIANT_SCENARIOS


def test_the_bound_is_actually_reached_rather_than_vacuously_satisfied() -> None:
    """Guards against the assertion that passed for the wrong reason.

    With 40 state-changing tools the cap must genuinely bind; if this comes back
    far below 64, some other limit is stopping generation early again.
    """
    spec = _spec(*_state_changing_tools(40))

    scenarios = build_outcome_variant_cases(spec, seed=SEED)

    assert len(scenarios) > MAX_FAULT_VARIANT_SCENARIOS - 10, (
        f"only {len(scenarios)} scenarios generated from 40 state-changing "
        "tools; the fault cap is not the binding limit"
    )


def test_generation_stays_deterministic_at_the_cap() -> None:
    spec = _spec(*_state_changing_tools(40))

    first = build_outcome_variant_cases(spec, seed=SEED)
    second = build_outcome_variant_cases(spec, seed=SEED)

    assert [s.scenario_id for s in first] == [s.scenario_id for s in second]


def test_read_only_tools_are_still_skipped_entirely() -> None:
    @function_tool
    def get_thing(record_id: str) -> str:
        """Look up a thing."""
        raise AssertionError("original handler must never run")

    spec = _spec(get_thing, *_state_changing_tools(3))

    assert "get_thing" not in _covered_tools(build_outcome_variant_cases(spec, seed=SEED))
