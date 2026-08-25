"""A destructive action that takes no arguments must still be fault-tested.

Schema-boundary generation asks "what invalid call can be constructed", and for
a tool that declares no parameters the honest answer is "none". That answer was
being reused as "no valid argument object exists either", so the whole
behavioural fault family was skipped: the loop that builds tool-failure,
degraded-evidence and ambiguous-timeout cases bails on a missing baseline.

The result was that `cancel_trip()` -- destructive, zero-argument, found in the
official OpenAI customer-support sample -- produced exactly one scenario, and
that scenario carried no fixtures, no trajectory constraints and no output
criteria. It could not fail. `delete_account(account_id)`, identical in every
way that matters except for taking an argument, produced ten.

An empty object is a valid in-contract argument set for a schema that declares
no parameters. These tests pin that, and try to break the fix in the directions
that would make it unsafe: by inventing arguments, by escaping the bound, by
letting an unvalidated schema through, or by disturbing parameterised tools.
"""

from __future__ import annotations

from typing import Any

from agents import Agent, function_tool

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.generate.boundaries import (
    MAX_FAULT_VARIANT_SCENARIOS,
    build_outcome_variant_cases,
    derive_boundaries,
)

SEED = 1729


@function_tool
def cancel_trip() -> str:
    """Cancel the traveller's upcoming trip and note the refund."""
    raise AssertionError("original handler must never run")


@function_tool
def delete_account(account_id: str) -> str:
    """Permanently delete the named account."""
    raise AssertionError("original handler must never run")


@function_tool
def list_trips() -> str:
    """List the traveller's trips."""
    raise AssertionError("original handler must never run")


def _spec(*tools: Any):
    return OpenAIAgentsAdapter().inspect(
        Agent(name="T", instructions="Assist.", tools=list(tools), model="gpt-4.1-mini")
    )


def _suffixes(scenarios: tuple[Any, ...], tool: str) -> set[str]:
    prefix = f"action-{tool.replace('_', '-')}-"
    return {
        s.scenario_id[len(prefix) :]
        for s in scenarios
        if s.scenario_id.startswith(prefix)
    }


# --- the defect itself ------------------------------------------------------


def test_a_zero_argument_destructive_tool_gets_its_fault_family() -> None:
    """The regression. Before the fix this set was empty."""
    scenarios = build_outcome_variant_cases(_spec(cancel_trip), seed=SEED)
    suffixes = _suffixes(scenarios, "cancel_trip")

    assert "tool-failure" in suffixes
    for degraded in (
        "empty-response",
        "malformed-response",
        "partial-response",
        "stale-response",
    ):
        assert degraded in suffixes, f"{degraded} missing for a zero-argument tool"


def test_a_zero_argument_destructive_tool_keeps_its_ambiguous_timeout() -> None:
    """The duplicate-side-effect question is the whole point of a `cancel_trip()`."""
    scenarios = build_outcome_variant_cases(_spec(cancel_trip), seed=SEED)

    assert "ambiguous-outcome" in _suffixes(scenarios, "cancel_trip")


def test_taking_an_argument_is_not_what_earns_fault_coverage() -> None:
    """Two destructive tools that differ only in arity must be treated alike."""
    scenarios = build_outcome_variant_cases(_spec(cancel_trip, delete_account), seed=SEED)

    assert _suffixes(scenarios, "cancel_trip") == _suffixes(scenarios, "delete_account")


# --- the fix must not invent anything ---------------------------------------


def test_the_generated_call_carries_no_invented_arguments() -> None:
    """A zero-parameter schema admits exactly one in-contract call: the empty one.

    Supplying anything else would be an out-of-contract call dressed as a
    positive path, and the gateway is required to fail closed on those.
    """
    scenarios = build_outcome_variant_cases(_spec(cancel_trip), seed=SEED)

    seen_a_call = False
    for scenario in scenarios:
        for behavior in (
            *scenario.required_tool_behavior,
            *scenario.allowed_tool_behavior,
        ):
            if behavior.tool_name != "cancel_trip":
                continue
            seen_a_call = True
            assert behavior.arguments_match in (None, {}), (
                f"{scenario.scenario_id} invented arguments for a zero-parameter "
                f"schema: {behavior.arguments_match!r}"
            )

    for scenario in scenarios:
        for fixture in scenario.tool_fixtures:
            if fixture.tool_name != "cancel_trip":
                continue
            seen_a_call = True
            assert fixture.arguments_match in (None, {}), (
                f"{scenario.scenario_id} bound a fixture to invented arguments: "
                f"{fixture.arguments_match!r}"
            )

    assert seen_a_call, "no call contract was asserted at all; the test proved nothing"


def test_a_read_only_zero_argument_tool_still_gets_no_fault_family() -> None:
    """Arity was never the gate -- declared risk is. Retrying a lookup is ordinary."""
    scenarios = build_outcome_variant_cases(_spec(list_trips), seed=SEED)

    assert _suffixes(scenarios, "list_trips") == set()


def test_no_schema_boundary_is_invented_for_a_zero_parameter_tool() -> None:
    """There is genuinely nothing to probe at the boundary. That part was right."""
    assert derive_boundaries(_spec(cancel_trip).tools.items[0].value) == ()


# --- the fix must not disturb what already worked ---------------------------


def test_a_parameterised_tool_is_unchanged_by_the_fix() -> None:
    alone = _suffixes(build_outcome_variant_cases(_spec(delete_account), seed=SEED), "delete_account")
    together = _suffixes(
        build_outcome_variant_cases(_spec(cancel_trip, delete_account), seed=SEED),
        "delete_account",
    )

    assert alone == together


def test_generation_stays_deterministic() -> None:
    spec = _spec(cancel_trip, delete_account)
    first = build_outcome_variant_cases(spec, seed=SEED)
    second = build_outcome_variant_cases(spec, seed=SEED)

    assert [s.scenario_id for s in first] == [s.scenario_id for s in second]
    assert [s.fingerprint for s in first] == [s.fingerprint for s in second]


def test_generation_stays_bounded_when_every_tool_is_zero_argument() -> None:
    """Zero-argument tools must not be a way around the spec-wide cap."""

    def _make(index: int) -> Any:
        async def _fn() -> str:
            raise AssertionError("original handler must never run")

        _fn.__name__ = f"cancel_thing_{index}"
        _fn.__doc__ = f"Permanently delete thing {index}."
        return function_tool(_fn)

    spec = _spec(*[_make(i) for i in range(40)])
    scenarios = build_outcome_variant_cases(spec, seed=SEED)

    assert len(scenarios) <= MAX_FAULT_VARIANT_SCENARIOS
