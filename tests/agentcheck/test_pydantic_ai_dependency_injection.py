"""PydanticAI dependency injection: real-world compatibility, kept safe.

Real-world validation of PR #46 found that `deps_type`/`RunContext[Deps]` is
the dominant pattern in realistic PydanticAI agents (both official examples
inspected used it), and AgentCheck rejected all of them outright. This file
proves the fix is genuinely safe, not merely permissive: every tool handler
here raises immediately if the original ever runs, and every dependency
placeholder raises immediately if anything actually reads it.

No provider is contacted anywhere in this file.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse, ToolCallPart

from agentcheck.adapters import PydanticAIAdapter
from agentcheck.adapters.pydantic_ai import DependencyAccessError, _InertDependencies
from agentcheck.domain import (
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolFixture,
    ToolOutcomeStatus,
    WorldStateEffect,
)
from agentcheck.runner import ToolGateway

from tests.agentcheck.test_pydantic_ai_adapter import (
    ORIGINAL_HANDLER_CALLS,
    Deps,
    _agent,
    _definitions,
    _prepare,
    _run,
    _same_stage_response,
    _script,
    _text,
)


# --- _InertDependencies itself: the fail-closed placeholder -----------------


def test_inert_dependencies_blocks_ordinary_attribute_access() -> None:
    deps = _InertDependencies()

    with pytest.raises(DependencyAccessError, match="client"):
        deps.client  # noqa: B018 - intentional access under test

    with pytest.raises(DependencyAccessError, match="database"):
        deps.database.query("select 1")  # never reaches .query


def test_inert_dependencies_permits_dunder_duck_typing_checks() -> None:
    """A framework doing `hasattr(deps, "__iter__")`-style introspection gets
    an ordinary "no", not a crash -- only a concrete attribute name (a real
    attempt to *use* a dependency) fails loudly."""

    deps = _InertDependencies()

    assert hasattr(deps, "__iter__") is False
    assert hasattr(deps, "__len__") is False
    assert repr(deps) == "<AgentCheck simulated dependencies: no real dependency object is constructed>"


def test_inert_dependencies_treats_a_named_attribute_as_a_real_access_attempt() -> None:
    """Unlike dunder introspection, `hasattr` on an ordinary name is not
    silently False -- it is exactly the kind of access this must fail loudly
    on, so `hasattr` itself propagates the error rather than swallowing it."""

    deps = _InertDependencies()

    with pytest.raises(DependencyAccessError):
        hasattr(deps, "client")


def test_inert_dependencies_never_performs_real_io() -> None:
    """The placeholder itself proves nothing network/filesystem-shaped ever
    runs -- there is no code path in it that could."""

    deps = _InertDependencies()
    for name in ("open", "connect", "request", "get", "post", "execute", "fetch"):
        with pytest.raises(DependencyAccessError):
            getattr(deps, name)


# --- preflight + inspection: ctx is excluded from the model-facing schema --


def test_ctx_first_argument_is_excluded_from_the_declared_schema() -> None:
    agent = _agent()

    @agent.tool
    def get_weather(ctx: RunContext[Deps], city: str, units: str = "metric") -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    spec = PydanticAIAdapter().inspect(agent)
    tool = next(item.value for item in spec.tools.items if item.value.name == "get_weather")

    assert set(tool.input_schema["properties"]) == {"city", "units"}
    assert "ctx" not in tool.input_schema["properties"]
    assert tool.input_schema["required"] == ["city"]


def test_zero_visible_parameters_besides_ctx_yields_an_empty_schema() -> None:
    agent = _agent()

    @agent.tool
    def get_user_preferences(ctx: RunContext[Deps]) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    spec = PydanticAIAdapter().inspect(agent)
    tool = next(item.value for item in spec.tools.items if item.value.name == "get_user_preferences")

    assert tool.input_schema.get("properties", {}) == {}


def test_multiple_context_aware_tools_all_pass_preflight() -> None:
    agent = _agent(deps_type=Deps)

    @agent.tool
    def get_lat_lng(ctx: RunContext[Deps], city: str) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    @agent.tool
    def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    report = PydanticAIAdapter().preflight(agent)

    assert report.supported, [(i.code, i.message) for i in report.issues]


# --- end-to-end: reconstructed context-aware tools run safely --------------


def test_a_context_aware_read_tool_runs_through_the_gateway_not_the_handler() -> None:
    agent = _agent(deps_type=Deps)

    @agent.tool
    def get_weather(ctx: RunContext[Deps], city: str) -> str:
        ORIGINAL_HANDLER_CALLS.append("get_weather")
        raise AssertionError("original handler must never run")

    agent.model = _script(
        ModelResponse(parts=[ToolCallPart("get_weather", {"city": "NYC"})]), _text("done")
    )
    gateway = ToolGateway(
        _definitions(agent),
        [ToolFixture(fixture_id="f", tool_name="get_weather", outcome=SimulatedToolOutcome(
            status=SimulatedToolStatus.SUCCESS, result={"temp_c": 21}
        ))],
    )
    prepared = _prepare(agent, gateway)
    run = _run(prepared, "weather?")

    assert ORIGINAL_HANDLER_CALLS == []
    assert run.tool_outcomes[0].result == {"temp_c": 21}


def test_a_context_aware_state_changing_tool_runs_through_the_gateway() -> None:
    agent = _agent(deps_type=Deps)

    @agent.tool
    def update_preferences(ctx: RunContext[Deps], units: str) -> str:
        ORIGINAL_HANDLER_CALLS.append("update_preferences")
        raise AssertionError("original handler must never run")

    agent.model = _script(
        ModelResponse(parts=[ToolCallPart("update_preferences", {"units": "metric"})]), _text("done")
    )
    gateway = ToolGateway(
        _definitions(agent),
        [ToolFixture(
            fixture_id="f",
            tool_name="update_preferences",
            outcome=SimulatedToolOutcome(
                status=SimulatedToolStatus.SUCCESS,
                result={"ok": True},
                state_effects=(WorldStateEffect(path="units", after="metric"),),
            ),
        )],
        world={"units": None},
    )
    prepared = _prepare(agent, gateway)
    _run(prepared, "set units")

    assert ORIGINAL_HANDLER_CALLS == []
    assert gateway.world.get("units") == "metric"
    assert len(gateway.state_transitions) == 1


def test_a_dependency_access_inside_the_original_handler_would_be_unreachable() -> None:
    """Even a handler written to legitimately use ctx.deps never runs at all
    -- the tripwire proves the reconstructed invoker is what actually
    executes, never the target's function body."""

    agent = _agent(deps_type=Deps)

    @agent.tool
    def get_secret(ctx: RunContext[Deps], key: str) -> str:
        ORIGINAL_HANDLER_CALLS.append("get_secret")
        return ctx.deps.token  # would read a real dependency if this ever ran

    agent.model = _script(ModelResponse(parts=[ToolCallPart("get_secret", {"key": "x"})]), _text("done"))
    gateway = ToolGateway(
        _definitions(agent),
        [ToolFixture(fixture_id="f", tool_name="get_secret", outcome=SimulatedToolOutcome(
            status=SimulatedToolStatus.SUCCESS, result={"masked": True}
        ))],
    )
    prepared = _prepare(agent, gateway)
    run = _run(prepared, "get it")

    assert ORIGINAL_HANDLER_CALLS == []
    assert run.tool_outcomes[0].result == {"masked": True}


# --- fault families remain reachable for context-aware tools ---------------


@pytest.mark.parametrize(
    "status,kwargs",
    [
        (SimulatedToolStatus.ERROR, {"error_code": "boom", "error_message": "boom"}),
        (SimulatedToolStatus.TIMEOUT, {"error_code": "timeout", "error_message": "timed out"}),
        (SimulatedToolStatus.EMPTY, {}),
        (SimulatedToolStatus.MALFORMED, {}),
        (SimulatedToolStatus.PARTIAL, {}),
        (SimulatedToolStatus.STALE, {}),
    ],
)
def test_fault_statuses_reach_a_context_aware_tool(status: SimulatedToolStatus, kwargs: dict[str, Any]) -> None:
    agent = _agent(deps_type=Deps)

    @agent.tool
    def get_weather(ctx: RunContext[Deps], city: str) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    agent.model = _script(ModelResponse(parts=[ToolCallPart("get_weather", {"city": "NYC"})]), _text("done"))
    gateway = ToolGateway(
        _definitions(agent),
        [ToolFixture(fixture_id="f", tool_name="get_weather", outcome=SimulatedToolOutcome(status=status, **kwargs))],
    )
    prepared = _prepare(agent, gateway)
    run = _run(prepared, "weather?")

    assert ORIGINAL_HANDLER_CALLS == []
    assert run.tool_outcomes[0].status == status


def test_ambiguous_timeout_still_applies_to_a_destructive_context_aware_tool() -> None:
    from agentcheck.domain import ToolDefinition

    agent = _agent(deps_type=Deps)

    @agent.tool
    def cancel_subscription(ctx: RunContext[Deps], user_id: str) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    agent.model = _script(ModelResponse(parts=[ToolCallPart("cancel_subscription", {"user_id": "u1"})]), _text("done"))
    definitions = [
        d.model_copy(update={"state_changing": True, "destructive": True})
        if isinstance(d, ToolDefinition) and d.name == "cancel_subscription"
        else d
        for d in _definitions(agent)
    ]
    gateway = ToolGateway(
        definitions,
        [ToolFixture(
            fixture_id="f",
            tool_name="cancel_subscription",
            outcome=SimulatedToolOutcome(
                status=SimulatedToolStatus.TIMEOUT, error_code="timeout", error_message="timed out"
            ),
        )],
    )
    prepared = _prepare(agent, gateway)
    run = _run(prepared, "cancel it")

    assert ORIGINAL_HANDLER_CALLS == []
    outcome = run.tool_outcomes[0]
    assert outcome.status == ToolOutcomeStatus.TIMEOUT
    assert outcome.error is not None
    assert outcome.error.code == "ambiguous_timeout"


# --- concurrent context-aware tools ------------------------------------------


def test_two_context_aware_tools_in_one_launch_group_stay_independent() -> None:
    """Mirrors the milestone's own example: get_lat_lng + get_user_preferences
    decided together, both taking RunContext, dispatched concurrently."""

    agent = _agent(deps_type=Deps)

    @agent.tool
    def get_lat_lng(ctx: RunContext[Deps], city: str) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    @agent.tool
    def get_user_preferences(ctx: RunContext[Deps]) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    agent.model = _script(
        ModelResponse(
            parts=[
                ToolCallPart("get_lat_lng", {"city": "NYC"}, tool_call_id="c1"),
                ToolCallPart("get_user_preferences", {}, tool_call_id="c2"),
            ]
        ),
        _text("done"),
    )
    gateway = ToolGateway(
        _definitions(agent),
        [
            ToolFixture(fixture_id="ll", tool_name="get_lat_lng", outcome=SimulatedToolOutcome(
                status=SimulatedToolStatus.SUCCESS, result={"lat": 40.7, "lng": -74.0}
            )),
            ToolFixture(fixture_id="prefs", tool_name="get_user_preferences", outcome=SimulatedToolOutcome(
                status=SimulatedToolStatus.SUCCESS, result={"units": "imperial"}
            )),
        ],
    )
    prepared = _prepare(agent, gateway)
    run = _run(prepared, "weather please")

    assert ORIGINAL_HANDLER_CALLS == []
    by_tool = {outcome.tool_name: outcome for outcome in run.tool_outcomes}
    assert by_tool["get_lat_lng"].result == {"lat": 40.7, "lng": -74.0}
    assert by_tool["get_user_preferences"].result == {"units": "imperial"}

    # Launch-group evidence must still be correct for context-aware calls.
    from agentcheck.evaluate.launch import analyze_launches

    analysis = analyze_launches(run)
    attempts_by_tool = {a.tool_name: a for a in run.tool_attempts}
    assert analysis.same_launch_group(
        attempts_by_tool["get_lat_lng"].attempt_id,
        attempts_by_tool["get_user_preferences"].attempt_id,
    ) is True


def test_two_context_aware_same_stage_calls_get_deterministic_fixtures_regardless_of_order() -> None:
    agent = _agent(deps_type=Deps)

    @agent.tool
    def cancel_subscription(ctx: RunContext[Deps], user_id: str) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    agent.model = _script(_same_stage_response("cancel_subscription", {"user_id": "u1"}), _text("done"))
    gateway = ToolGateway(
        _definitions(agent),
        [
            ToolFixture(fixture_id="first", tool_name="cancel_subscription", invocation_index=1,
                        outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result={"call": 1})),
            ToolFixture(fixture_id="second", tool_name="cancel_subscription", invocation_index=2,
                        outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result={"call": 2})),
        ],
    )
    prepared = _prepare(agent, gateway)
    run = _run(prepared, "cancel it twice")

    assert ORIGINAL_HANDLER_CALLS == []
    assert [outcome.result for outcome in run.tool_outcomes] == [{"call": 1}, {"call": 2}]


def test_repeated_concurrent_context_aware_runs_are_deterministic() -> None:
    def build_and_run() -> tuple[Any, ...]:
        agent = _agent(deps_type=Deps)

        @agent.tool
        def cancel_subscription(ctx: RunContext[Deps], user_id: str) -> str:  # pragma: no cover
            raise AssertionError("original handler must never run")

        agent.model = _script(_same_stage_response("cancel_subscription", {"user_id": "u1"}), _text("done"))
        gateway = ToolGateway(
            _definitions(agent),
            [
                ToolFixture(fixture_id="first", tool_name="cancel_subscription", invocation_index=1,
                            outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result={"call": 1})),
                ToolFixture(fixture_id="second", tool_name="cancel_subscription", invocation_index=2,
                            outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result={"call": 2})),
            ],
        )
        prepared = _prepare(agent, gateway)
        run = _run(prepared, "cancel it twice")
        assert ORIGINAL_HANDLER_CALLS == []
        return tuple(outcome.result["call"] for outcome in run.tool_outcomes)

    results = {build_and_run() for _ in range(8)}
    assert results == {(1, 2)}


# --- prerequisite / dependent chains across turns, with deps ----------------


def test_prerequisite_and_dependent_context_aware_calls_in_separate_turns_are_observed_before() -> None:
    from agentcheck.evaluate.launch import analyze_launches

    agent = _agent(deps_type=Deps)

    @agent.tool
    def get_lat_lng(ctx: RunContext[Deps], city: str) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    @agent.tool
    def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> str:  # pragma: no cover
        raise AssertionError("original handler must never run")

    agent.model = _script(
        ModelResponse(parts=[ToolCallPart("get_lat_lng", {"city": "NYC"}, tool_call_id="c1")]),
        ModelResponse(parts=[ToolCallPart("get_weather", {"lat": 1.0, "lng": 2.0}, tool_call_id="c2")]),
        _text("done"),
    )
    gateway = ToolGateway(
        _definitions(agent),
        [
            ToolFixture(fixture_id="ll", tool_name="get_lat_lng", outcome=SimulatedToolOutcome(
                status=SimulatedToolStatus.SUCCESS, result={"lat": 1.0, "lng": 2.0}
            )),
            ToolFixture(fixture_id="w", tool_name="get_weather", outcome=SimulatedToolOutcome(
                status=SimulatedToolStatus.SUCCESS, result={"temp_c": 10}
            )),
        ],
    )
    prepared = _prepare(agent, gateway)
    run = _run(prepared, "weather please", max_turns=8)

    assert ORIGINAL_HANDLER_CALLS == []
    analysis = analyze_launches(run)
    attempts_by_tool = {a.tool_name: a for a in run.tool_attempts}
    assert analysis.same_launch_group(
        attempts_by_tool["get_lat_lng"].attempt_id, attempts_by_tool["get_weather"].attempt_id
    ) is False
    assert analysis.observed_before(
        attempts_by_tool["get_lat_lng"].attempt_id, attempts_by_tool["get_weather"].attempt_id
    ) is True
