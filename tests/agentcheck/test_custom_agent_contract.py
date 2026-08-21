"""The custom-agent contracts, and what they deliberately do not contain.

These are typing tests, and typing tests earn their place here for one reason:
the safety argument for custom agents is a property of the *shape* of the
contract. AgentCheck cannot execute ``cancel_order`` because it is never handed
``cancel_order``. If a future change adds a handler field for convenience, that
argument silently becomes "we promise not to call it", and nothing else in the
suite would notice. `test_contract_never_accepts_a_callable_handler` is the one
that would.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, Sequence, get_type_hints

import pytest

from agentcheck import CustomAgentProtocol, ToolRuntime, TurnResult
from agentcheck.adapters import FrameworkAdapter, OpenAIAgentsAdapter, PydanticAIAdapter
from agentcheck.domain.agent_spec import ToolDefinition
from agentcheck.domain.run import ToolOutcome, ToolOutcomeStatus


# --------------------------------------------------------------------------
# The common runtime contract
# --------------------------------------------------------------------------


def test_framework_adapter_declares_the_universal_run_contract() -> None:
    """The four-method contract a custom-agent adapter will have to satisfy.

    ``run()`` is already abstract here -- this pins it, because it is the method
    the worker calls to execute a prepared target and the reason a custom agent
    can become a fourth adapter rather than a second execution path through the
    runner. Nothing in this PR adds it; the test exists so a later refactor
    cannot quietly drop it.
    """

    assert FrameworkAdapter.__abstractmethods__ == frozenset(
        {"inspect", "preflight", "prepare", "run"}
    )


@pytest.mark.parametrize("adapter", [OpenAIAgentsAdapter, PydanticAIAdapter])
def test_existing_adapters_still_satisfy_the_interface(adapter: type) -> None:
    """Declaring run() must not have made a shipped adapter abstract."""

    assert issubclass(adapter, FrameworkAdapter)
    assert not getattr(adapter, "__abstractmethods__", frozenset()), (
        f"{adapter.__name__} became abstract, which means the declared contract "
        "does not match what it already implements."
    )
    adapter()  # constructible, so nothing was left unimplemented


@pytest.mark.parametrize("adapter", [OpenAIAgentsAdapter, PydanticAIAdapter])
def test_adapter_run_signature_matches_the_declared_contract(adapter: type) -> None:
    """The declaration describes the adapters; the adapters were not reshaped."""

    declared = inspect.signature(FrameworkAdapter.run)
    actual = inspect.signature(adapter.run)
    assert list(actual.parameters) == list(declared.parameters), (
        f"{adapter.__name__}.run parameters drifted from the declared contract"
    )
    for name, parameter in declared.parameters.items():
        assert actual.parameters[name].kind == parameter.kind, name
        assert actual.parameters[name].default == parameter.default, name


@pytest.mark.parametrize("adapter", [OpenAIAgentsAdapter, PydanticAIAdapter])
def test_adapter_run_is_still_a_coroutine_function(adapter: type) -> None:
    assert inspect.iscoroutinefunction(adapter.run)


# --------------------------------------------------------------------------
# The custom-agent contracts
# --------------------------------------------------------------------------


class _ExampleAgent:
    """The smallest thing that satisfies the protocol.

    Deliberately holds no callable for its tools: it declares them and reaches
    them only through the supplied runtime.
    """

    tools: Sequence[ToolDefinition] = (
        ToolDefinition(name="get_order", input_schema={"type": "object"}),
        ToolDefinition(
            name="cancel_order",
            input_schema={"type": "object"},
            state_changing=True,
            destructive=True,
        ),
    )

    def start(self, message: str, tools: ToolRuntime) -> TurnResult:
        outcome = tools.call("get_order", {"order_id": "W1"})
        return TurnResult(output=f"found {outcome.tool_name}", state={"seen": True})

    def resume(self, state: Any, message: str, tools: ToolRuntime) -> TurnResult:
        tools.call("cancel_order", {"order_id": "W1"})
        return TurnResult(output="cancelled", state=state)


class _RecordingRuntime:
    """A stand-in runtime. Records calls; executes nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def call(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        self.calls.append((name, dict(arguments)))
        return ToolOutcome(
            outcome_id=f"outcome-{len(self.calls)}",
            attempt_id=f"attempt-{len(self.calls)}",
            event_id=f"event-{len(self.calls)}",
            tool_name=name,
            status=ToolOutcomeStatus.SUCCESS,
            result={"ok": True},
        )


def test_a_minimal_agent_satisfies_the_protocol() -> None:
    assert isinstance(_ExampleAgent(), CustomAgentProtocol)


def test_a_runtime_stand_in_satisfies_the_tool_runtime_protocol() -> None:
    assert isinstance(_RecordingRuntime(), ToolRuntime)


def test_turns_thread_state_and_reach_tools_only_through_the_runtime() -> None:
    agent, runtime = _ExampleAgent(), _RecordingRuntime()

    first = agent.start("cancel my order", runtime)
    assert first.state == {"seen": True}

    second = agent.resume(first.state, "yes, cancel it", runtime)
    assert second.output == "cancelled"
    assert second.state == first.state, "opaque state is handed back unchanged"

    assert [name for name, _ in runtime.calls] == ["get_order", "cancel_order"]


def test_turn_result_keeps_state_opaque_and_unserialized() -> None:
    """State is an in-process handle, so it may be any object at all."""

    sentinel = object()
    result = TurnResult(output="done", state=sentinel)
    assert result.state is sentinel
    assert result.metadata == {}
    assert is_dataclass(TurnResult)


# --------------------------------------------------------------------------
# The property the whole design rests on
# --------------------------------------------------------------------------


def test_contract_never_accepts_a_callable_handler() -> None:
    """No path in the contract carries a real tool implementation.

    AgentCheck's zero-execution guarantee for declared tools is structural: it
    holds no reference to the function, so it cannot call it. A field named
    `handler`, `callable`, `fn` or similar would quietly downgrade that to a
    promise.
    """

    banned = ("handler", "callable", "func", "fn", "impl", "target", "tool_callable")

    for name in [field.name for field in fields(TurnResult)]:
        assert not any(token in name.lower() for token in banned), name

    for name in ToolDefinition.model_fields:
        assert not any(token in name.lower() for token in banned), (
            f"ToolDefinition.{name} looks like a handler reference"
        )

    hints = get_type_hints(_ExampleAgent)
    assert "Callable" not in str(hints.get("tools", "")), (
        "declared tools must be data, not callables"
    )


def test_tool_runtime_exposes_only_simulated_calls() -> None:
    """One method in, canonical outcome out -- no escape hatch."""

    public = [
        name
        for name in dir(ToolRuntime)
        if not name.startswith("_") and callable(getattr(ToolRuntime, name, None))
    ]
    assert public == ["call"], f"ToolRuntime grew extra surface: {public}"

    returns = inspect.signature(ToolRuntime.call).return_annotation
    assert "ToolOutcome" in str(returns), (
        "a simulated call must return the canonical outcome, so a custom loop "
        "sees simulated failures and ambiguity the way a real one would"
    )


def test_custom_contracts_are_importable_from_both_paths() -> None:
    from agentcheck import custom

    assert custom.ToolRuntime is ToolRuntime
    assert custom.TurnResult is TurnResult
    assert custom.CustomAgentProtocol is CustomAgentProtocol
    assert set(custom.__all__) == {"CustomAgentProtocol", "ToolRuntime", "TurnResult"}


def test_custom_agent_execution_is_not_wired_up_yet() -> None:
    """PR A is contracts only; nothing may execute a custom agent."""

    from agentcheck.config import AgentCheckConfig
    from agentcheck.runner import worker

    adapters = getattr(worker, "_ADAPTERS", {})
    assert "custom" not in adapters, "a custom adapter was registered too early"

    annotation = str(AgentCheckConfig.model_fields["adapter"].annotation)
    assert "custom" not in annotation, "config gained a custom adapter option"
