"""PydanticAI tools must carry the risk markers the other adapter carries.

`classify_tool` says why it exists: "a tool definition must carry its risk
markers before a capability can be built from it". The OpenAI Agents adapter
calls it. The PydanticAI adapter hardcoded `state_changing=False,
destructive=False` and never did.

Everything downstream reads those two flags. Fault generation skips a tool that
is not state-changing, and the ambiguous-timeout case is destructive-only, so a
PydanticAI target received no tool-failure, no degraded-evidence and no
duplicate-side-effect case for any tool at all -- the whole systematic fault
capability was inert for one of the two supported frameworks.

It was visible in `inspect` as a contradiction: the summary counted zero
state-changing actions while the capability listing under it described the same
tool as state-changing and destructive, because capability extraction classifies
independently of the adapter.

These tests hold the two adapters to the same answer for the same tool.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.test import TestModel

from agentcheck.adapters import OpenAIAgentsAdapter, PydanticAIAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.generate.boundaries import build_outcome_variant_cases
from agentcheck.generate.suite import build_frozen_suite

SEED = 1729


def _pydantic_spec() -> Any:
    agent = PydanticAgent(TestModel(), system_prompt="You help customers manage orders.")

    @agent.tool_plain
    def cancel_subscription(customer_id: str) -> str:
        """Cancel the customer subscription."""
        raise AssertionError("original handler must never run")

    @agent.tool_plain
    def update_record(record_id: str) -> str:
        """Update a record."""
        raise AssertionError("original handler must never run")

    @agent.tool_plain
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        raise AssertionError("original handler must never run")

    return PydanticAIAdapter().inspect(agent)


def _openai_spec() -> Any:
    from agents import Agent, function_tool

    @function_tool
    def cancel_subscription(customer_id: str) -> str:
        """Cancel the customer subscription."""
        raise AssertionError("original handler must never run")

    @function_tool
    def update_record(record_id: str) -> str:
        """Update a record."""
        raise AssertionError("original handler must never run")

    @function_tool
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        raise AssertionError("original handler must never run")

    return OpenAIAgentsAdapter().inspect(
        Agent(
            name="T",
            instructions="You help customers manage orders.",
            tools=[cancel_subscription, update_record, lookup_order],
            model="gpt-4.1-mini",
        )
    )


def _markers(spec: Any) -> dict[str, tuple[bool, bool]]:
    return {
        item.value.name: (item.value.state_changing, item.value.destructive)
        for item in spec.tools.items
    }


# --- the defect -------------------------------------------------------------


def test_a_pydantic_tool_carries_risk_markers_at_all() -> None:
    """The regression. Every marker was hardcoded False."""
    markers = _markers(_pydantic_spec())

    assert markers["cancel_subscription"] == (True, True), (
        "a cancellation was not marked state-changing and destructive"
    )
    assert markers["update_record"] == (True, False)


def test_both_adapters_classify_the_same_tool_the_same_way() -> None:
    """Same name, same description, same declared schema -- same answer.

    Asserted as parity rather than against fixed expectations, so this test
    keeps holding if the shared classifier's judgement changes.
    """
    assert _markers(_pydantic_spec()) == _markers(_openai_spec())


def test_a_pydantic_target_now_gets_its_fault_family() -> None:
    """Nothing downstream fires while the flags are False."""
    scenarios = build_outcome_variant_cases(_pydantic_spec(), seed=SEED)
    titles = " ".join(s.title for s in scenarios)

    assert scenarios, "no fault scenario was generated for any PydanticAI tool"
    assert "cancel_subscription" in titles
    # Destructive-only: the duplicate-side-effect question this tool most needs.
    assert any("times out" in s.title for s in scenarios)


def test_the_inspect_summary_agrees_with_its_own_capability_listing() -> None:
    """The contradiction users actually saw: 0 state-changing, next to a
    capability described as state-changing."""
    spec = _pydantic_spec()

    counted = sum(1 for item in spec.tools.items if item.value.state_changing)
    from agentcheck.inspect.capabilities import extract_capabilities

    definitions = [item.value for item in spec.tools.items]
    extracted = extract_capabilities(definitions)
    described = sum(1 for cap in extracted if cap.capability.state_changing)

    assert counted == described


# --- the fix must not over-reach -------------------------------------------


def test_a_lookup_is_still_not_state_changing() -> None:
    """Parity means the same classifier, not a blanket "everything is risky"."""
    markers = _markers(_pydantic_spec())

    assert markers["lookup_order"] == (False, False)


def test_a_pydantic_suite_still_builds_and_stays_deterministic() -> None:
    spec = _pydantic_spec()
    first = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    second = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)

    assert first.fingerprint == second.fingerprint
    assert len(first.cases) > 0
