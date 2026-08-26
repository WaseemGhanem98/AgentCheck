"""An async ``start``/``resume`` pair: what it can do, and what it still can't.

A custom agent whose own loop needs a real ``await`` (a genuine async
provider client, for example) no longer has to fake a synchronous boundary
around it -- ``start``/``resume`` may both be coroutine functions, given an
``AsyncToolRuntime`` whose ``call()`` is awaited instead of called directly.

That is a real capability gap closed (today, ANY async turn method is
rejected outright at preflight, even one that never touches concurrent
dispatch). It is deliberately *not* the same thing as concurrent tool-call
dispatch: ``AsyncToolRuntime.call`` has no internal ``await`` of its own, so
even "concurrent-looking" syntax (``asyncio.gather``, ``asyncio.create_task``)
still executes each call to completion, strictly in the order it was
scheduled, before the next begins. This file proves both halves: the new
capability works end to end, and the old determinism guarantee survives
being wrapped in `gather`-style syntax rather than being silently upgraded
into something AgentCheck cannot actually prove.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from agentcheck import AsyncToolRuntime, TurnResult
from agentcheck.domain import ToolDefinition

from tests.agentcheck.test_custom_agent_adapter import (
    CANCEL_ORDER,
    GET_ORDER,
    _fixture,
    _prepare,
    _run,
    _turn,
)


def _codes(target: Any) -> set[str]:
    from agentcheck.adapters import CustomAgentAdapter

    return {issue.code for issue in CustomAgentAdapter().preflight(target).issues}


class AsyncSupportAgent:
    """Mirrors ``SupportAgent`` from test_custom_agent_adapter.py, but async.

    ``start`` awaits a fake async "model" call (``asyncio.sleep(0)``) before
    calling a tool -- proving a real ``await`` inside the turn boundary works,
    which is exactly what the synchronous contract could not offer.
    """

    tools: Sequence[ToolDefinition] = (GET_ORDER, CANCEL_ORDER)

    def __init__(self) -> None:
        self.turns: list[str] = []

    async def start(self, message: str, tools: AsyncToolRuntime) -> TurnResult:
        self.turns.append(message)
        await asyncio.sleep(0)  # stands in for a real async provider call
        order = await tools.call("get_order", {"order_id": "W123"})
        status = (order.result or {}).get("status")
        return TurnResult(
            output=f"Order W123 is {status}. Cancelling is permanent -- confirm?",
            state={"order_id": "W123"},
        )

    async def resume(self, state: Any, message: str, tools: AsyncToolRuntime) -> TurnResult:
        self.turns.append(message)
        await asyncio.sleep(0)
        if "yes" not in message.lower():
            return TurnResult(output="Leaving the order in place.", state=state)
        outcome = await tools.call("cancel_order", {"order_id": state["order_id"]})
        return TurnResult(
            output=f"Cancelled W123 ({outcome.status.value}).",
            state={**state, "cancelled": True},
        )


def test_a_consistently_async_agent_is_supported() -> None:
    assert _codes(AsyncSupportAgent()) == set()


def test_an_async_agent_runs_end_to_end_through_the_real_gateway() -> None:
    agent = AsyncSupportAgent()
    adapter, prepared, _gateway = _prepare(
        agent,
        [
            _fixture("get_order", {"status": "pending"}),
            _fixture("cancel_order", {"cancelled": True}),
        ],
    )
    run = _run(
        adapter,
        prepared,
        "Please cancel order W123.",
        followups=[_turn("t1", "yes, cancel it")],
    )

    assert agent.turns == ["Please cancel order W123.", "yes, cancel it"]
    assert [outcome.tool_name for outcome in run.tool_outcomes] == [
        "get_order",
        "cancel_order",
    ]
    assert run.final_output == "Cancelled W123 (success)."


class GatherAgent:
    """A single turn that issues two independent tool calls via ``asyncio.gather``.

    Nothing here claims the two calls were "decided together" -- there is no
    model event for AgentCheck to derive that from for a custom agent, so no
    same-stage/launch-group fact is produced either way. What this proves is
    narrower and provable: however the agent chooses to *schedule* the two
    coroutines, they still execute strictly in the order given to ``gather``,
    every time, because ``AsyncToolRuntime.call`` never yields mid-call.
    """

    tools: Sequence[ToolDefinition] = (GET_ORDER, CANCEL_ORDER)

    async def start(self, message: str, tools: AsyncToolRuntime) -> TurnResult:
        first, second = await asyncio.gather(
            tools.call("get_order", {"order_id": "A"}),
            tools.call("get_order", {"order_id": "B"}),
        )
        return TurnResult(
            output=f"{first.result} then {second.result}",
            state={},
        )

    async def resume(self, state: Any, message: str, tools: AsyncToolRuntime) -> TurnResult:
        return TurnResult(output="unused", state=state)


def test_gather_dispatched_calls_still_execute_in_scheduling_order() -> None:
    """Repeated runs must all observe the same order: proof, not one sample."""

    for _ in range(25):
        adapter, prepared, _gateway = _prepare(
            GatherAgent(),
            [
                _fixture("get_order", "order-A", invocation_index=1),
                _fixture("get_order", "order-B", invocation_index=2),
            ],
        )
        run = _run(adapter, prepared, "look up both orders")

        assert [outcome.result for outcome in run.tool_outcomes] == [
            "order-A",
            "order-B",
        ]
        assert run.final_output == "order-A then order-B"


def test_preflight_reports_a_mismatched_sync_async_pair_before_any_turn_runs() -> None:
    class Mixed:
        tools: Sequence[ToolDefinition] = (GET_ORDER,)

        async def start(self, message: str, tools: AsyncToolRuntime) -> TurnResult:  # type: ignore[override]
            return TurnResult(output="unused")

        def resume(self, state: Any, message: str, tools: Any) -> TurnResult:
            return TurnResult(output="unused", state=state)

    assert "mismatched_turn_method_concurrency" in _codes(Mixed())
