"""Adversarial coverage for ToolGateway's plan/commit reservation split.

The central claim under test: fixture ownership, invocation-index assignment,
and budget consumption are decided once, deterministically, when a batch is
planned -- and nothing about committing those reservations later, in any
order, can change what was already decided. This is what makes it safe for an
adapter to let several simulated calls from one launch group be dispatched
concurrently: whichever one's task happens to run first, the gateway's
bookkeeping already knows the answer.

No threads, no sleeps, no timing assumptions: every test drives ordering
explicitly to prove the result does not depend on it.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentcheck.domain.agent_spec import ToolDefinition
from agentcheck.domain.run import ToolOutcomeStatus
from agentcheck.runner import (
    FixtureNotFoundError,
    ToolCallBlockedError,
    ToolGateway,
)


def _tool(name: str, *, state_changing: bool = True, destructive: bool = False) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        input_schema={
            "type": "object",
            "properties": {"user_id": {"type": "integer"}},
            "required": ["user_id"],
            "additionalProperties": False,
        },
        state_changing=state_changing,
        destructive=destructive,
        replaceable=True,
    )


def _gateway(tools: list[ToolDefinition], fixtures: list[dict[str, Any]]) -> ToolGateway:
    return ToolGateway(tools, fixtures, world={}, run_id="concurrency-test")


# --- fixture/invocation-index assignment is decided at plan time -----------


def test_two_calls_to_the_same_tool_get_distinct_invocation_indices_regardless_of_commit_order() -> (
    None
):
    gateway = _gateway(
        [_tool("delete_user")],
        [
            {
                "fixture_id": "first",
                "tool_name": "delete_user",
                "invocation_index": 1,
                "outcome": {"status": "success", "result": {"call": 1}},
            },
            {
                "fixture_id": "second",
                "tool_name": "delete_user",
                "invocation_index": 2,
                "outcome": {"status": "success", "result": {"call": 2}},
            },
        ],
    )
    calls = [("delete_user", {"user_id": 7}), ("delete_user", {"user_id": 7})]
    reservations = gateway.plan_batch(calls)

    assert reservations[0].invocation_index == 1
    assert reservations[1].invocation_index == 2

    # Commit in reverse order: the *plan* already decided who owns which
    # fixture, so this must not change which result each call gets.
    second_outcome = gateway.commit(reservations[1])
    first_outcome = gateway.commit(reservations[0])

    assert first_outcome.result == {"call": 1}
    assert second_outcome.result == {"call": 2}


def test_a_single_use_fixture_is_never_assigned_to_two_reservations() -> None:
    """Two same-stage calls competing for one single-use fixture: the second
    must fail closed at plan time, not race for it at commit time."""

    gateway = _gateway(
        [_tool("charge_card")],
        [
            {
                "fixture_id": "only-one",
                "tool_name": "charge_card",
                "outcome": {"status": "success", "result": {"charged": True}},
            }
        ],
    )
    calls = [("charge_card", {"user_id": 1}), ("charge_card", {"user_id": 1})]
    reservations = gateway.plan_batch(calls)

    assert reservations[0].blocked is None
    assert reservations[1].blocked is not None
    assert reservations[1].blocked.code == "fixture_not_found"

    first_outcome = gateway.commit(reservations[0])
    assert first_outcome.result == {"charged": True}
    with pytest.raises(FixtureNotFoundError):
        gateway.commit(reservations[1])


def test_planning_is_deterministic_and_order_is_not_inferred_from_commit_calls() -> None:
    """Planning the same batch twice on independent gateways yields identical
    fixture/invocation-index assignment -- the plan does not consult anything
    about how or when commit will later be called."""

    def build() -> ToolGateway:
        return _gateway(
            [_tool("reserve_inventory")],
            [
                {
                    "fixture_id": f"slot-{index}",
                    "tool_name": "reserve_inventory",
                    "invocation_index": index,
                    "outcome": {"status": "success", "result": {"slot": index}},
                }
                for index in (1, 2, 3)
            ],
        )

    calls = [("reserve_inventory", {"user_id": 42})] * 3
    left = build().plan_batch(calls)
    right = build().plan_batch(calls)

    assert [r.invocation_index for r in left] == [r.invocation_index for r in right]
    assert [r.blocked for r in left] == [r.blocked for r in right]


# --- budget/retry bookkeeping follows decision order, not commit order -----


def _signature_status(gateway: ToolGateway, tool_name: str, arguments: dict[str, Any]) -> Any:
    import json

    key = (tool_name, json.dumps(arguments, sort_keys=True, separators=(",", ":")))
    return gateway._signature_status.get(key)  # white-box: internal bookkeeping under test


def _charge_card_gateway() -> ToolGateway:
    return _gateway(
        [_tool("charge_card")],
        [
            {
                "fixture_id": "err",
                "tool_name": "charge_card",
                "invocation_index": 1,
                "outcome": {"status": "error"},
            },
            {
                "fixture_id": "ok",
                "tool_name": "charge_card",
                "invocation_index": 2,
                "outcome": {"status": "success", "result": {"charged": True}},
            },
        ],
    )


def test_retry_budget_reflects_decision_order_even_when_committed_in_reverse() -> None:
    """call1 errors; call2 (same signature, same stage) is planned right
    after, so its retry check sees call1's decided status immediately --
    exactly as it would if both had run serially. Which one *commits* first
    afterwards must not change that."""

    gateway = _charge_card_gateway()
    calls = [("charge_card", {"user_id": 9}), ("charge_card", {"user_id": 9})]
    reservations = gateway.plan_batch(calls)
    assert reservations[0].blocked is None
    assert reservations[1].blocked is None

    # Decision order already fixed the final signature status.
    assert _signature_status(gateway, "charge_card", {"user_id": 9}) == ToolOutcomeStatus.SUCCESS

    # Commit call 2 first, then call 1 -- reversed from decision order.
    second_outcome = gateway.commit(reservations[1])
    first_outcome = gateway.commit(reservations[0])

    assert first_outcome.status == ToolOutcomeStatus.ERROR
    assert second_outcome.status == ToolOutcomeStatus.SUCCESS

    # Committing in reverse order must not have disturbed the decision-order
    # status a later call would see.
    assert _signature_status(gateway, "charge_card", {"user_id": 9}) == ToolOutcomeStatus.SUCCESS


def test_commit_order_cannot_reintroduce_a_stale_retry_budget_charge() -> None:
    """Guards the specific bug this design exists to prevent: if commit order
    were allowed to drive `_signature_status` (as it did before this split),
    committing the erroring call *last* would leave the signature looking
    like it still needs a retry charge, even though decision order says
    otherwise."""

    gateway = _charge_card_gateway()
    calls = [("charge_card", {"user_id": 9}), ("charge_card", {"user_id": 9})]
    reservations = gateway.plan_batch(calls)

    # Commit the *second* (success) reservation first, then the error.
    gateway.commit(reservations[1])
    gateway.commit(reservations[0])

    # If commit order (not decision order) drove `_signature_status`, this
    # would now read ERROR -- the last one committed -- instead of SUCCESS.
    assert _signature_status(gateway, "charge_card", {"user_id": 9}) == ToolOutcomeStatus.SUCCESS


# --- invoke() stays exactly equivalent to plan_one + commit -----------------


def test_invoke_is_still_exactly_plan_one_then_commit() -> None:
    gateway = _gateway(
        [_tool("read_balance", state_changing=False)],
        [
            {
                "fixture_id": "f1",
                "tool_name": "read_balance",
                "outcome": {"status": "success", "result": {"balance": 5}},
            }
        ],
    )
    outcome = gateway.invoke("read_balance", {"user_id": 1})
    assert outcome.result == {"balance": 5}
    assert outcome.status == ToolOutcomeStatus.SUCCESS
    assert len(gateway.attempts) == 1
    assert len(gateway.outcomes) == 1


def test_blocked_reservation_from_unknown_tool_still_raises_on_commit() -> None:
    gateway = _gateway([_tool("read_balance", state_changing=False)], [])
    reservation = gateway.plan_one("delete_everything", {"user_id": 1})
    assert reservation.blocked is not None
    with pytest.raises(ToolCallBlockedError):
        gateway.commit(reservation)
    assert len(gateway.attempts) == 1
    assert gateway.outcomes[0].status == ToolOutcomeStatus.BLOCKED
