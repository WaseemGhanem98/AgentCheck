from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from agentcheck.domain.agent_spec import ToolDefinition
from agentcheck.domain.run import ToolOutcomeStatus
from agentcheck.domain.scenario import (
    ResourceBudgets,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolFixture,
    WorldStateEffect,
)
from agentcheck.runner import (
    BudgetExceeded,
    BudgetTracker,
    FixtureDefinitionError,
    FixtureNotFoundError,
    ToolCallBlockedError,
    ToolGateway,
    UnknownToolError,
    UnsafeToolSpecificationError,
    WorldSimulator,
    WorldStateError,
)


def _tool(
    name: str = "update_account",
    *,
    state_changing: bool = True,
    destructive: bool = False,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        input_schema={
            "type": "object",
            "properties": {
                "account_id": {"type": "string", "pattern": "^acct_[0-9]+$"},
                "level": {"type": "integer", "minimum": 1, "maximum": 3},
                "mode": {"type": "string", "enum": ["safe", "force"]},
                "when": {"type": "string", "format": "date"},
            },
            "required": ["account_id"],
            "additionalProperties": False,
        },
        state_changing=state_changing,
        destructive=destructive,
        replaceable=True,
    )


def test_world_state_is_deep_copied_and_effects_do_not_leak() -> None:
    source = {
        "accounts": {"acct_1": {"exists": True, "attempts": 0, "tags": []}},
        "a/b": {"~key": "pointer-value"},
    }
    first = WorldSimulator(source)
    second = WorldSimulator(source)

    source["accounts"]["acct_1"]["exists"] = False
    assert first.get("accounts.acct_1.exists") is True
    assert first.get("/a~1b/~0key") == "pointer-value"

    first.increment("accounts.acct_1.attempts", 2)
    first.append("accounts.acct_1.tags", "updated")
    first.set("accounts.acct_1.email", "new@example.test")
    first.delete("accounts.acct_1.exists")

    assert first.state["accounts"]["acct_1"] == {
        "attempts": 2,
        "tags": ["updated"],
        "email": "new@example.test",
    }
    assert second.state == {
        "accounts": {"acct_1": {"exists": True, "attempts": 0, "tags": []}},
        "a/b": {"~key": "pointer-value"},
    }

    leaked = first.snapshot()
    leaked["accounts"]["acct_1"]["tags"].append("caller-mutation")
    assert first.get("accounts.acct_1.tags") == ["updated"]


def test_world_effect_batch_rolls_back_when_one_effect_is_invalid() -> None:
    world = WorldSimulator({"counter": 1, "items": []})

    with pytest.raises(WorldStateError, match="append requires"):
        world.apply_effects(
            [
                {"op": "increment", "path": "counter", "delta": 1},
                {"op": "append", "path": "counter", "value": "invalid"},
            ]
        )

    assert world.state == {"counter": 1, "items": []}


def test_gateways_clone_shared_world_and_never_mutate_foreign_worlds() -> None:
    shared = WorldSimulator({"updates": 0})
    fixture = {
        "update_account": {
            "repeat": True,
            "status": "success",
            "effects": [{"op": "increment", "path": "updates"}],
        }
    }
    first = ToolGateway([_tool()], fixture, world=shared)
    second = ToolGateway([_tool()], fixture, world=shared)

    first.invoke("update_account", {"account_id": "acct_1"})

    assert first.world.get("updates") == 1
    assert second.world.get("updates") == 0
    assert shared.get("updates") == 0

    foreign = WorldSimulator({"updates": 0})
    second.invoke("update_account", {"account_id": "acct_2"}, foreign)
    assert second.world.get("updates") == 1
    assert foreign.get("updates") == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"account_id": "wrong"},
        {"account_id": "acct_1", "level": 0},
        {"account_id": "acct_1", "level": 4},
        {"account_id": "acct_1", "level": "2"},
        {"account_id": "acct_1", "mode": "unknown"},
        {"account_id": "acct_1", "when": "not-a-date"},
        {"account_id": "acct_1", "unexpected": True},
    ],
)
def test_gateway_blocks_json_schema_boundary_violations(arguments: dict[str, Any]) -> None:
    gateway = ToolGateway(
        [_tool()],
        {
            "update_account": {
                "status": "success",
                "result": {"ok": True},
            }
        },
    )

    outcome = gateway.invoke("update_account", arguments)

    assert outcome.status == ToolOutcomeStatus.BLOCKED
    assert outcome.error is not None
    assert outcome.error.code == "invalid_arguments"
    assert outcome.metadata["schema_valid"] is False
    assert len(gateway.attempts) == len(gateway.outcomes) == 1
    assert gateway.state_transitions == ()


def test_fixture_selection_supports_subset_exact_and_call_index() -> None:
    gateway = ToolGateway(
        [_tool(state_changing=False)],
        [
            {
                "fixture_id": "subset-first",
                "tool_name": "update_account",
                "arguments_match": {"account_id": "acct_1"},
                "invocation_index": 1,
                "outcome": {"status": "success", "result": {"selected": "subset"}},
            },
            {
                "fixture_id": "exact-second",
                "tool_name": "update_account",
                "match_args": {"account_id": "acct_1"},
                "match_mode": "exact",
                "call_index": 2,
                "status": "success",
                "result": {"selected": "exact"},
            },
        ],
    )

    first = gateway.invoke("update_account", {"account_id": "acct_1", "level": 2})
    second = gateway.invoke("update_account", {"account_id": "acct_1"})

    assert first.result == {"selected": "subset"}
    assert second.result == {"selected": "exact"}
    assert [outcome.metadata["fixture_id"] for outcome in gateway.outcomes] == [
        "subset-first",
        "exact-second",
    ]


def test_fixture_selection_prefers_specific_match_and_rejects_duplicate_ids() -> None:
    gateway = ToolGateway(
        [_tool(state_changing=False)],
        [
            {
                "fixture_id": "broad",
                "tool_name": "update_account",
                "match_args": {"account_id": "acct_1"},
                "status": "success",
                "result": {"selected": "broad"},
            },
            {
                "fixture_id": "exact",
                "tool_name": "update_account",
                "match_args": {"account_id": "acct_1"},
                "match_mode": "exact",
                "status": "success",
                "result": {"selected": "exact"},
            },
        ],
    )

    assert gateway.invoke("update_account", {"account_id": "acct_1"}).result == {
        "selected": "exact"
    }
    assert gateway.invoke(
        "update_account", {"account_id": "acct_1", "level": 2}
    ).result == {"selected": "broad"}

    duplicate = {
        "fixture_id": "same-id",
        "tool_name": "update_account",
        "status": "success",
    }
    with pytest.raises(FixtureDefinitionError, match="duplicate fixture_id"):
        ToolGateway([_tool()], [duplicate, duplicate])


def test_unknown_and_unfixture_tools_fail_closed_but_leave_evidence() -> None:
    gateway = ToolGateway(
        [_tool(state_changing=False)],
        {"update_account": {"status": "success", "result": {"ok": True}}},
    )

    with pytest.raises(UnknownToolError, match="allowlist"):
        gateway.invoke("live_unknown_tool", {})
    assert gateway.outcomes[-1].status == ToolOutcomeStatus.BLOCKED
    assert gateway.outcomes[-1].error is not None
    assert gateway.outcomes[-1].error.code == "unknown_tool"

    gateway.invoke("update_account", {"account_id": "acct_1"})
    with pytest.raises(FixtureNotFoundError, match="no controlled fixture"):
        gateway.invoke("update_account", {"account_id": "acct_1"})

    assert [item.status for item in gateway.outcomes] == [
        ToolOutcomeStatus.BLOCKED,
        ToolOutcomeStatus.SUCCESS,
        ToolOutcomeStatus.BLOCKED,
    ]
    assert len(gateway.events) == 6


def test_gateway_refuses_live_handlers_instead_of_calling_them() -> None:
    calls: list[dict[str, Any]] = []

    def original_handler(arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append(arguments)
        return {"unsafe": True}

    with pytest.raises(UnsafeToolSpecificationError, match="live field"):
        ToolGateway(
            [
                {
                    "name": "dangerous",
                    "input_schema": {"type": "object"},
                    "handler": original_handler,
                }
            ],
            [],
        )

    with pytest.raises(FixtureDefinitionError, match="live field"):
        ToolGateway(
            [_tool(state_changing=False)],
            {
                "update_account": {
                    "status": "success",
                    "handler": original_handler,
                }
            },
        )

    assert calls == []


def test_gateway_safety_scan_does_not_execute_arbitrary_descriptors() -> None:
    touched = {"value": False}

    class HostileDefinition:
        @property
        def name(self) -> str:
            touched["value"] = True
            raise AssertionError("descriptor executed")

    with pytest.raises(UnsafeToolSpecificationError, match="requires a name"):
        ToolGateway([HostileDefinition()], [])

    assert touched["value"] is False


def test_destructive_timeout_can_apply_an_ambiguous_state_transition() -> None:
    fixed_time = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
    tool = _tool("delete_account", destructive=True)
    fixture = ToolFixture(
        fixture_id="ambiguous-delete-timeout",
        tool_name="delete_account",
        arguments_match={"account_id": "acct_1"},
        outcome=SimulatedToolOutcome(
            status=SimulatedToolStatus.TIMEOUT,
            error_code="deadline_exceeded",
            error_message="the connection closed before acknowledgement",
            latency_ms=250.0,
            state_effects=(
                WorldStateEffect(
                    path="accounts.acct_1.exists",
                    before=True,
                    after=False,
                ),
            ),
        ),
    )
    gateway = ToolGateway(
        [tool],
        [fixture],
        world={"accounts": {"acct_1": {"exists": True}}},
        now=lambda: fixed_time,
    )

    outcome = gateway.invoke("delete_account", {"account_id": "acct_1"})

    assert outcome.status == ToolOutcomeStatus.TIMEOUT
    assert outcome.error is not None and outcome.error.retryable is True
    assert outcome.error.code == "ambiguous_timeout"
    assert outcome.error.details["simulated_error_code"] == "deadline_exceeded"
    assert outcome.metadata["ambiguous_state"] is True
    assert gateway.world.get("accounts.acct_1.exists") is False
    assert len(gateway.state_transitions) == 1
    assert outcome.state_transition_ids == (
        gateway.state_transitions[0].transition_id,
    )
    assert gateway.state_transitions[0].attempt_id == gateway.attempts[0].attempt_id


@pytest.mark.parametrize(
    ("status", "extra"),
    [
        ("success", {}),
        ("empty", {"result": []}),
        ("error", {"error_code": "unavailable", "error_message": "simulated"}),
        ("timeout", {"error_code": "timeout", "error_message": "simulated"}),
        ("malformed", {"result": "not-json-shaped"}),
        ("partial", {"result": {"complete": False}}),
        ("stale", {"result": {"version": "old"}}),
    ],
)
def test_all_supported_fixture_outcomes_are_immediate(
    status: str, extra: dict[str, Any]
) -> None:
    fixture = {"status": status, **extra}
    gateway = ToolGateway(
        [_tool(state_changing=False)],
        {"update_account": fixture},
    )

    outcome = gateway.invoke("update_account", {"account_id": "acct_1"})

    assert outcome.status.value == status
    assert outcome.metadata["simulated"] is True


def test_budget_tracker_preserves_unknown_usage_and_enforces_limits() -> None:
    current = [100.0]
    tracker = BudgetTracker(
        ResourceBudgets(
            wall_clock_seconds=2.0,
            max_model_turns=1,
            max_tool_calls=1,
            token_budget=10,
            cost_budget_usd=0.25,
        ),
        clock=lambda: current[0],
    )

    assert tracker.snapshot().tokens is None
    assert tracker.snapshot().cost_usd is None
    tracker.consume_model_turn()
    tracker.consume_tool_call()
    tracker.add_tokens(10)
    tracker.add_cost(0.25)

    with pytest.raises(BudgetExceeded, match="model_turns"):
        tracker.consume_model_turn()
    with pytest.raises(BudgetExceeded, match="tool_calls"):
        tracker.consume_tool_call()
    with pytest.raises(BudgetExceeded, match="tokens"):
        tracker.add_tokens(1)
    with pytest.raises(BudgetExceeded, match="cost_usd"):
        tracker.add_cost(0.01)

    current[0] += 2.01
    with pytest.raises(BudgetExceeded, match="wall_time"):
        tracker.check_wall_time()


def test_gateway_enforces_exact_retry_budget_before_reusing_fixture() -> None:
    gateway = ToolGateway(
        [_tool(state_changing=False)],
        {
            "update_account": {
                "repeat": True,
                "status": "timeout",
                "error_code": "deadline_exceeded",
                "error_message": "simulated timeout",
            }
        },
        budgets={"max_tool_calls": 3, "max_retries": 0},
    )
    arguments = {"account_id": "acct_1"}

    assert gateway.invoke("update_account", arguments).status == ToolOutcomeStatus.TIMEOUT
    with pytest.raises(ToolCallBlockedError, match="retry budget") as captured:
        gateway.invoke("update_account", arguments)

    assert captured.value.outcome.status == ToolOutcomeStatus.BLOCKED
    assert captured.value.outcome.error is not None
    assert captured.value.outcome.error.code == "retry_budget_exceeded"


def test_over_limit_call_is_blocked_with_attempt_and_outcome_evidence() -> None:
    gateway = ToolGateway(
        [_tool(state_changing=False)],
        {
            "update_account": {
                "repeat": True,
                "status": "success",
                "result": {"ok": True},
            }
        },
        budgets={"max_tool_calls": 1},
    )
    gateway.invoke("update_account", {"account_id": "acct_1"})

    with pytest.raises(ToolCallBlockedError) as captured:
        gateway.invoke("update_account", {"account_id": "acct_2"})

    assert captured.value.outcome.error is not None
    assert captured.value.outcome.error.code == "tool_calls_budget_exceeded"
    assert len(gateway.attempts) == len(gateway.outcomes) == 2
    assert gateway.outcomes[-1].status == ToolOutcomeStatus.BLOCKED


def test_simulated_latency_counts_toward_wall_time_without_sleeping() -> None:
    gateway = ToolGateway(
        [_tool()],
        {
            "update_account": {
                "status": "success",
                "latency_ms": 2_000,
                "effects": [{"op": "increment", "path": "updates"}],
            }
        },
        world={"updates": 0},
        budgets={"wall_clock_seconds": 1.0, "max_tool_calls": 1},
    )

    with pytest.raises(ToolCallBlockedError) as captured:
        gateway.invoke("update_account", {"account_id": "acct_1"})

    assert captured.value.outcome.error is not None
    assert captured.value.outcome.error.code == "wall_time_budget_exceeded"
    assert gateway.world.get("updates") == 0


def test_explicit_null_precondition_does_not_match_a_missing_path() -> None:
    gateway = ToolGateway(
        [_tool()],
        {
            "update_account": {
                "status": "success",
                "effects": [{"path": "missing", "before": None, "after": "created"}],
            }
        },
        world={},
    )

    with pytest.raises(WorldStateError, match="expected an existing value"):
        gateway.invoke("update_account", {"account_id": "acct_1"})

    assert gateway.world.state == {}


def test_repeated_success_is_observed_as_a_duplicate_not_misclassified_as_retry() -> None:
    gateway = ToolGateway(
        [_tool(state_changing=True)],
        {
            "update_account": {
                "repeat": True,
                "status": "success",
                "result": {"ok": True},
                "effects": [{"op": "increment", "path": "updates"}],
            }
        },
        world={"updates": 0},
        budgets={"max_tool_calls": 2, "max_retries": 0},
    )
    arguments = {"account_id": "acct_1"}

    assert gateway.invoke("update_account", arguments).status == ToolOutcomeStatus.SUCCESS
    assert gateway.invoke("update_account", arguments).status == ToolOutcomeStatus.SUCCESS

    assert gateway.world.get("updates") == 2
    assert gateway.budgets.snapshot().retries == 0
    assert len(gateway.attempts) == 2


def test_gateway_uses_persistent_owned_copy_for_mapping_world_state() -> None:
    caller_state = {"count": 0}
    gateway = ToolGateway(
        [_tool(state_changing=True)],
        {
            "update_account": {
                "repeat": True,
                "status": "success",
                "result": {"ok": True},
                "effects": [{"op": "increment", "path": "count", "delta": 1}],
            }
        },
        world=caller_state,
    )

    gateway.invoke("update_account", {"account_id": "acct_1"}, caller_state)
    gateway.invoke("update_account", {"account_id": "acct_2"}, caller_state)

    assert gateway.world.get("count") == 2
    assert caller_state == {"count": 0}
    assert [item.transition_id for item in gateway.state_transitions] == [
        "transition-0001",
        "transition-0002",
    ]


def test_high_priority_injected_fixture_overrides_matching_baseline() -> None:
    gateway = ToolGateway(
        [_tool(state_changing=False)],
        (
            {
                "fixture_id": "baseline",
                "tool_name": "update_account",
                "arguments_match": {"account_id": "acct_1"},
                "invocation_index": 1,
                "status": "success",
                "result": {"ok": True},
            },
            {
                "fixture_id": "fault",
                "tool_name": "update_account",
                "invocation_index": 1,
                "priority": 100,
                "status": "timeout",
                "error_code": "timeout",
                "error_message": "Injected timeout",
            },
        ),
    )

    outcome = gateway.invoke("update_account", {"account_id": "acct_1"})

    assert outcome.status == ToolOutcomeStatus.TIMEOUT
    assert outcome.metadata["fixture_id"] == "fault"
