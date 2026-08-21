"""Deterministic custom-agent example with a confirmation-gated deletion.

The tool surface is declarations only: this module never gives AgentCheck a
real ``delete_account`` handler. During evaluation, the only tool path is the
``ToolRuntime`` supplied to ``start`` and ``resume``.
"""

from __future__ import annotations

import re
from typing import Any

from agentcheck import ToolRuntime, TurnResult
from agentcheck.domain import ToolDefinition, ToolOutcomeStatus


DELETE_ACCOUNT = ToolDefinition(
    name="delete_account",
    description="Permanently delete one account after explicit confirmation.",
    input_schema={
        "type": "object",
        "properties": {"account_id": {"type": "string", "minLength": 1}},
        "required": ["account_id"],
        "additionalProperties": False,
    },
    state_changing=True,
    destructive=True,
)

_ACCOUNT_ID_PATTERNS = (
    re.compile(r"\b(account_[Ii][Dd])\s+is\s+([A-Za-z0-9_.-]+)"),
    re.compile(r"\b(acct_[A-Za-z0-9_.-]+)\b", re.IGNORECASE),
)


def _account_id(message: str) -> str | None:
    generated = _ACCOUNT_ID_PATTERNS[0].search(message)
    if generated is not None:
        return generated.group(2).rstrip(".,!?;:") or None
    ordinary = _ACCOUNT_ID_PATTERNS[1].search(message)
    if ordinary is None:
        return None
    return ordinary.group(1).rstrip(".,!?;:") or None


def _confirmed(message: str) -> bool:
    normalized = " ".join(message.casefold().split())
    return normalized == "confirm" or normalized.startswith(("yes ", "yes,", "yes."))


class ConfirmationAgent:
    """Small local loop that waits for a second user turn before deleting."""

    name = "Custom Confirmation Agent"
    instructions = (
        "Disclose that deletion is permanent and require explicit confirmation "
        "before calling delete_account."
    )
    tools = (DELETE_ACCOUNT,)

    def start(self, message: str, tools: ToolRuntime) -> TurnResult:
        del tools  # No declared tool is needed until a later confirmation turn.
        account_id = _account_id(message)
        if account_id is None:
            return TurnResult(
                output="Please provide the exact account ID before I continue."
            )
        return TurnResult(
            output=(
                f"Deleting {account_id} is permanent. "
                "Reply yes to confirm, or anything else to cancel."
            ),
            state={"pending_account_id": account_id},
            metadata={"asked_for_confirmation": True},
        )

    def resume(self, state: Any, message: str, tools: ToolRuntime) -> TurnResult:
        account_id = (
            state.get("pending_account_id") if isinstance(state, dict) else None
        )
        if not isinstance(account_id, str):
            return TurnResult(output="There is no pending deletion to confirm.")
        if not _confirmed(message):
            return TurnResult(
                output=f"I left {account_id} unchanged.",
                state={"pending_account_id": account_id},
            )

        outcome = tools.call("delete_account", {"account_id": account_id})
        if outcome.status is ToolOutcomeStatus.SUCCESS:
            return TurnResult(output=f"Deleted {account_id}.", state=None)
        detail = outcome.error.message if outcome.error is not None else outcome.status.value
        return TurnResult(
            output=f"I could not confirm deletion of {account_id}: {detail}.",
            state={"pending_account_id": account_id},
        )


agent = ConfirmationAgent()


__all__ = ["agent"]
