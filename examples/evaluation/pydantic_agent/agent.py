"""Credential-free PydanticAI target for the AgentCheck worked example."""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


ORIGINAL_HANDLER_CALLS: list[str] = []
_LOOKUP_REQUEST = "Look up order order_123 and tell me its current status."


def lookup_order(order_id: str) -> str:
    """Look up one order's current status."""
    ORIGINAL_HANDLER_CALLS.append("lookup_order")
    raise AssertionError("original lookup_order handler must never run")


def _script(messages: list[Any], info: AgentInfo) -> ModelResponse:
    del info
    parts = [
        part
        for message in messages
        for part in getattr(message, "parts", ())
    ]
    if any(isinstance(part, ToolCallPart) for part in parts):
        return ModelResponse(
            parts=[TextPart("Order order_123 is ready for pickup.")]
        )
    text = "\n".join(
        content
        for part in parts
        if isinstance((content := getattr(part, "content", None)), str)
    )
    if _LOOKUP_REQUEST in text:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "lookup_order",
                    {"order_id": "order_123"},
                )
            ]
        )
    return ModelResponse(
        parts=[TextPart("Please provide a valid order ID to look up.")]
    )


agent = Agent(
    FunctionModel(_script),
    instructions=(
        "Help customers look up orders. Use lookup_order only when the customer "
        "supplies a concrete order ID."
    ),
    tools=[lookup_order],
    name="Offline Order Support",
)
