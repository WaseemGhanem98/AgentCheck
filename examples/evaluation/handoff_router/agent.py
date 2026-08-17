"""Deliberately flawed, fully local multi-agent OpenAI Agents SDK example.

Three module-level agents route work through static ``handoff()`` factory
edges.  Every model is scripted, so inspection and tests make no provider
calls and need no API key.  The function-tool handlers are tripwires:
AgentCheck must replace every one of them on every reachable agent before a
run.  Reaching an original handler is an immediate safety failure.

The routing intentionally contains three observable defects:

- a billing request that is misrouted to the docs agent;
- a triage/billing handoff ping-pong loop that only stops at the turn budget;
- a success claim after the invoice tool reports an error.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, NoReturn

from agents import Agent, Model, function_tool, handoff
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.tool import Tool
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)


# Tests inspect this list after running scenarios.  A non-empty value proves
# that a live/original handler escaped AgentCheck's replacement boundary.
ORIGINAL_TOOL_CALLS: list[tuple[str, dict[str, Any]]] = []


def _original_tool_tripwire(tool_name: str, arguments: dict[str, Any]) -> NoReturn:
    ORIGINAL_TOOL_CALLS.append((tool_name, arguments))
    print(f"[debug] original {tool_name} handler executed")
    raise RuntimeError(
        f"UNSAFE: original {tool_name} handler ran during an AgentCheck test"
    )


@function_tool
def lookup_invoice(invoice_id: str) -> dict[str, Any]:
    """Look up one invoice by its exact identifier."""

    _original_tool_tripwire("lookup_invoice", {"invoice_id": invoice_id})


@function_tool
def search_docs(query: str) -> dict[str, Any]:
    """Search the public product documentation."""

    _original_tool_tripwire("search_docs", {"query": query})


def _message(text: str, item_id: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=item_id,
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _call(name: str, arguments: dict[str, Any], index: int) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        arguments=json.dumps(arguments, sort_keys=True),
        call_id=f"call-{name}-{index}",
        name=name,
        type="function_call",
        status="completed",
    )


def _prompt_text(input: str | Sequence[Any]) -> str:
    if isinstance(input, str):
        return input.casefold()
    for item in input:
        if isinstance(item, dict) and item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                return content.casefold()
    return ""


def _tool_call_count(input: str | Sequence[Any], name: str) -> int:
    if isinstance(input, str):
        return 0
    count = 0
    for item in input:
        if isinstance(item, dict) and item.get("name") == name:
            count += 1
        elif isinstance(item, ResponseFunctionToolCall) and item.name == name:
            count += 1
    return count


class _ScriptedModelBase(Model):
    """Shared deterministic plumbing; behavior lives in ``outputs_for``."""

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
    ) -> ModelResponse:
        del (
            system_instructions,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        return ModelResponse(
            output=self.outputs_for(input),
            usage=Usage(),
            response_id=None,
            request_id=None,
            raw_usage=None,
        )

    def outputs_for(self, input: str | list[TResponseInputItem]) -> list[Any]:
        raise NotImplementedError

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[TResponseStreamEvent]:
        del args, kwargs
        raise NotImplementedError


class ScriptedTriageModel(_ScriptedModelBase):
    def outputs_for(self, input: str | list[TResponseInputItem]) -> list[Any]:
        prompt = _prompt_text(input)
        transfers = _tool_call_count(input, "transfer_to_billing_agent")
        if "ping pong" in prompt:
            # Defect: triage keeps bouncing the case back to billing forever.
            return [_call("transfer_to_billing_agent", {}, transfers + 1)]
        if "misroute" in prompt:
            # Defect: a billing question is routed to the docs agent.
            return [_call("transfer_to_docs_agent", {}, 1)]
        if "invoice" in prompt:
            return [_call("transfer_to_billing_agent", {}, 1)]
        return [_message("How can I help you today?", "triage-help")]


class ScriptedBillingModel(_ScriptedModelBase):
    def outputs_for(self, input: str | list[TResponseInputItem]) -> list[Any]:
        prompt = _prompt_text(input)
        if "ping pong" in prompt:
            transfers = _tool_call_count(input, "transfer_to_triage_agent")
            return [_call("transfer_to_triage_agent", {}, transfers + 1)]
        lookups = _tool_call_count(input, "lookup_invoice")
        # Schema-boundary missing-required cases withhold invoice_id; inventing
        # one would be an agent FAIL, not a scripted-model fixture.
        if lookups == 0 and "inv_" in prompt:
            return [_call("lookup_invoice", {"invoice_id": "inv_42"}, 1)]
        if "fabricate" in prompt:
            # Defect: claims success even when the tool reported an error.
            return [
                _message(
                    "Your invoice lookup completed successfully.",
                    "billing-fabricated",
                )
            ]
        return [_message("Invoice inv_42 total is 42 dollars.", "billing-answer")]


class ScriptedDocsModel(_ScriptedModelBase):
    def outputs_for(self, input: str | list[TResponseInputItem]) -> list[Any]:
        del input
        return [
            _message(
                "Here is a documentation article about invoices.",
                "docs-answer",
            )
        ]


billing_agent = Agent(
    name="Billing Agent",
    instructions="Resolve billing questions with the invoice tools.",
    tools=[lookup_invoice],
    model=ScriptedBillingModel(),
)

docs_agent = Agent(
    name="Docs Agent",
    instructions="Answer documentation questions.",
    tools=[search_docs],
    model=ScriptedDocsModel(),
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="Route each request to the correct specialist agent.",
    handoffs=[
        handoff(billing_agent, tool_name_override="transfer_to_billing_agent"),
        handoff(docs_agent, tool_name_override="transfer_to_docs_agent"),
    ],
    model=ScriptedTriageModel(),
)

billing_agent.handoffs.append(
    handoff(triage_agent, tool_name_override="transfer_to_triage_agent")
)
docs_agent.handoffs.append(
    handoff(triage_agent, tool_name_override="transfer_to_triage_agent")
)
