"""Developer-declared MCP tool manifests: loader safety and adapter wiring.

The core claim under test: a PydanticAI agent whose tools come from an
external (non-function) toolset -- the shape a real MCP-backed agent has --
is refused today, and a developer-declared, frozen manifest is the only
thing that lifts that refusal. AgentCheck never connects to a real MCP
server anywhere in this file; the fake external toolset below raises
immediately if either of its methods is ever called, so a passing test here
is itself proof that path was never touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import AbstractToolset

from agentcheck.adapters import PydanticAIAdapter
from agentcheck.domain import (
    ConversationRole,
    ConversationTurn,
    ResourceBudgets,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolFixture,
    ToolOutcomeStatus,
)
from agentcheck.errors import ConfigurationError
from agentcheck.mcp_manifest import DeclaredMcpTool, McpManifest, load_mcp_manifest
from agentcheck.runner import ToolGateway
from agentcheck.runner.budgets import BudgetTracker


class _NeverCalledExternalToolset(AbstractToolset[Any]):
    """Stands in for a real MCP toolset without ever touching a real server.

    Every method raises on entry. AgentCheck must never call any of them --
    that is the entire safety property a manifest exists to preserve, and a
    test that let this run for real would prove nothing about it.
    """

    @property
    def id(self) -> str | None:
        return "fake-mcp"

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, Any]:
        raise AssertionError("a real MCP connection must never be attempted")

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: Any
    ) -> Any:
        raise AssertionError("the original external tool must never execute")


def _agent_with_external_toolset(*, extra_function_tool: bool = False) -> Agent:
    tools = []
    if extra_function_tool:

        def existing_tool(x: int) -> int:
            raise AssertionError("real function tool handler must never run")

        tools.append(existing_tool)
    return Agent(
        "test",
        tools=tools,
        toolsets=[_NeverCalledExternalToolset()],
    )


# --- preflight: refused without a manifest, supported with one -------------


def test_external_toolset_without_manifest_is_still_refused() -> None:
    agent = _agent_with_external_toolset()
    report = PydanticAIAdapter().preflight(agent)
    assert any(issue.code == "unsupported_toolset" for issue in report.issues)


def test_external_toolset_with_manifest_is_supported() -> None:
    agent = _agent_with_external_toolset()
    manifest = McpManifest(
        tools={
            "send_message": DeclaredMcpTool(
                description="Send a message.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        }
    )
    report = PydanticAIAdapter().preflight(agent, mcp_manifest=manifest)
    assert not any(issue.code == "unsupported_toolset" for issue in report.issues)


# --- inspect: manifest tools become real, risk-resolved ToolDefinitions ----


def test_manifest_tool_is_added_to_the_spec_and_risk_resolved() -> None:
    agent = _agent_with_external_toolset()
    manifest = McpManifest(
        tools={
            "delete_reminder": DeclaredMcpTool(
                description="Delete a reminder by id.",
                input_schema={
                    "type": "object",
                    "properties": {"reminder_id": {"type": "string"}},
                    "required": ["reminder_id"],
                },
            )
        }
    )
    spec = PydanticAIAdapter().inspect(agent, mcp_manifest=manifest)
    names = {item.value.name for item in spec.tools.items}
    assert "delete_reminder" in names
    definition = next(
        item.value for item in spec.tools.items if item.value.name == "delete_reminder"
    )
    # "delete" matches the existing name-token vocabulary -- proves the
    # manifest path shares the same risk resolver as function tools, not a
    # second, weaker one.
    assert definition.state_changing is True
    assert definition.destructive is True


def test_manifest_name_colliding_with_a_real_tool_is_rejected() -> None:
    agent = _agent_with_external_toolset(extra_function_tool=True)
    manifest = McpManifest(
        tools={"existing_tool": DeclaredMcpTool(description="Collides.", input_schema={})}
    )
    with pytest.raises(ConfigurationError, match="existing_tool"):
        PydanticAIAdapter().inspect(agent, mcp_manifest=manifest)


# --- prepare/run: the manifest tool reaches ToolGateway, never the real one -


@dataclass
class _Deps:
    pass


def test_manifest_declared_tool_is_simulated_through_the_gateway() -> None:
    """End-to-end: the model calls the manifest tool, AgentCheck answers it
    from a fixture, and the real external toolset is never touched."""

    manifest = McpManifest(
        tools={
            "send_message": DeclaredMcpTool(
                description="Send a message.",
                input_schema={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        }
    )

    def scripted(messages: list[Any], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name="send_message", args={"text": "hi"})]
            )
        return ModelResponse(parts=[TextPart("done")])

    agent_with_model = Agent(
        FunctionModel(scripted),
        tools=[],
        toolsets=[_NeverCalledExternalToolset()],
    )

    adapter = PydanticAIAdapter()
    spec = adapter.inspect(agent_with_model, mcp_manifest=manifest)
    tools = tuple(item.value for item in spec.tools.items)
    gateway = ToolGateway(
        tools,
        (
            ToolFixture(
                fixture_id="fixture:send_message",
                tool_name="send_message",
                outcome=SimulatedToolOutcome(
                    status=SimulatedToolStatus.SUCCESS, result={"ok": True}
                ),
            ),
        ),
        budgets=BudgetTracker(ResourceBudgets()),
        run_id="test-run",
    )
    prepared = adapter.prepare(
        agent_with_model,
        gateway,
        world_state=gateway.world,
        controlled_model=False,
        mcp_manifest=manifest,
    )
    import asyncio

    result = asyncio.run(
        adapter.run(
            prepared,
            (
                ConversationTurn(
                    turn_id="turn-1", role=ConversationRole.USER, content="please send hi"
                ),
            ),
            run_id="test-run",
            max_turns=5,
            scenario_id="test-scenario",
            target_id=spec.spec_id,
        )
    )
    tool_events = [
        event
        for event in result.events
        if event.event_type.value in {"tool_attempt", "tool_result"}
    ]
    assert len(tool_events) == 2
    outcome_events = [e for e in tool_events if e.event_type.value == "tool_result"]
    assert outcome_events[0].payload["status"] == ToolOutcomeStatus.SUCCESS.value


# --- loader: same safety discipline as fixtures ----------------------------


def test_load_mcp_manifest_absent_file_returns_none(tmp_path: Path) -> None:
    assert load_mcp_manifest(tmp_path) is None


def test_load_mcp_manifest_valid_file(tmp_path: Path) -> None:
    (tmp_path / "agentcheck-mcp-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.mcp_manifest.v1",
                "tools": {
                    "search": {
                        "description": "Search things.",
                        "input_schema": {"type": "object"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = load_mcp_manifest(tmp_path)
    assert manifest is not None
    assert "search" in manifest.tools


def test_load_mcp_manifest_rejects_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "agentcheck-mcp-manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_mcp_manifest(tmp_path)


def test_load_mcp_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    (tmp_path / "agentcheck-mcp-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.mcp_manifest.v1",
                "tools": {},
                "unexpected_field": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        load_mcp_manifest(tmp_path)


def test_load_mcp_manifest_rejects_remote_schema_reference(tmp_path: Path) -> None:
    (tmp_path / "agentcheck-mcp-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.mcp_manifest.v1",
                "tools": {
                    "search": {
                        "description": "Search.",
                        "input_schema": {"$ref": "https://example.com/schema.json"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="local fragment"):
        load_mcp_manifest(tmp_path)


def test_load_mcp_manifest_rejects_oversized_file(tmp_path: Path) -> None:
    huge_description = "x" * (256 * 1024 + 1)
    (tmp_path / "agentcheck-mcp-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.mcp_manifest.v1",
                "tools": {"a": {"description": huge_description, "input_schema": {}}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="byte manifest limit"):
        load_mcp_manifest(tmp_path)


def test_load_mcp_manifest_refuses_a_symlink_escaping_the_target(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-mcp-manifest.json"
    outside.write_text(
        json.dumps({"schema_version": "agentcheck.mcp_manifest.v1", "tools": {}}),
        encoding="utf-8",
    )
    link = tmp_path / "agentcheck-mcp-manifest.json"
    link.symlink_to(outside)
    with pytest.raises(ConfigurationError):
        load_mcp_manifest(tmp_path)
    outside.unlink()
