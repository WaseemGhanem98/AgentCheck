"""Developer-declared schemas for an external (non-function) toolset.

PydanticAI agents that get their tools from an MCP server (``MCPToolset`` and
its predecessors ``MCPServerStdio``/``MCPServerSSE``/etc.) do not carry a
static tool list the way a plain ``@agent.tool``-decorated function toolset
does: the schemas only exist once something actually connects to that real
server and asks it. AgentCheck never does that -- connecting would mean
either spawning the target's declared subprocess or reaching a real network
endpoint during ``inspect``, both of which are the exact things this
project's network-deny-by-default and no-target-code-execution invariants
exist to prevent (see ``../../projects/agentcheck/architecture/
safety-invariants.md`` invariants 1, 2, and 8 in the Company Brain).

This is the developer's answer, in the same spirit as
``agentcheck-fixtures.json``: a frozen, committed snapshot of what the real
MCP server was observed to expose, authored by the developer running their
own introspection against their own server, with their own credentials,
entirely outside AgentCheck. AgentCheck reads only this static file and never
connects to the real server itself.

This is explicitly a snapshot, not a live mirror. It can drift out of sync
with the real server exactly the way a stale fixture can, and AgentCheck has
no way to detect that -- the same limitation this project already accepts for
fixtures, stated here for the same reason.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agentcheck.domain import ContractModel, JsonObject


MCP_MANIFEST_CONTRACT_VERSION: Literal["agentcheck.mcp_manifest.v1"] = (
    "agentcheck.mcp_manifest.v1"
)

DEFAULT_MCP_MANIFEST_FILENAME = "agentcheck-mcp-manifest.json"


class DeclaredMcpTool(ContractModel):
    """One tool the developer observed the real MCP server exposing.

    Shape mirrors ``ToolDefinition`` deliberately narrow: only what a
    developer can read off the server's own ``tools/list`` response. Risk
    (``state_changing``/``destructive``) is not declared here -- it goes
    through the same ``tool_risk`` block in ``agentcheck.json`` every other
    tool uses, so one declaration mechanism covers both tool sources rather
    than a second, competing one.
    """

    description: str | None = Field(default=None, max_length=8_000)
    input_schema: JsonObject = Field(default_factory=dict)


class McpManifest(ContractModel):
    """Frozen, developer-declared schemas for one target's external toolsets.

    Unknown fields are rejected, matching every other committed AgentCheck
    contract file.
    """

    schema_version: Literal["agentcheck.mcp_manifest.v1"] = (
        MCP_MANIFEST_CONTRACT_VERSION
    )
    tools: dict[str, DeclaredMcpTool] = Field(default_factory=dict)


__all__ = [
    "DEFAULT_MCP_MANIFEST_FILENAME",
    "MCP_MANIFEST_CONTRACT_VERSION",
    "DeclaredMcpTool",
    "McpManifest",
]
