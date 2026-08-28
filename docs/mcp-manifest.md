# MCP tool manifests

An agent whose tools come from an external toolset — the shape a real
MCP-backed agent has — is refused at `preflight` with `unsupported_toolset`.
AgentCheck only replaces tools it can fully own: the agent's own function
toolset, whose schemas are static and known at inspect time. An MCP toolset's
tools are discovered by connecting to a live server, and AgentCheck will not
do that during `inspect`, `generate`, or `run` — that would mean either
spawning a subprocess (stdio MCP) or making a network call (remote MCP)
against target-controlled code, which is exactly what the no-target-execution
and network-deny-by-default invariants exist to rule out.

A developer-declared manifest is the only thing that lifts the refusal.

## What it is

`agentcheck-mcp-manifest.json` in the target root, alongside
`agentcheck.json`:

```json
{
  "schema_version": "agentcheck.mcp_manifest.v1",
  "tools": {
    "send_message": {
      "description": "Send a message to a Slack channel.",
      "input_schema": {
        "type": "object",
        "properties": { "channel": { "type": "string" }, "text": { "type": "string" } },
        "required": ["channel", "text"]
      }
    }
  }
}
```

You write this by hand, once, out of band — by connecting to the real MCP
server yourself, with your own credentials, and copying down the tool names,
descriptions, and JSON Schemas it reports. AgentCheck never reads it back
from a live server; it only ever reads this frozen file.

Each declared tool is folded into the target's spec exactly like a function
tool: it gets a `ToolDefinition`, participates in risk resolution (name +
description inference, or an explicit override in `agentcheck.json`'s
`tool_risk` block — see [fault-testing.md](fault-testing.md)), and is
simulated through `ToolGateway` from fixtures like every other tool. The
agent's real external toolset is never queried and never called; only the
name has to match for AgentCheck to intercept the call.

A name in the manifest that collides with a real function tool on the agent
is a configuration error (`ConfigurationError`), not a silent override — the
manifest may only name tools that come from an external toolset.

## What it is not

Not a live mirror. The manifest is a snapshot, with the same staleness risk
as a fixtures file: if the real MCP server's schema changes, AgentCheck keeps
testing against what you declared until you update the manifest by hand.

Not validated against the real server by AgentCheck. Getting the schema
wrong just means the simulated tool doesn't match reality — the same failure
mode as a hand-written fixture that doesn't match a real tool's behavior.

Not a way to test the MCP server itself, or the transport, or authentication
to it. AgentCheck tests how the agent behaves given tool calls and results
shaped the way you declared — the same behavioral surface it always tests,
now extended to toolsets it previously couldn't see into at all.

## Safety

The manifest file goes through the same loader discipline as
`agentcheck-fixtures.json`: contained-path resolution with `O_NOFOLLOW` (a
symlink escaping the target directory is refused), a byte-size cap, strict
schema validation (`extra="forbid"`), and `input_schema` for every declared
tool is run through the same `offline_validator` that rejects non-local
`$ref`/`$dynamicRef` — a manifest cannot smuggle in a remote schema fetch any
more than a fixture can.
