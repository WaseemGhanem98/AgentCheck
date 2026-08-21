# Custom agent with confirmation-gated deletion

This deterministic example implements AgentCheck's custom-agent contract
directly. It uses no model provider, needs no API key, and makes no network
request. Its one declared action, `delete_account`, is destructive, so the
agent asks on its opening turn and calls the tool only after the scenario sends
a later confirmation turn.

The example contains no real `delete_account` handler. `agent.py` supplies only
a `ToolDefinition`; the call in `resume()` goes through AgentCheck's
`ToolRuntime` and is answered by `ToolGateway` from the scenario fixture. The
configured `derived_tool_risk_v1` policy pack turns the declared destructive
risk into confirmation, no-duplicate, ambiguous-timeout, and truthful-output
rules for this tool.

Run it from the repository root:

```bash
agentcheck inspect  examples/evaluation/custom_agent
agentcheck generate examples/evaluation/custom_agent
agentcheck test     examples/evaluation/custom_agent
agentcheck report   examples/evaluation/custom_agent --latest
```

The generated confirmation case supplies its reply only after the first agent
turn. The recorded trajectory can therefore show whether `delete_account`
happened before or after consent; no confirmation is inferred from prose.

This example demonstrates the declared-tool boundary, not a sandbox for
arbitrary Python. Code in `start()` and `resume()` really executes. A direct
filesystem, subprocess, or local-database side effect written there is outside
the declared-tool guarantee. Process isolation, an empty environment allowlist,
and deny-by-default egress still apply, but they do not make arbitrary local
Python side effects safe.

See [Custom Python agents](../../../docs/custom-agents.md) for the full contract,
configuration, safety boundary, and model-observability limitations.
