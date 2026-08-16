# Deliberately flawed account-support agent

This OpenAI Agents SDK target is the deterministic fixture for AgentCheck.
Its model is local and scripted, so inspection and tests make no provider calls and
need no API key. The four original function handlers are raising tripwires; every
evaluation invocation must be handled by AgentCheck's controlled `ToolGateway`.

The agent intentionally contains five observable defects:

- deletes without explicit confirmation;
- selects the wrong account ID from an ambiguous request;
- retries a destructive call after an ambiguous timeout;
- claims an email update succeeded after the tool reports an error;
- invokes the same email-update side effect twice.

Run it from the repository root:

```bash
agentcheck inspect examples/evaluation/account_agent
agentcheck generate examples/evaluation/account_agent
agentcheck test examples/evaluation/account_agent
agentcheck report examples/evaluation/account_agent --latest
agentcheck replay examples/evaluation/account_agent \
  --manifest .agentcheck/replay/<run-id>.json
agentcheck shrink examples/evaluation/account_agent \
  --manifest .agentcheck/replay/<run-id>.json
```

`generate` is optional. Without a frozen suite file, `test` still runs the
Phase 1 built-in cases. The frozen file is a reviewable suite, not a replay
manifest.
