# Deliberately flawed multi-agent handoff router

This OpenAI Agents SDK target is a deterministic handoff fixture for
AgentCheck. Three module-level agents (`triage_agent`, `billing_agent`,
`docs_agent`) are connected only by static `handoff()` factory edges with no
callbacks, payload schemas, input filters, or dynamic enablement — exactly the
surface AgentCheck can prove safe and reconstruct.

Every model is local and scripted, so inspection and tests make no provider
calls and need no API key. The two original function handlers are raising
tripwires; every evaluation invocation must be handled by AgentCheck's
controlled `ToolGateway`, on every reachable agent.

The routing intentionally contains three observable defects:

- a billing request that is misrouted to the docs agent;
- a triage/billing handoff ping-pong loop that only stops at the turn budget;
- a success claim after the invoice tool reports an error.

Run it from the repository root:

```bash
agentcheck inspect examples/evaluation/handoff_router
agentcheck generate examples/evaluation/handoff_router
agentcheck test examples/evaluation/handoff_router
```

The CLI commands below verify inspection, safe reconstruction, and generated
schema-boundary cases. They are not a CLI demonstration of the three planted
handoff defects and may report 100% with zero action paths exercised. The
handoff-specific behavioral scenarios live in
`tests/agentcheck/test_example_handoff_agent.py`.

`inspect` prints the reachable handoff topology. Without a frozen suite,
`test` fails closed because the built-in `account_support_v1` suite does not
match this target's tools; `generate` freezes schema-boundary cases for the
merged tool surface.
