# Security policy

## Reporting a vulnerability

Do **not** file a security vulnerability as a public Issue. Use **Report a
vulnerability** in this repository's Security tab to send a private report
through GitHub Private Vulnerability Reporting.

Include the affected version or commit, impact, and minimal reproduction steps.
Do not include production credentials, customer data, or unrelated provider
payloads. If the private reporting button is unavailable, open a public Issue
containing only a request for a private reporting channel—no vulnerability
details or payloads.

AgentCheck is pre-1.0. Security fixes target the latest release and the default
branch; there is no long-term-support or backport policy yet.

## Security boundary

AgentCheck evaluates **trusted local Python code**. Understand the boundary before
running a target:

- Inspection imports the target, so its module-level code executes.
- Declared tool calls routed through `ToolGateway` are simulated; the original
  declared handler does not execute.
- Unknown tools, missing fixtures, schema-invalid calls, and containment failures
  fail closed.
- Scenarios run in child processes with a constrained environment and network
  denied by default.
- The declared-tool guarantee is not a general Python sandbox.
  Direct filesystem writes, subprocess execution, and direct local-database
  access from target imports or arbitrary orchestration are outside the
  guarantee; network denial does not prevent those local effects.
- Artifact and console values pass through bounded credential redaction, but
  novel secret formats may not be recognized.

Never store secrets or production data in `agentcheck.json`, fixture packs,
frozen suites, reports, replay manifests, or other `.agentcheck/` artifacts.
Artifacts can contain prompts, model output, tool inputs, and absolute paths;
review them before sharing.
