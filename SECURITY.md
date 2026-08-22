# Security policy

## Reporting a vulnerability

Please report security issues **privately**, using GitHub's private
vulnerability reporting flow from this repository's **Security** tab
("Report a vulnerability").

Do not open a public issue for a vulnerability, and do not include credentials,
API keys, prompts, model output, or customer data in a report.

A useful report includes the affected version or commit, the impact, and
reproduction steps. Reports are assessed as promptly as maintainers are
available; this is a small pre-1.0 project, not a staffed security team, and
that is worth knowing before you rely on a response time.

GitHub only offers private vulnerability reporting on public repositories, so
while this repository is private that Security-tab flow is unavailable and
enabling it is part of the publication checklist.

If for any reason the flow is not available when you need it, open a public issue
containing **only** a request for a private reporting channel — no details, no
reproduction, no payloads.

There is deliberately no security email address here. Publishing one that is not
monitored would be worse than publishing none.

## Supported versions

AgentCheck is pre-1.0. Security fixes target the latest commit on the default
branch. There is no long-term support branch and no backport policy yet.

## Security model

AgentCheck evaluates **trusted local agents** — code you already run yourself.
Understand the boundary before relying on it:

- **Inspection imports your agent.** Module-level code in the target executes.
  AgentCheck is not a sandbox for untrusted or hostile agent source, and does
  not claim to be one.
- **Simulated evaluation does not execute your tool handlers.** Accepted tools
  are reconstructed with an AgentCheck-owned invoker; the original callable
  stays on the source agent.
- **Scenarios run in isolated child processes** with a constrained environment.
  The environment allowlist is empty by default, so provider credentials are not
  passed to a worker unless you explicitly add them.
- **Network is denied by default.** Reaching a non-allowlisted destination is
  reported as a containment failure rather than silently permitted.
- **The declared-tool guarantee is not a general Python sandbox.** Target
  imports execute. Direct filesystem writes, subprocess execution, and direct
  local-database access performed by imports or custom orchestration are outside
  the guarantee; network denial does not prevent those local effects.
- **Artifacts are redacted** with bounded credential redaction before being
  written or printed. Redaction is best-effort pattern matching: it is not a
  guarantee that a novel secret format will be caught. Treat run artifacts as
  potentially sensitive and review them before sharing.

Artifacts under `.agentcheck/` can contain prompts, model output, tool inputs,
and absolute host paths. They are local files. Do not commit them to a public
repository without reading them first.

This document describes intended controls. It is not a claim of formal security
certification or of an audit.
