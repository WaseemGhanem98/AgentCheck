# Development history

AgentCheck's public repository begins with an import commit, which makes it look
newer than it is. This document records where the code actually came from and
what was built when, so the provenance is not lost.

## Origin

AgentCheck was built as an evaluation subsystem inside a private project called
AgentLens, an observability product for AI agent runs. The two solve different
problems — AgentCheck decides whether an agent behaved safely, AgentLens shows
what an agent did — and once AgentCheck had a coherent product surface it was
extracted into this repository as an independent open-source product.

The extraction was a boundary change, not a rewrite. The implementation that
landed here is byte-identical to the one that left: at the point of extraction
both repositories held the same `agentcheck/` tree, verified by identical git
tree hashes across all 83 modules.

AgentLens remains private and is not a dependency. Nothing in this repository
imports it, and a test enforces that.

## Timeline

Development ran from **2026-08-16 to 2026-08-20** across 57 commits. It was
short and dense rather than long and sparse; that is the honest shape of it, and
this document does not try to make it look otherwise.

### 2026-08-16 — the evaluation core

The first day established most of the pipeline that still exists:

- deterministic evaluation MVP and the `agentcheck init` command;
- systematic capability extraction from a target's declared tool schemas;
- schema-boundary scenario generation, then workflow mutation generation;
- frozen suites — fingerprinted, deterministic documents so the same target,
  config, and seed always freeze the same suite;
- policy packs, SQLite persistence, stored-run reporting;
- coverage-based scenario selection;
- secure replay manifests, then hardened replay source bindings;
- deterministic counterexample shrinking, later repeated to a 1-minimal fixpoint;
- human finding review, and regression gating that fails closed on an unsigned
  baseline.

### 2026-08-17 — real targets, and the isolation that makes them safe

Work shifted from a bundled example to agents somebody else wrote:

- explicit target environments and factory entrypoints;
- zero-argument tool generation, and target working-directory isolation;
- a statically proven `on_handoff` context-assignment subset — the original
  callbacks are never executed;
- source-inventory exclusions, then a correction restoring fail-closed source
  binding (the permissive version was the bug);
- statically typed structured output via `agent.output_type`;
- network egress containment inside worker processes.

### 2026-08-18 — offline evaluation of real agents

- AF_UNIX connect denial and network-denial diagnostics, closing the gap where
  a target could reach a local model server;
- **the controlled model**: real external agents reach behavioural verdicts
  offline, with no provider call and no credential;
- behavioural policy scope derived from each target's own inferred tool risk;
- semantic tool-invocation identity, so two tools with the same name are not
  confused;
- a scoped provider endpoint, for when a real model is deliberately driving the
  evaluation;
- contract-derived action-path scenarios, representative input values, and
  honest reporting of whether an action path was actually exercised.

### 2026-08-19 — making the oracles observable

- action-path coverage honesty carried into the report;
- failure and ambiguity oracles given something to observe on third-party agents;
- **declared prerequisite fixtures**: a generated action case can answer the
  tools that gate it, so the focal action becomes reachable;
- generated confirmation cases that a correct call can actually pass.

### 2026-08-20 — interaction, a second framework, and extraction

- **interactive scenarios** (`followup_turns`): the scripted user can answer the
  agent mid-run rather than only speaking first;
- **PydanticAI** as the second framework adapter, alongside the OpenAI Agents
  SDK;
- the redaction layer given its own home so the package could ship without the
  private project;
- extraction into this repository, and independent CI.

## What was investigated and did not work out

A benchmark was attempted in which AgentCheck would detect a real historical bug
at its buggy revision and pass at its fixed revision. Eight candidates from
three repositories were examined across two searches. **None qualified**, for
structural reasons rather than convenience:

- every framework-SDK candidate's buggy revision predates the supported adapter
  version gate;
- the closest candidate's failure mode classifies as `INFRA_ERROR`, not `FAIL` —
  a harness fault, which the verdict model deliberately refuses to report as a
  behavioural failure;
- behavioural defects are rarely filed as bugs at all, because stochastic
  behaviour resists the regression test that would pin a fix to a revision.

The search is recorded as a **negative result**. Eight candidates is far too
small to support a detection-rate figure, and none is quoted anywhere. See
[validation evidence](validation-evidence.md).

## Note on pull requests

Development happened as pull requests in the private repository. Those PR
objects cannot be moved here, and manufacturing substitutes would be
dishonest — so this repository has none from that period, and the commit history
is the record instead. Historical PR numbers are deliberately not linked: the
repository they belong to is private, so such links would be dead for every
reader.
