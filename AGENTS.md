# AGENTS.md — working on AgentCheck

Instructions for AI coding agents and new contributors working in this
repository. Read this before changing anything under `agentcheck/`.

## What this project is

AgentCheck evaluates AI agents. It imports a trusted local agent, derives
adversarial scenarios from what it finds, and runs them in isolated child
processes where every tool call is simulated. It emits `PASS` / `FAIL` /
`INCONCLUSIVE` / `INFRA_ERROR`, an HTML report, and a replay manifest.

The product's value is that its verdicts can be trusted. Every rule below
exists to protect that.

## Non-negotiable invariants

Breaking any of these is a correctness bug, not a style issue:

1. **The original tool handler never executes during a simulated evaluation.**
   Interception replaces the invoker before the handler is reached.
2. **Unknown tools fail closed.** Never synthesize a tool result.
3. **No real mutations.** Only the simulated world changes.
4. **Worker isolation.** Scenarios run in child processes; the environment
   allowlist is empty by default.
5. **Network denied by default**, and containment failures surface.
6. **`INCONCLUSIVE` and `INFRA_ERROR` never collapse into `PASS`.**
7. **Redaction at the artifact and log boundary**, before writing or printing.
8. **Never overstate replay.** It reproduces inputs and harness behavior, not
   model determinism.

## Layout

| Path | Role |
|---|---|
| `agentcheck/domain/` | Contract models: scenarios, specs, runs, verdicts, findings. Fingerprinted and versioned. |
| `agentcheck/adapters/` | The only layer allowed to import a framework SDK. |
| `agentcheck/inspect/` | Import a target and extract an `AgentSpec` without running a turn. |
| `agentcheck/generate/` | Derive, lint, select, and freeze suites. |
| `agentcheck/coverage/` | Derived behavioral coverage over a spec and a scenario set. Pure contract analysis; imports no target. |
| `agentcheck/runner/` | Orchestrator, worker, tool gateway, simulated world, budgets, network guard. |
| `agentcheck/evaluate/` | Oracle evaluation and verdict assignment. |
| `agentcheck/replay/` | Manifests, source binding, filesets. |
| `agentcheck/report/`, `agentcheck/baseline/`, `agentcheck/review/` | Reporting, CI gating, human decisions on findings. |
| `agentcheck/regression/` | Run-to-run behavioral comparison over stored artifacts. Executes nothing. |
| `agentcheck/gate.py` | The CI decision: runs the suite, compares the baseline, and answers whether a change blocks the build. Orchestrates the above; decides no verdicts of its own. |
| `agentcheck/identity.py` | Portable target identity and its bounded legacy compatibility path. |
| `agentcheck/redaction.py`, `agentcheck/privacy.py` | Credential redaction for artifacts and logs. |
| `agentcheck/cli.py` | The `agentcheck` command. |

Dependency direction: `adapters` may import `domain`; `domain` must not import
`adapters`. Nothing in `agentcheck/` may import a private or host-repo module —
`tests/agentcheck/test_package_boundary.py` enforces this statically.

## Serialized contracts — change deliberately

These are hashed or stored, so changing them re-identifies existing artifacts:

- `Scenario.expected_fingerprint()` hashes the scenario document minus display
  identity.
- `FrozenSuite.expected_fingerprint()` hashes the whole suite document, which
  includes `provenance.generator_version`. That is
  **`GENERATOR_COMPATIBILITY_VERSION`, not `agentcheck.__version__`** — the two
  are deliberately different things and must not be conflated again.
  - `agentcheck.__version__` is the distribution release. It appears in
    `--version`, and in run, baseline, replay and review records as "which build
    executed this". Releasing must not re-identify a suite that asserts exactly
    the same thing, so it is not part of suite provenance.
  - `GENERATOR_COMPATIBILITY_VERSION` (in `agentcheck/generate/suite.py`) is the
    generation semantics. Raise it when a change *should* re-identify otherwise
    equivalent suites: a new case family, a changed default budget, different
    fixture wiring, altered lint or selection behaviour. Adding an unused field,
    renaming the package, or cutting a release does not qualify.
  - Raising it never invalidates an artifact already on disk: a stored suite
    validates against the `generator_version` it recorded, not against today's.
- Replay manifests, baselines, and review records embed `agentcheck.__version__`
  as execution provenance. That is a record of the run, not of what a suite
  asserts, and it is not suite identity.

If a fingerprint changes and you did not intend a contract change, stop and find
out why rather than re-freezing the expected values.

## Testing

```bash
python -m pytest tests/agentcheck/test_<area>.py -q     # focused, preferred
python -m pytest tests -q -n 2                          # full, slow
python -m ruff check agentcheck tests
python -m mypy agentcheck
```

Tests must be offline, credential-free, and free. Never add a test that calls a
real provider. The bundled example target uses a scripted local model precisely
so the suite costs nothing.

## Adapters

An adapter reads framework-private attributes to build an `AgentSpec`, so it is
pinned to a single verified minor version. On an unverified version the failure
mode is a *wrong spec*, not a crash — which silently tests the wrong thing. Do
not widen a version gate without evidence the new version was verified.

Each adapter lives behind its own optional extra and must raise an actionable
`AdapterDependencyError` (see `missing_extra_message`) when that extra is
absent, rather than surfacing an import traceback.

## Style

Match the surrounding code: explicit contracts, comments that explain *why* a
constraint exists rather than restating the code, and no defensive fallback that
turns a failure into a plausible-looking success. In this codebase, a silent
fallback is usually a bug in disguise.
