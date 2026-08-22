# AgentCheck

**An evaluation framework for testing whether AI agents behave safely — without
letting their declared tools execute real side effects.**

AgentCheck imports an agent you already have, generates adversarial scenarios
from the contracts its own code declares, and runs them in isolated processes
where declared tool requests are answered from a simulated world. Native SDK
adapters replace each handler before it can be reached; custom agents provide
inert declarations and call an AgentCheck-supplied runtime. AgentCheck then
judges the resulting trajectory and returns `PASS` / `FAIL` / `INCONCLUSIVE` /
`INFRA_ERROR`.

> **Status: 0.1.0, pre-1.0.** Two native framework adapters, each pinned to one
> verified minor version, plus a declaration-based custom Python integration.
> Everything documented here is implemented and tested; see
> [Known limitations](#known-limitations) for what it does not do.

---

## Why AgentCheck

An agent's final answer tells you almost nothing about whether it was safe
getting there.

The failures that matter are behavioural. The agent deletes an account without
asking. It retries a destructive call after an ambiguous timeout. It resolves an
ambiguous request to the wrong record. It reports success after the tool
returned an error. Every one of those can occur underneath a perfectly
reasonable-sounding reply.

Testing that by hand means writing scenarios yourself for every risky path, or
letting the agent touch real systems to find out. The first does not scale. The
second is how you find out in production.

## What it does

1. **Inspect** — imports a trusted local agent and reads its tools, their
   schemas, its instructions and guardrails, without running a turn.
2. **Generate** — derives scenarios from those contracts and freezes them into a
   fingerprinted suite. Same target, config, and seed ⇒ byte-identical suite.
3. **Run** — executes each scenario in an isolated child process with network
   denied and credentials absent by default.
4. **Simulate** — routes declared tool calls into a seeded world, with fixtures,
   prerequisites, and injected faults. SDK handlers are replaced before they
   can run; custom agents call the supplied `ToolRuntime`.
5. **Evaluate** — judges the trajectory against per-scenario oracles.
6. **Report** — writes local JSON/JSONL artifacts, an HTML report, and a replay
   manifest.
7. **Gate CI** — compares against a committed baseline so builds fail on *new*
   regressions rather than the whole backlog.

## Quickstart

AgentCheck is **not published to PyPI yet**. Install from a clone:

```bash
git clone https://github.com/WaseemGhanem98/AgentCheck.git
cd AgentCheck
python -m pip install "agentcheck-ai[openai-agents] @ file://$PWD"
```

The repository ships a deliberately flawed example agent. Its model is local and
scripted, so this makes **no provider calls and needs no API key**:

```bash
agentcheck inspect  examples/evaluation/account_agent
agentcheck generate examples/evaluation/account_agent
agentcheck test     examples/evaluation/account_agent
agentcheck report   examples/evaluation/account_agent --latest
```

`agentcheck test` is *expected* to report failures against it — the example
contains five planted defects (deletes without confirmation, resolves an
ambiguous account to the wrong ID, retries a destructive call after a timeout,
claims an email update succeeded when the tool errored, applies the same side
effect twice). That is the demonstration, not a bug.

### Once published

The distribution will be **`agentcheck-ai`**, so the install will be:

```bash
pip install "agentcheck-ai[openai-agents]"
```

The import package and the command stay `agentcheck` either way —
`import agentcheck`, `agentcheck --help`. Only the name you `pip install`
differs, the way `scikit-learn` installs `sklearn`.

**Do not `pip install agentcheck`.** That name on PyPI belongs to an unrelated
project, and nothing here has been published yet. Use the source install above.

## Supported frameworks

| Integration | Install surface | Support contract |
|---|---|---|
| OpenAI Agents SDK | `agentcheck-ai[openai-agents]` | `openai-agents >=0.20,<0.21` |
| PydanticAI | `agentcheck-ai[pydantic-ai]` | `pydantic-ai-slim >=2.32,<2.33` |
| Custom Python agent | base `agentcheck-ai` package | `CustomAgentProtocol`: inert `ToolDefinition` values plus synchronous `start` / `resume` methods |

AgentCheck supports OpenAI Agents SDK and PydanticAI natively, plus custom
Python agents through a lightweight integration contract. It does not have
native adapters for LangGraph, CrewAI, AutoGen, or other framework objects, and
there is no partial or best-effort inspection mode. Code you own can use the
custom contract; pointing `adapter: "custom"` at an arbitrary framework object
does not make that object supported.

The two SDK gates are pinned to a **single verified minor**, deliberately. An
SDK adapter reconstructs an `AgentSpec` by reading framework-private attributes.
On an unverified version those attributes can move, and the failure mode is not
a clean crash — it is a subtly wrong spec, which means a suite that confidently
tests the wrong thing. AgentCheck refuses the version instead of guessing. The
custom integration has no framework version to widen: preflight validates its
small public contract directly.

### SDK setup and offline runs

For PydanticAI, follow the
[self-contained setup guide](docs/pydantic-ai.md) or run the
[credential-free worked example](examples/evaluation/pydantic_agent/README.md).
It covers the exact supported extra, target export, fixtures and prerequisites,
`inspect` / `generate` / `test`, and the difference between a local scripted
model and ControlledModel.

For OpenAI Agents SDK, install the verified
`agentcheck-ai[openai-agents]` extra (`openai-agents >=0.20,<0.21`). Set
`"controlled_model": true` in `agentcheck.json` to substitute AgentCheck's
deterministic, neutral offline model during evaluation. It makes no provider
request and never chooses a tool, so check the CLI's action-path coverage before
treating a passing action case as exercised. The bundled
[account-support example](examples/evaluation/account_agent/README.md) instead
uses a local scripted SDK model to exercise tool calls without a provider.

## How evaluation works

```text
your agent
   ↓  inspect (no turn is run)
AgentCheck adapter
   ├─ SDK target: rebuilds each tool with an AgentCheck-owned invoker
   └─ custom target: reads inert ToolDefinition values and supplies ToolRuntime
   ↓
isolated child process  ── network denied, credentials absent
   ↓
declared tool request  ── SDK replacement or custom tools.call(...)
   ↓
ToolGateway  ── validates and simulates; unknown tools fail closed
   ↓
simulated fixture  ── seeded world, prerequisites, injected faults
   ↓
behavioural oracle
   ↓
PASS / FAIL / INCONCLUSIVE / INFRA_ERROR  + report + replay manifest
```

The declared-tool boundary is the safety argument. For an SDK target, AgentCheck
does not wrap your handler and discard the result — it replaces the invoker, so
the original callable stays on the source agent and is never reached. For a
custom target, no handler is supplied at all; the agent receives a
`ToolRuntime` whose calls go to the same `ToolGateway`. See
[Custom Python agents](docs/custom-agents.md) for that contract and its exact
boundary.

## Simulated tools

Tool results come from a simulated world, seeded per scenario:

- **fixtures** supply representative records so action paths are reachable;
- **prerequisite chains** let a scenario satisfy the lookups that gate a focal
  action, so the interesting call is actually attempted;
- **injected faults** produce the timeout and error cases that expose bad retry
  and bad success-reporting behaviour;
- **confirmation policies** express which actions require the user to agree
  first, making "deleted without asking" a detectable verdict.

An unknown tool is an error, never an improvised result. A harness that invents
tool output invents passing runs.

## Interactive scenarios

A scenario is not limited to one opening prompt. `followup_turns` let the
scripted user answer the agent *mid-run*.

This is what makes confirmation behaviour testable. The agent discloses what it
is about to do and asks; the scenario supplies the answer; the run continues
from there. Because the follow-up does not exist until the agent has finished
speaking, it cannot be mistaken for prior consent — and the confirmation is
carried as structured metadata, so no prose is parsed and no consent is inferred
from free text.

## Verdicts

| Verdict | Meaning | Exit code |
|---|---|---|
| `PASS` | Oracles were evaluated and the agent satisfied them. | `0` |
| `FAIL` | Oracles were evaluated and the agent violated them. | `1` |
| `INCONCLUSIVE` | The run could not authoritatively decide. **Not** evidence of correctness. | `3` |
| `INFRA_ERROR` | The harness itself failed. Says nothing about the agent. | `2` |

The separation is load-bearing. Collapsing `INCONCLUSIVE` or `INFRA_ERROR` into
`PASS` would let a harness fault read as a clean bill of health, which is the
most dangerous thing an evaluation tool can do. An infrastructure failure is
never reported as a behavioural failure, and vice versa.

## Determinism and replay

Be precise about this, because it is where evaluation tools usually oversell.

**Deterministic:** scenario generation, suite freezing and fingerprinting, tool
simulation, fixture and fault injection, oracle evaluation, and verdict
assignment given the same recorded inputs. Offline evaluation through the
controlled model is deterministic, which is why the bundled example costs
nothing and needs no credential.

**Not deterministic:** a real provider model. Its outputs vary between runs, and
so can the verdict.

A **replay manifest is a re-execution recipe, not a captured provider replay.**
It reproduces the inputs and the harness; it does not replay recorded model
output as though the model were deterministic. AgentCheck does not make an LLM
deterministic and does not claim to.

Note that suite fingerprints incorporate the target's absolute entrypoint path,
so they are stable per machine and location rather than portable across them.

## Reports

Each run writes to the target's artifacts directory (`.agentcheck/` by default):

- a standalone **HTML report** with per-case findings;
- **JSON/JSONL artifacts** for the run and its cases;
- a **replay manifest**;
- rows in a local **SQLite** store, queryable through `agentcheck report`.

`agentcheck replay` re-executes a stored manifest; `agentcheck shrink` minimizes
a failing case while preserving its failure signature; `agentcheck review`
records a human decision on a finding.

## CI usage

`agentcheck baseline` exists so CI fails on **new** regressions rather than on
the entire historical backlog:

```bash
# once, locally, from a run you have actually reviewed:
agentcheck test "$TARGET" --no-store
agentcheck baseline create "$TARGET" --latest --out agentcheck-baseline.json

# in CI:
agentcheck test "$TARGET" --no-store --run-id "ci-$RUN_ID"
agentcheck baseline check "$TARGET" --baseline agentcheck-baseline.json \
  --run-id "ci-$RUN_ID" --json
```

A failing test run is never an implicit baseline — you commit one explicitly or
the gate refuses to run. `.github/workflows/agentcheck-example.yml` is a
copyable template that needs no credentials and calls no hosted service.

## Safety model

AgentCheck is built for **trusted local targets**: code you already run. It is
not a sandbox for hostile code, and inspection imports your module, so
module-level code executes.

Within that boundary:

- **Declared real tool handlers never execute** during a simulated evaluation.
  SDK tools are rebuilt with an AgentCheck-owned invoker; custom agents supply
  declarations without a handler and reach tools only through `ToolRuntime`.
- **No real declared-tool mutations.** Calls through the replacement invoker or
  `ToolRuntime` change only the simulated world.
- **Worker isolation.** Each scenario runs in a child process whose environment
  allowlist is empty by default, so provider credentials are absent unless you
  add them explicitly.
- **Fail closed.** Unknown tools, dynamic instructions, output validators, and
  anything else that cannot be proven statically produce a refusal with a
  reason, never an approximation.
- **Network denied by default**, including AF_UNIX, so a target cannot quietly
  reach a local model server.
- **Source integrity.** Runs are bound to the source fileset they came from, so
  a replay cannot silently drift onto changed code.
- **Bounded credential redaction** at the artifact and log boundary.

For a custom target, the orchestration inside `start()` and `resume()` really
executes. A direct filesystem, subprocess, or local-database side effect written
there is outside the declared-tool guarantee. Isolation and denied egress remain
active, but AgentCheck does not claim they sandbox arbitrary local Python side
effects.

Network denial is not a general operating-system sandbox and does not stop
those local effects.

These are the properties the test suite is built around; see
[CONTRIBUTING.md](CONTRIBUTING.md) for the invariants contributors must preserve.

## Known limitations

- **Two native SDK adapters** are each pinned to one minor version. Custom
  Python agents use the separate declaration contract; arbitrary framework
  objects are not accepted through it.
- **PydanticAI targets are refused** when they declare dynamic instructions,
  output validators, `deps_type` dependency injection, or agent capabilities —
  all of these would require executing target code, so the adapter fails closed
  rather than approximating.
- **Custom orchestration is trusted code.** Direct arbitrary Python side effects
  inside `start()`, `resume()`, imports, or helpers are outside the declared-tool
  guarantee.
- **ControlledModel is unsupported for custom agents.** Their model calls live
  inside orchestration AgentCheck cannot replace, so the configuration is
  refused rather than accepted with a silent fallback.
- **Custom model turns are unobservable.** Agent turns are not model turns; an
  unobserved count is recorded as unknown, and a required model-turn constraint
  becomes `INCONCLUSIVE` rather than a vacuous `PASS`.
- **Trusted targets only.** Importing an agent executes its module-level code.
- **Prerequisite fixtures are single-use** within a scenario; an agent that
  retries a gating lookup can exhaust them.
- **Omission is harder than commission.** AgentCheck detects "the agent did
  something it should not have" far more readily than "the agent failed to do
  something it should have", because omission usually needs an authoritative
  contract stating the obligation.
- **Provider-backed runs are stochastic** and cost money. The deterministic
  offline path is the CI-safe one.
- **`INCONCLUSIVE` is common** and is not a soft pass.
- **No historical buggy→FAIL / fixed→PASS benchmark exists.** One was attempted
  against eight real candidates and none qualified; the search is recorded as a
  negative result in [validation evidence](docs/validation-evidence.md). No
  detection-rate figure is claimed anywhere.
- **Not published to PyPI**, and pre-1.0: contracts are versioned and
  fingerprinted but not yet stable across releases by policy.

## Development

```bash
python -m pip install -e ".[dev]"  # agentcheck-ai[dev] from this checkout

python -m pytest tests -q -n 2                              # full suite
python -m pytest tests/agentcheck/test_openai_adapter.py -q  # focused
python -m ruff check agentcheck tests scripts
python -m mypy agentcheck
python -m build
```

Tests must be offline, credential-free, and free of provider spend.

## Documentation

- [Custom Python agents](docs/custom-agents.md) — contract, configuration,
  declared-tool boundary, and observability limitations
- [Development history](docs/development-history.md) — where this came from and
  what was built when
- [Validation evidence](docs/validation-evidence.md) — what has actually been
  demonstrated, and what has not
- [CI trust model](docs/ci-trust-model.md) — how CI runs and what must change
  before this repository goes public

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). One rule outranks the rest: an adapter
that lets the original tool handler run during a simulated evaluation is not
acceptable, however convenient.

## Security

See [SECURITY.md](SECURITY.md) for private vulnerability reporting.

## Relationship to AgentLens

AgentCheck began as an evaluation subsystem inside AgentLens, a separate
observability product, and was extracted into this repository as an independent
project.

They do different jobs: **AgentCheck tests behaviour**, deciding whether an agent
did something unsafe. **AgentLens provides observability and debugging** for
agent runs. AgentCheck does not depend on AgentLens, does not require it, and
does not talk to it — everything here works with AgentCheck alone.

Because AgentCheck writes its results as local artifacts, any platform can ingest
them without AgentCheck taking on a dependency in return.

## License

MIT — see [LICENSE](LICENSE).
