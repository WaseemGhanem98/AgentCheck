# AgentCheck

AgentCheck runs your AI agent against generated adversarial scenarios in an
isolated process, with every tool call simulated, and tells you whether the
agent behaved safely.

> **Status: pre-1.0 (0.1.0), alpha.** The contracts described here are
> implemented and tested, but the project is young and the supported framework
> surface is deliberately narrow. See [Known limitations](#known-limitations).

## The problem

An agent that passes your unit tests can still delete the wrong account. The
failures that matter are behavioral: it skips a confirmation, it retries a
destructive call after an ambiguous timeout, it picks the wrong record out of
an ambiguous request, it reports success after the tool returned an error.

Testing that by hand means either writing scenarios yourself for every risky
path, or letting the agent touch real systems to find out. The first does not
scale. The second is how you find out in production.

## What AgentCheck does

1. **Inspects** a local agent you already trust — its tools, their schemas, its
   instructions, its guardrails — by importing it, without running a turn.
2. **Generates** a suite of scenarios from what it found, then freezes the suite
   into a fingerprinted document so the same target, config, and seed always
   produce a byte-identical suite.
3. **Runs** each scenario in an isolated child process where every tool call is
   intercepted before the original handler and answered from a simulated world.
4. **Judges** the resulting trajectory against per-scenario oracles and emits
   `PASS` / `FAIL` / `INCONCLUSIVE` / `INFRA_ERROR` plus an HTML report and a
   replay manifest.

Your real tool handlers never execute. Nothing outside the sandbox is mutated.

## How it works

```
your agent  ──inspect──▶  AgentSpec  ──generate──▶  frozen suite (fingerprinted)
                                                          │
                                                          ▼
                                         isolated child process, per scenario
                                                          │
                            ┌─────────────────────────────┴──────────────┐
                            │  agent asks to call a tool                 │
                            │  ToolGateway intercepts BEFORE the handler │
                            │  simulated world answers from fixtures     │
                            │  unknown tool ⇒ fail closed, never guessed │
                            └─────────────────────────────┬──────────────┘
                                                          ▼
                                       verdict + findings + report + manifest
```

The interception point is the whole safety argument. AgentCheck does not wrap
your handler and hope it is side-effect free — it replaces the invoker, so the
original callable is left behind on the source agent and is never reached
during a simulated evaluation.

## Installation

AgentCheck is **not on PyPI yet**, so install it from a clone or a built wheel.

```bash
git clone https://github.com/WaseemGhanem98/AgentCheck.git
cd AgentCheck
python -m pip install -e ".[openai-agents]"     # or ".[pydantic-ai]", or ".[all]"
```

Or build and install a wheel:

```bash
python -m build
python -m pip install "dist/agentcheck-0.1.0-py3-none-any.whl[openai-agents]"
```

Once a release is published, the install will be `pip install "agentcheck[openai-agents]"`.
Do not assume that works today; it does not.

The base install has no framework in it. Install the extra for the framework you
actually evaluate — evaluating an OpenAI Agents target should not drag PydanticAI
onto your machine. If an extra is missing, the adapter says so and names the
command to fix it rather than raising an import traceback.

## Quickstart

The repository ships a deliberately flawed example target. Its model is local
and scripted, so this makes **no provider calls and needs no API key**.

```bash
python -m pip install -e ".[openai-agents]"

agentcheck inspect  examples/evaluation/account_agent
agentcheck generate examples/evaluation/account_agent
agentcheck test     examples/evaluation/account_agent
agentcheck report   examples/evaluation/account_agent --latest
```

The example agent has five planted defects — it deletes without confirmation,
resolves an ambiguous account to the wrong ID, retries a destructive call after
a timeout, claims an email update succeeded when the tool errored, and applies
the same side effect twice. `agentcheck test` is expected to report failures
against it. That is the demonstration, not a bug.

## Supported frameworks

| Framework | Extra | Supported versions |
|---|---|---|
| OpenAI Agents SDK | `agentcheck[openai-agents]` | `openai-agents >=0.20,<0.21` |
| PydanticAI | `agentcheck[pydantic-ai]` | `pydantic-ai-slim >=2.32,<2.33` |

Both gates are pinned to a **single verified minor**, on purpose. Each adapter
reads framework-private attributes to reconstruct an `AgentSpec`. On an
unverified version those attributes can move, and the failure mode is not a
clean crash — it is a subtly wrong spec, which means a suite that tests the
wrong thing. AgentCheck refuses the version rather than guess.

No other framework is supported. There is no partial or best-effort mode.

## Safety model

AgentCheck is designed for **trusted local targets**: code you already run. It
is not a sandbox for hostile code, and importing an agent runs that agent's
module-level code.

Within that boundary:

- **No original handler execution.** Accepted tools are reconstructed with an
  AgentCheck-owned invoker. The original callable and every advanced callback
  stay on the source agent.
- **Process isolation.** Each scenario runs in a child process with a
  constrained environment. The environment allowlist is empty by default, so
  provider credentials are absent unless you explicitly add them.
- **Fail closed on unknown tools.** A tool the gateway does not recognize is an
  error, never an improvised result. Guessing a plausible response is how a
  harness invents a passing run.
- **Network containment.** Network is denied by default; reaching a
  non-allowlisted destination is a containment failure, not a silent success.
- **Bounded credential redaction** at the artifact and log boundary, applied
  before anything is written to a report or printed.
- **Source integrity.** Runs are bound to the source fileset they were produced
  from, so a replay cannot silently drift onto changed code.

## Simulated tools

Tool results come from a simulated world, seeded per scenario:

- **fixtures** supply representative records so action paths are reachable;
- **prerequisite chains** let a scenario require that an earlier step happened;
- **injected faults** produce the timeout and error cases that expose bad retry
  and bad success-reporting behavior;
- **confirmation policies** express which actions require the user to agree
  first, so "deleted without asking" is a detectable verdict.

## Interactive scenarios

Scenarios are not limited to a single opening prompt. `followup_turns` let the
scripted user answer the agent mid-run, which is what makes confirmation
behavior testable: the agent asks, the scenario answers, and the run continues
from there.

## Reports

Every run writes to the target's artifacts directory (`.agentcheck/` by
default):

- an **HTML report** of the run, per case, with findings;
- a **replay manifest** describing exactly what was executed;
- stored run records queryable through `agentcheck report`.

`agentcheck replay` re-executes a stored manifest, and `agentcheck shrink`
minimizes a failing case while preserving its failure signature.

## Verdicts

| Verdict | Meaning | Exit code |
|---|---|---|
| `PASS` | The oracles were evaluated and the agent satisfied them. | `0` |
| `FAIL` | The oracles were evaluated and the agent violated them. | `1` |
| `INCONCLUSIVE` | The run could not authoritatively decide — e.g. a budget was not measurable. Not evidence of correctness. | `3` |
| `INFRA_ERROR` | The harness itself failed. Says nothing about the agent. | `2` |

The separation is deliberate. Collapsing `INCONCLUSIVE` into `PASS` would let a
harness failure read as a clean bill of health, and that is the single most
dangerous thing an evaluation tool can do.

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

A failing test run is never an implicit baseline. You commit a baseline
explicitly or the gate refuses to run.

`.github/workflows/agentcheck-example.yml` in this repository is a copyable
template. It needs no provider credentials and calls no hosted service.

## Deterministic vs stochastic

Be precise about what is guaranteed, because this is where evaluation tools
usually oversell:

**Deterministic.** Scenario generation, suite freezing and fingerprinting, tool
simulation, fixture and fault injection, oracle evaluation, verdict assignment,
and the report contents given the same recorded inputs. The same target, config,
and seed freeze a byte-identical suite.

**Not deterministic.** The model. If you evaluate against a real provider, the
agent's own outputs can vary between runs, and so can the verdict. AgentCheck
does not make an LLM deterministic and does not claim to. That is why the
bundled example uses a scripted local model, and why replay re-executes a
manifest rather than replaying recorded model output as if it were a guarantee.

A replay reproduces the **inputs and the harness**, not the model's freedom.

## Known limitations

- Two framework adapters, each pinned to one minor version. That is the entire
  supported surface.
- Trusted targets only. Importing an agent executes its module-level code.
  AgentCheck is not a defense against malicious agent source.
- Pre-1.0. Contracts are versioned and fingerprinted, but not yet stable across
  releases by policy.
- Not published to PyPI. Install from source or a built wheel.
- Scenario generation derives from what inspection can see. An agent whose
  behavior depends on state AgentCheck cannot observe will be under-tested.
- Evaluating against a real provider costs money and reintroduces model
  variance. The deterministic path is the one that is CI-safe.
- AgentCheck reports what its oracles can decide. `INCONCLUSIVE` is common and
  is not a soft pass.

## Development

```bash
python -m pip install -e ".[dev]"

python -m pytest tests -q                 # full suite
python -m pytest tests/agentcheck/test_openai_adapter.py -q
python -m ruff check agentcheck tests
python -m mypy agentcheck
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). One rule matters more than the rest: an
adapter that lets the original tool handler run during a simulated evaluation
is not acceptable, however convenient it is. Read the safety invariants before
opening an adapter PR.

## Security

See [SECURITY.md](SECURITY.md) for private vulnerability reporting.

## Relationship to AgentLens

AgentCheck is developed alongside [AgentLens](https://github.com/WaseemGhanem98/AgentLens),
a separate observability platform for AI agent runs.

They do different jobs. **AgentCheck tests behavior** — it decides whether an
agent did something unsafe. **AgentLens provides observability and debugging**
for agent runs. AgentCheck does not depend on AgentLens, does not require it,
and does not talk to it. Everything documented here works with AgentCheck alone.

AgentCheck writes its results as local JSON artifacts and reports, which is a
deliberate integration boundary: any platform, AgentLens included, can ingest
those artifacts without AgentCheck taking on a dependency in return.

## License

MIT — see [LICENSE](LICENSE).
