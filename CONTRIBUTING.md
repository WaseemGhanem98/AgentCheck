# Contributing to AgentCheck

Thanks for considering a contribution. AgentCheck is an evaluation tool, which
means a bug here does not just break a feature — it can produce a confident
`PASS` for an agent that is unsafe. The review bar reflects that.

## Environment setup

```bash
git clone https://github.com/WaseemGhanem98/AgentCheck.git
cd AgentCheck
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`[dev]` installs both framework extras plus the test and quality tooling.
Python 3.10+ is required; CI validates 3.10 through 3.12.

## Testing

Run the focused suite for what you touched — the full suite is slow and CI runs
it anyway:

```bash
python -m pytest tests/agentcheck/test_openai_adapter.py -q
python -m pytest tests/agentcheck/test_pydantic_ai_adapter.py -q
python -m pytest tests/agentcheck/test_world_gateway.py -q
python -m pytest tests -q -n 2                 # everything
```

Tests must not require provider credentials, must not make network calls, and
must not spend money. If a test needs a model, use the scripted local model the
existing tests use.

## Code quality

```bash
python -m ruff check agentcheck tests
python -m mypy agentcheck
python -m build                                 # packaging must stay buildable
```

## Safety invariants

These are not style preferences. A change that breaks one of them will be
rejected regardless of how much it improves anything else.

1. **The original tool handler must never execute during a simulated
   evaluation.** Interception happens *before* the handler, by replacing the
   invoker — not by wrapping the call and discarding the result, and not by
   trusting a handler to be side-effect free.
2. **Unknown tools fail closed.** If the gateway does not recognize a tool, that
   is an error. Never synthesize a plausible result. A harness that invents tool
   output invents passing runs.
3. **No real mutations.** Evaluation touches the simulated world only.
4. **Worker isolation holds.** Scenarios run in child processes with a
   constrained environment. Do not widen the environment allowlist by default,
   and do not leak credentials into a worker.
5. **Network stays denied by default.** Reaching a non-allowlisted destination
   is a containment failure that must surface, not be swallowed.
6. **`INCONCLUSIVE` never collapses into `PASS`.** If the harness could not
   decide, it must say so. Same for `INFRA_ERROR`.
7. **Redaction runs at the artifact and log boundary**, before anything is
   written or printed.
8. **Do not overstate replay.** Replay reproduces recorded inputs and harness
   behavior. It does not make a model deterministic, and no message or document
   may imply that it does.

## Things that are never acceptable trade-offs

Green CI is not the goal; a trustworthy verdict is. These are the shortcuts that
look reasonable in the moment and are always rejected:

- **Never widen a safety or resource budget to make CI pass.** Scenario
  wall-clock budgets are a product default, and the suite's ability to notice a
  starved scenario depends on them. If a run is starving, reduce concurrency or
  shard the job — never raise the budget, which trades away the signal to hide
  the symptom.
- **Never turn `INCONCLUSIVE` or `INFRA_ERROR` into `PASS`** to clear a red
  build. A harness fault reported as a clean bill of health is the single most
  dangerous output this project can produce.
- **Never weaken a test to match changed behaviour** without saying so
  explicitly in the PR. If a fingerprint or a contract moved, that is a
  compatibility event and needs to be argued, not absorbed.
- **Never widen a framework version gate without evidence** that the new version
  was actually verified. "It seemed to work" is not evidence; adapters read
  framework-private attributes, so an unverified version fails by producing a
  wrong `AgentSpec`, not by crashing.
- **Keep determinism claims scoped.** Generation, freezing, simulation, and
  oracle evaluation are deterministic. A provider model is not. Do not describe
  replay as reproducing model behaviour — it re-executes a recipe. Documentation
  that overstates this is a defect like any other.

## Framework adapter expectations

New adapters are welcome, but an adapter is the highest-risk code in the
project. An adapter PR is expected to:

- intercept **before** the original handler, and leave the original callable on
  the source agent;
- pin its supported framework version to a **verified range**, and refuse
  anything outside it with a clear message rather than degrading silently.
  Version gates exist because adapters read framework-private attributes; on an
  unverified version the failure is a wrong `AgentSpec`, not a clean crash;
- live behind its own optional extra, so users of one framework never install
  another;
- raise an actionable `AdapterDependencyError` when its extra is missing,
  instead of surfacing an import traceback;
- ship tests that prove the original handler did not run.

Widening an existing version gate needs evidence that the new version was
actually verified, not just that it appeared to work once.

## Pull request workflow

1. Branch per coherent change.
2. Keep the diff scoped. Unrelated refactors in a safety-critical file make
   review harder and are usually asked to be split out.
3. Explain *why* in the PR body, especially for anything touching adapters, the
   tool gateway, isolation, or verdict semantics.
4. Note explicitly if a change affects a fingerprint, a serialized contract, or
   a stored artifact's shape — those are compatibility events. Suite
   fingerprints include the generator version, so they are easy to move by
   accident.
5. Make sure CI is green. CI needs no secrets; if your change appears to need
   one, that is worth discussing first.
