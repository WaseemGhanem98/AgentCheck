# Contributing to AgentCheck

External contributions are welcome. AgentCheck is an evaluation tool, so a bug
can produce a confident verdict about unsafe behavior; changes receive a
correspondingly careful review.

## Workflow

1. Fork `WaseemGhanem98/AgentCheck`.
2. Create a branch for one coherent change.
3. Add focused, offline tests.
4. Open a pull request and complete the concise safety checklist.
5. Address review and GitHub-hosted CI results.

Only the repository owner approves and merges changes into the official `main`
branch and publishes official releases. Contributors do not need repository
secrets or provider credentials.

## Setup

```bash
git clone https://github.com/YOUR-USER/AgentCheck.git
cd AgentCheck
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Python 3.10 through 3.12 is supported.

## Validation

Run focused tests first. Use exactly `-n 2` for the full suite:

```bash
python -m pytest tests/agentcheck/test_<area>.py -q
python -m pytest tests -q -n 2
python -m ruff check agentcheck tests scripts
python -m mypy agentcheck
python -m build
```

Tests must be offline, credential-free, free of provider spend, and independent
of local machine state. Use existing scripted or controlled models rather than a
real provider.

## Safety invariants

Changes must preserve all of the following:

- original declared tool handlers never execute during simulated evaluation;
- unknown tools fail closed and receive no invented result;
- declared simulated tools mutate only simulated state;
- worker isolation and deny-by-default network containment remain intact;
- `INCONCLUSIVE` and `INFRA_ERROR` never collapse into `PASS`;
- redaction happens before artifact or console output;
- replay remains a source-bound re-execution recipe, not deterministic provider
  replay;
- safety and wall-clock budgets are not widened to make tests pass.

Custom orchestration is trusted code. Direct filesystem, subprocess, or database
effects outside `ToolRuntime` are not covered by the declared-tool guarantee.

## Review expectations

Keep diffs scoped and explain compatibility impact when touching adapters,
serialized contracts, fingerprints, fixtures, policies, worker boundaries, or
verdict logic. Do not widen an SDK version gate without evidence that the new
minor version was tested. Dependency and workflow changes require owner review
and are never auto-merged.

Package releases and frozen-suite generation compatibility use separate
versions. The distribution release (`agentcheck.__version__`) records which
build executed a run and does not participate in suite provenance or suite
fingerprints. `GENERATOR_COMPATIBILITY_VERSION` is recorded in suite provenance
and does participate in suite fingerprints; raise it only when changed
generation semantics should re-identify otherwise-equivalent suites. A package
release alone is not a suite compatibility event.
