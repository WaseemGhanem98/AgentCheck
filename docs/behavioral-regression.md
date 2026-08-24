# Behavioral regression comparison

`agentcheck compare` classifies the behavioral differences between two stored
runs of one target. It reads only stored JSON/JSONL artifacts: it imports
nothing, executes no target, invokes no tool, and re-derives no verdict.

```bash
agentcheck compare ./my-agent --base run-2024-06-01 --head run-2024-06-08
agentcheck compare ./my-agent --base run-2024-06-01 --latest --json
```

This is not the same tool as `agentcheck baseline check`. A baseline is a
curated, committed set of accepted failure identities for a CI gate. A run
comparison answers a different question — *what changed between these two
executions* — and needs nothing committed.

## Scenario identity, not array position

Scenarios are matched by structural fingerprint. A run can order, select, or
lint its cases differently, so position is never an identity. A scenario whose
fingerprint changed is matched by display ID instead and reported as
`changed_scenario`, because its stimulus or oracle moved and its two verdicts
answer different questions.

## Categories

| Category | Meaning | Blocks |
|---|---|---|
| `new_regression` | Head FAILs a scenario the base did not FAIL | yes |
| `changed_failure` | Both FAIL, with different failure signatures | yes |
| `unchanged_failure` | Both FAIL with the same failure signature | no |
| `resolved_failure` | Base FAILed, head PASSes | no |
| `uncertifiable_failure` | A FAIL whose identity could not be established | no, exits 2 |
| `infra_change` | Either side is `INFRA_ERROR` | no, exits 2 |
| `inconclusive_change` | A verdict moved into or out of `INCONCLUSIVE` | no |
| `changed_scenario` | Matched by ID with a different fingerprint | no |
| `new_scenario` / `removed_scenario` | Present on only one side | no |
| `deselected_scenario` | Absent because the head run's selection excluded it | no |
| `unchanged` | Same verdict, same identity | no |

`INCONCLUSIVE` and `INFRA_ERROR` never collapse into `PASS`. A `FAIL` that
became `INCONCLUSIVE` is not a resolved failure: weak evidence is not a pass.
An `INFRA_ERROR` on either side is never a behavioral regression and never a
behavioral pass; it exits 2 because AgentCheck could not certify the outcome.

A scenario the head run excluded before execution is `deselected_scenario`, not
`removed_scenario`: that run holds no evidence about it either way. The reverse
direction has no separate category, because a scenario the *base* run excluded
already reports as `new_scenario`, which never blocks and never claims lost
coverage.

Detail rows are bounded, and blocking and uncertifiable rows are ordered first,
so a large suite omits only quiet detail. The counts are always computed over
every scenario, never over the visible rows.

## Comparability

Two runs can differ in ways that change what a verdict difference means. Each
such binding is reported as an explicit caveat rather than silently ignored:

- `spec_changed` — the inspected specification changed, so a difference may
  describe a different agent.
- `scenario_set_changed` — the runs evaluated different scenario sets, so added
  and removed scenarios are expected.
- `suite_fingerprint_changed` — both runs recorded a frozen suite fingerprint
  and they differ.
- `seed_changed` — scenario identities may differ for reasons unrelated to the
  agent.
- `selection_active` — an absent scenario may have been excluded rather than
  removed.
- `source_revision_unrecorded` — a difference cannot be attributed to a source
  change.
- `source_revision_unchanged` — both runs came from the same source revision,
  so any difference is run-to-run variation.
- `scenario_identity_changed` — at least one scenario matched by ID with a
  different fingerprint.

A comparison is refused outright, with exit code 3 and no scenario
classification, only when there is nothing to compare: both arguments resolved
to the same run, or the two runs share no scenario identity at all. Refusing is
not a clean result.

## Stochastic executions are not deterministic

Every comparison states this, and it is the most important sentence in the
output:

> A verdict pair is evidence about these two executions. AgentCheck reproduces
> inputs and harness behavior, not model output, so an unchanged verdict is not
> proof that a behavior is stable and a changed verdict is not proof that the
> source caused it. Repeat runs to establish stability.

Replay is source-bound re-execution, not deterministic provider replay. A
single pair of executions of a stochastic agent cannot establish that a
behavior is deterministic, and `agentcheck compare` never claims that it does.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Comparison completed with no attributable regression |
| 1 | At least one `new_regression` or `changed_failure` |
| 2 | An outcome AgentCheck could not certify (`infra_change`, `uncertifiable_failure`) |
| 3 | The runs are incomparable |

## Compatibility

`agentcheck.run_comparison.v1` is a new derived contract. It adds no field to
`Scenario`, `FrozenSuite`, `agentcheck.baseline.v1`,
`agentcheck.baseline_comparison.v1`, replay manifests, or review records, so no
existing fingerprint moves. The per-scenario classifier is the one the baseline
gate already uses, so a repository cannot block on two different definitions of
"regression".
