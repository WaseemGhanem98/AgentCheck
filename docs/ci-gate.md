# The CI gate

```bash
agentcheck gate .
```

One command for the release question: run the trusted suite, compare it to the
trusted baseline, and return one status CI can branch on. It orchestrates the
existing operations — the same `test` execution and the same `baseline check`
comparison — rather than reimplementing either.

## Exit codes

The contract `agentcheck test` already publishes, reused unchanged:

| Code | Meaning | CI should |
| --- | --- | --- |
| `0` | no new behavioural failure | allow |
| `1` | a behavioural failure is new against the baseline | block |
| `2` | the run is **not certifiable** | block |
| `3` | the suite could not decide | block |

## Why "not certifiable" is its own answer

A case that stopped on infrastructure — a missing fixture, a worker that could
not start, a suite or source or baseline that does not match — has said nothing
about the agent. Reporting that as a pass hides a broken harness behind a green
build. Reporting it as a regression blames the agent for something it did not
do. So it is neither: exit `2`, and the gate does not consult the baseline at
all, because there is nothing trustworthy to compare.

An infrastructure error outranks a failure in the same run. A suite that only
half executed cannot certify the half that did.

`INCONCLUSIVE` is never quietly a pass either. Where the evidence a criterion
needs was not observed, the gate blocks with exit `3` and says so.

## Declared risk creates a required evidence obligation

The same principle applies to a requirement that never received a case at all,
not only to a case that ran and could not decide. Both are the same state —
no evidence — so both block.

When you declare a tool's risk, through `tool_risk` in `agentcheck.json` or on
a custom agent's own `ToolDefinition`, that declaration is authoritative. It
makes four behaviours required for that tool:

- `fabricated_success_after_failure`
- `duplicate_action`
- `ambiguous_outcome`
- `retry_control`

If the suite produces no evidence for one of them, the gate blocks with exit
`3` and names the tool, the requirement, and what to do next. A suite that
never exercises a tool you called destructive is not a passing suite.

This is scoped deliberately:

- **Only declared risk counts.** Whether an obligation exists is read from the
  declaration itself, not from what the coverage report happens to say. A tool
  whose risk AgentCheck merely *inferred* from its name can never create an
  obligation, even if its coverage shows a gap. Inference is not authority.
- **Only a total absence of evidence counts.** `partial` evidence does not
  block; it is the ordinary state of a healthy suite.
- **An uncovered tool is not by itself a failure.** `success_path`,
  `failure_handling`, and `timeout_handling` apply to every tool regardless of
  risk and are not part of this floor.
- **A target that declares no risk is unaffected.**

Whether an obligation is met is evaluated from the declared risk and the
scenarios directly, not read back out of the coverage report. The report is a
presentation: its per-requirement detail is bounded and its subject names are
redacted, so a tool with a large schema or a secret-shaped name can be
unreadable there. Deciding from it would let such a tool slip through.

To resolve a block: add cases exercising the listed behaviours, or correct the
`tool_risk` declaration if the tool is not actually state-changing or
destructive.

`max_cases` bounds a run before obligations are evaluated, so a bounded run can
report obligations unmet simply because the cases were not selected. That can
happen well below the limit described next. Raise or remove `max_cases` and
re-run before concluding the evidence is genuinely absent.

### A known limit on large declared surfaces

Generation caps how many cases it emits per origin, so a target that declares
risk on many tools can reach a point where AgentCheck's own generator stops
producing the very cases this floor requires. Measured against the bundled
generator, targets declaring risk on around ten tools or fewer are satisfied by
their generated suite; past that, later tools begin receiving no fault variants
and their obligations stay unmet, and regenerating will not clear it.

**Do not delete a true risk declaration to get past this.** That trades a
correct description of your agent for a green build, which is the opposite of
what the gate is for. Supply the missing cases yourself, or reduce the number
of tools evaluated in one run. Raising the caps is tracked separately.

## Historical failures do not block

A baseline records the failures you have already accepted. The gate blocks on
what is **new** against it, so a target with known defects stays green until its
behaviour changes:

```bash
agentcheck test . --no-store
agentcheck baseline create . --latest --out agentcheck-baseline.json
```

Record a baseline from an explicit command, never from a failing CI run. Without
one the gate still answers the weaker question it can — it blocks on any failure
and tells you how to record a baseline — and says plainly that nothing was
compared.

## Machine-readable output

```bash
agentcheck gate . --json
```

`unmet_risk_obligations` lists any required evidence that was missing, each
with the tool, the dimension, and the coverage reason code.

Prints the decision, exit code, reason, verdict counts, run ID, suite
fingerprint and report path as JSON on stdout, with the human summary on stderr
so both are usable in the same step.

Every exit path prints a document, including the one where the suite never ran
at all -- a frozen suite that no longer matches its target, or a config the CLI
could not resolve. That case is the one a parsing step most needs, because it is
where a broken harness would otherwise look like nothing happened. It reports
`"decision": "block"` with exit code 2, and `"counts": {}` rather than zeroed
counts: no scenario executed, so `"fail": 0` would be a claim this run cannot
make.

## In GitHub Actions

`.github/workflows/agentcheck-example.yml` is a copyable template. It runs on a
GitHub-hosted runner with `contents: read`, no secrets, and no provider
credentials: the suite runs against simulated tools, so nothing in it needs a
model key.
