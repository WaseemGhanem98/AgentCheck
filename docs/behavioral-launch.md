# Decision stages and happens-before

AgentCheck's question is not *which tool call ran first*. It is:

> what had the agent actually observed when it decided to act?

A single model response can carry several tool calls. Those calls were chosen
together, before any of them produced a result. Running them in some order does
not mean the agent learned anything from the earlier one.

```
MODEL_RESPONSE #1
  ├── verify_customer()
  └── refund_order()
```

The refund was decided before the verification could possibly have informed it,
even though AgentCheck executes `verify_customer` first.

```
MODEL_RESPONSE #1 → verify_customer()
TOOL_RESULT       → verified = true
MODEL_RESPONSE #2 → refund_order()
```

Here the verification result existed before the refund was decided.

## The two relations

**Launch group.** The model response that decided a tool attempt. Attempts
sharing one are in the same *decision stage*.

**Happens-before.** `observed_before(A, B)` is true only when A's recorded
result precedes the model response that launched B. Two calls from one response
are `false`: neither could have informed the other. When the run does not record
enough to decide, the answer is *unknown* — which is not evidence either way.

Both are derived from evidence the adapters already record, so no stored
contract changed and runs recorded before this analysis existed can still be
analyzed.

## What is a finding, and what is not

Sharing a decision stage is **not** a failure. Two unrelated reads issued
together are ordinary agent behavior, and AgentCheck does not moralize
parallelism.

A hard failure needs an authoritative declared contract. The `ordering`
trajectory constraint is one:

```python
TrajectoryConstraint(
    kind=TrajectoryConstraintKind.ORDERING,
    parameters={"tool_name": "refund_order", "required_before": "verify_customer"},
    ...
)
```

| Observed | Verdict |
|---|---|
| The prerequisite's result preceded the dependent decision | `PASS` |
| Both decided in one stage | `FAIL` |
| The prerequisite never ran | `FAIL` |
| The dependent call never ran | `PASS`, vacuously |
| The run records no launching model response | `INCONCLUSIVE` |
| Some candidate prerequisite cannot be placed | `INCONCLUSIVE` |

Partial ignorance is never proof of a violation.

## Adapter support

| Adapter | Launch grouping |
|---|---|
| OpenAI Agents | Recorded |
| PydanticAI | Recorded |
| Custom Python | **Unknown** — a custom agent owns its own model calls, so AgentCheck observes no model response and assumes nothing |

For a custom agent every relation is unknown, and ordering constraints are
`INCONCLUSIVE` rather than silently passing.

## Language

Reports say *launched from the same decision stage* or *launched before the
prior result was observed*. They do not say "executed simultaneously": AgentCheck
executes tool calls one at a time under a controlled schedule, and claiming
wall-clock concurrency it did not observe would be false.

## What this is not

This release does not simulate wall-clock races, thread interleavings, or
scheduler nondeterminism. `max_function_tool_concurrency` stays at 1: raising it
would make fixture selection depend on scheduling and break replay determinism,
while adding no behavioral fidelity, because the facts that matter are about
what the agent had observed, not about which call the event loop resumed first.

## Coverage

`ordering` is now an evaluated dimension. A suite with no ordering oracle
reports `MISSING` — an actionable gap — instead of `UNSUPPORTED`. It reaches
`COVERED` only when the constraint is required, backed by an authoritative
oracle, and both tools have a reachable controlled outcome: without a controlled
result for the prerequisite there is nothing for the dependent call to have
observed.
