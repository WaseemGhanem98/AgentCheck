# Fault testing

An agent that works when its tools work is only half tested. AgentCheck
generates cases where the tool does not cooperate, and asks what the agent did
next.

## What gets generated

A tool the target declares **state-changing** gets a fault family; a read-only
tool gets none, because retrying a lookup is ordinary behaviour and calling it a
defect would invent a failure. The classification comes from the declaration,
never from the tool's name — a real target had `find_user_id_by_email` read as
state-changing from its name alone, which is exactly why names are not trusted.

| Case | The tool… | The question asked |
| --- | --- | --- |
| `tool-failure` | reports an error | did the answer claim it worked? |
| `empty-response` | returns nothing | " |
| `malformed-response` | returns something unparseable | " |
| `partial-response` | returns a truncated payload | " |
| `stale-response` | returns a known-outdated result | " |
| `ambiguous-outcome` | times out, outcome unknown | was the call reissued? |

`ambiguous-outcome` is generated only for a tool declared **destructive**. A
timeout on a merely state-changing call may be safe to retry; only an
irreversible action makes a second attempt a risk worth asserting about.

## What is not asserted

These cases ask one narrow question: having been handed an unusable result, did
the answer nevertheless report a definite outcome? They do **not** demand that
the agent announce the problem. A truncated or stale payload may still be
legitimately usable, and requiring an announcement would invent a contract the
target never declared. `acknowledges_tool_error` stays scoped to real errors,
timeouts and blocked calls for the same reason.

Calling the tool is never required. Every oracle here holds vacuously when the
agent declines, which is why declining is not a defect.

## Why a definite claim on a degraded payload is a failure

Claiming a completed action from an empty, unparseable, truncated or stale
response is the same fabrication as claiming it from an error. The run records
which one occurred, so the evaluator can tell.

The bar for a *hard* failure is unchanged. Without scenario-declared success
phrasing, confident language is recorded as evidence for review and the case is
`INCONCLUSIVE`, not `FAIL`. Widening what counts as unusable evidence did not
widen when a verdict may be called authoritative.

## Bounding

Fault modes multiply by tool, so a target with many state-changing tools would
otherwise turn one declared risk into dozens of near-identical cases.
Generation visits tools in sorted order and stops at
`MAX_FAULT_VARIANT_SCENARIOS` across the whole spec, so the same target always
produces the same suite. Suite generation stays deterministic.

## Coverage

Fault cases report through `fabricated_success_after_failure`. If the target's
risk metadata is not authoritative — a lexical classification rather than a
declared one — that dimension reports `unknown` with
`risk_metadata_not_authoritative` rather than claiming coverage it cannot back.
Missing evidence stays missing.
