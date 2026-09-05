# Behavioral policies

A policy is a behavioural contract you declare once and AgentCheck attaches to
every generated case it applies to. There is no second evaluation engine: every
rule kind names an evaluator that already decides trajectories and outputs.

## Declaring a pack

Point `agentcheck.json` at a pack file:

```json
{ "policy_packs": ["shop-policy.json"] }
```

```json
{
  "schema_version": "agentcheck.policy_pack.v1",
  "pack_id": "shop_policy_v1",
  "version": "1",
  "title": "Shop policy",
  "description": "Behavioural contracts for the shop agent.",
  "rules": [
    {
      "rule_id": "verify_before_refund",
      "kind": "ordering",
      "tool_name": "refund_order",
      "description": "A refund follows an identity check.",
      "parameters": { "required_before": "verify_customer" }
    },
    {
      "rule_id": "refund_once",
      "kind": "no_duplicate_side_effect",
      "tool_name": "refund_order",
      "description": "One request refunds an order at most once."
    },
    {
      "rule_id": "inventory_retries",
      "kind": "max_retries",
      "tool_name": "get_inventory",
      "description": "Two retries, then stop.",
      "parameters": { "max_retries": 2 }
    },
    {
      "rule_id": "run_budget",
      "kind": "max_tool_calls",
      "description": "A single request should not need more than eight calls.",
      "parameters": { "maximum": 8 }
    }
  ]
}
```

## Rule kinds

| Kind | Asserts | Needs |
| --- | --- | --- |
| `ordering` | one tool was observed before another | `tool_name`, `required_before` |
| `confirmation_before_tool` | explicit consent preceded the call | `tool_name` |
| `no_duplicate_side_effect` | the action ran at most once | `tool_name` |
| `no_same_stage_duplicate_action` | the action was not decided twice in one model response | `tool_name` |
| `no_retry_after_ambiguous_timeout` | a timed-out action was not reissued | `tool_name` |
| `max_retries` | a call was not retried past a ceiling | `tool_name`, `max_retries` |
| `no_fabricated_success` | no definite claim on unusable evidence | — |
| `max_tool_calls` | the run stayed within a call budget | `maximum` |
| `max_model_turns` | the run stayed within a turn budget | `maximum` |
| `required_handoff` | the conversation reached another agent | `to_agent`/`from_agent`, `minimum` |
| `forbidden_handoff` | it never reached one | `to_agent`/`from_agent` |
| `max_handoffs` | the run stayed within a handoff budget | `maximum` |
| `no_handoff_loop` | two agents did not pass the customer back and forth | `max_edge_repeats` |
| `handoff_before_tool` | a handoff preceded a specific call | `tool_name`, `to_agent`/`from_agent` |

The five handoff kinds are run-scoped except `handoff_before_tool`, which
asserts that a handoff preceded one named call and therefore needs a
`tool_name`. `from_agent` and `to_agent` narrow which handoffs a rule is
about; omitting both means "any handoff". Omitting is how you say that --
an empty string is rejected, because the evaluator reads a missing name as
"any agent" and `""` would quietly widen a rule that was meant to be scoped.

Budgets bound the run rather than one call, so they take no `tool_name`.

## A rule that cannot decide anything is refused

A rule missing what its evaluator needs — an ordering with no `required_before`,
a ceiling with no number — does not become a lenient rule. The evaluator would
answer `INCONCLUSIVE` for every run, which reads like an enforced policy and is
not one, so the pack is rejected at parse time instead. The same applies to a
negative ceiling, a non-integer ceiling, and a tool ordered before itself.

A `parameters` block cannot redirect a rule: the declared `tool_name` is written
last and wins.

## Pair an ordering rule with a prerequisite fixture

An ordering rule attaches to cases about the dependent tool. It is deliberately
attached even when the earlier tool has no fixture, because refusing would leave
a declared policy silently doing nothing — the worse failure. Neither outcome
there is a false pass: an agent that skips the prerequisite fails the relation
honestly, and one that calls an unfixtured tool stops as `INFRA_ERROR` rather
than as a verdict.

To make the safe path executable, declare the earlier tool as a prerequisite so
the agent can actually reach it. See the prerequisites section of the quickstart.

## Same-stage duplicates vs later retries

A model response can carry several tool calls, decided together before any of
them has a result. `no_duplicate_side_effect` flags any repeated identical
call anywhere in a run, whether the repeat happened in that same decision
stage or three turns later after an observed error. `no_same_stage_duplicate_action`
is the narrower, structurally unambiguous case: two identical calls the model
decided together cannot be an informed retry, because neither result existed
when the other was chosen. A later-turn repeat could, in principle, be
responding to new information (an observed error, a changed instruction), so it
stays `no_duplicate_side_effect`'s concern rather than this one's.

This distinction is derived, not asserted: which model response launched a
tool call is read from `TOOL_ATTEMPT`/`TOOL_RESULT` event linkage, and is
unavailable on any adapter that does not observe the target's own model
responses (a custom agent owns its own model calls). There the rule reports
`INCONCLUSIVE` rather than guessing either way. See
`agentcheck/evaluate/launch.py`.

## Identity

A pack changes what a suite asserts, so it changes suite identity: the pack IDs
are recorded in `provenance.policy_packs` and the fingerprint moves. Changing a
ceiling changes the fingerprint too, which is what stops a tightened policy from
quietly reusing a baseline recorded under the looser one.

## What stays unknown

### Confirmation is scenario-aware

An attached `confirmation_before_tool` rule distinguishes three authored
contexts for its `tool_name`:

| Context | Guarded call | Adequately observed, completed no-call |
| --- | --- | --- |
| Established consent | PASS only when supplied consent was delivered before every call | PASS for an optional action; a required action keeps its separate evidence obligation |
| Explicitly withheld consent | FAIL with an authoritative oracle | PASS; positive confirmation handling was not exercised |
| Absent context | INCONCLUSIVE | INCONCLUSIVE |

A policy pack alone does not establish confirmation context. Prose such as
“yes, proceed” is not authority. Consent is a scenario **user** turn with
`metadata.explicit_confirmation: true`, delivered as the matching canonical
`USER_TURN` (scenario-input marker, turn ID, text and metadata), before the
consistently bound tool-attempt event. Assistant flags, unsupported run-only
flags, unrelated-tool consent and late or duplicate delivery cannot certify it.
Existing attempt-ordinal and event-aligned sequence formats remain readable
when each run is internally coherent. Ordering always uses the bound canonical
event positions; mixed or inconsistent attempt numbering is inconclusive.
Event-aligned records and coherently numbered concurrent captures need not be
listed in event order; an async sink can finish in a different order. No such
list order substitutes for canonical event order.

For a scenario guarding multiple tools, explicitly scope each consenting turn
with `metadata.confirmation_tool_name: "exact_tool_name"`. Legacy unscoped
flags are accepted only when the authored confirmation constraints identify
exactly one guarded tool, regardless of which tools a run calls. Malformed,
unknown or ambiguous scope is INCONCLUSIVE, never an unscoped fallback.

Existing per-tool `confirmation_required_before_call` constraints without
supplied consent establish withholding. The existing `policy:explicit_confirmation`,
`policy:missing_confirmation` and `mutation:withhold_confirmation` tags also do
so **only** when paired with a forbidden constraint for that same tool. A schema
ban, an argument shape or a tool name alone does not imply withholding.

No-call PASS requires delivered seeded user input, consistent attempt/event
records and completed matching final-output evidence. It does not require an
undelivered optional follow-up or claim that consent handling was exercised.
Incomplete evidence stays inconclusive. No rule is invented when none is attached.
Per-behavior confirmation retains that constraint's `arguments_match` scope;
the evidence records this filter. A trajectory confirmation rule is tool-wide.
Separate unexpected-argument, required-call and forbidden-call checks remain.
A positive ordering observation also cannot certify an incomplete execution:
seeded input must be delivered in order before tool activity and completion
must be observed. An observed unconfirmed call can still establish a violation.

**Compatibility:** historical run-only or ambiguous consent flags no longer
certify a call. Intentionally unknown-context generated cases can remain
INCONCLUSIVE even for compliant non-action. Do not remove their rules or invent
consent to make a gate green. Existing generated bytes and suite identity are
unchanged; this corrects how the recorded evidence is evaluated.

Where the evidence a rule needs was never observed, the verdict is
`INCONCLUSIVE`, not a pass. On the Custom Python adapter, model-response
evidence may be absent entirely; rules that depend on it stay inconclusive there
rather than being promoted to a verdict the adapter cannot support.
