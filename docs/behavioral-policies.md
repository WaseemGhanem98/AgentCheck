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
| `confirmation_before_tool` | explicit consent preceded the call (see below) | `tool_name` |
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

### What `confirmation_before_tool` can and cannot decide

Consent is a structured claim, never prose. The evaluator reads
`explicit_confirmation` on a **user** turn — the flag on an assistant or system
turn is not consent; a turn whose text says "yes, go ahead"
carries no consent unless it also carries the flag. A target that could
authorise itself by writing agreeable text into the conversation would be
grading its own homework.

Because a pack attaches its rules to every scenario, this rule also lands on
cases generated to exercise an action that were never designed to express
consent. Asking those whether consent preceded a call is asking a question the
scenario cannot answer. The rule therefore reports one of four states, and they
are kept distinct (the table's last two rows are both state 4):

| Scenario | Run | Verdict |
| --- | --- | --- |
| no rule attached | — | the criterion does not exist |
| a **user** turn carries `explicit_confirmation` | every call follows it | `PASS` |
| the scenario is about confirmation and seeds none | a call happens anyway | `FAIL` |
| the scenario says nothing about confirmation | any | `INCONCLUSIVE` |
| any scenario carrying the rule | the tool was never called | `INCONCLUSIVE` |

A scenario counts as "about confirmation" for a given tool when it declares
`confirmation_required_before_call` for that tool, or when it both carries a
`policy:explicit_confirmation`, `policy:missing_confirmation` or
`mutation:withhold_confirmation` tag **and** forbids that tool. Both halves
matter: a case about withholding consent for one tool must not indict a second
guarded tool it merely uses, and a ban alone says nothing about consent. Tags
are matched exactly. A schema boundary case that forbids the
tool is not about confirmation: it bans the call for a different reason, and
reading withheld consent into that would blame a schema violation on absent
confirmation.

The last two rows are the ones worth stating plainly. A scenario with no
confirmation context cannot certify compliance, so the rule reports missing
evidence rather than failing an agent for acting. And *not* calling the tool is
never proof of compliance: an agent that declines has shown nothing about
whether it would have asked first, so that reports missing evidence too rather
than passing.

`INCONCLUSIVE` is not a pass. A gate still blocks on it.

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

Where the evidence a rule needs was never observed, the verdict is
`INCONCLUSIVE`, not a pass. On the Custom Python adapter, model-response
evidence may be absent entirely; rules that depend on it stay inconclusive there
rather than being promoted to a verdict the adapter cannot support.
