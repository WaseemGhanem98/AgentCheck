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
