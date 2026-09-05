# Fault testing

An agent that works when its tools work is only half tested. AgentCheck
generates cases where the tool does not cooperate, and asks what the agent did
next.

## What gets generated

A tool marked **state-changing** gets a fault family; a read-only tool gets
none, because retrying a lookup is ordinary behaviour and calling it a defect
would invent a failure.

Where that marking comes from is now explicit, with a fixed precedence:

1. **Developer declaration.** A custom Python agent *declares* it —
   `ToolDefinition(state_changing=True, destructive=True)` — directly on the
   tool. For every adapter, `agentcheck.json` can also declare it per tool:

   ```json
   {
     "tool_risk": {
       "find_user_id_by_email": { "destructive": true },
       "bash": { "state_changing": true, "destructive": true }
     }
   }
   ```

   Each axis (`state_changing`, `destructive`) is independently optional, and a
   declared axis always wins over whatever the adapter would otherwise infer.
   Every key must name a tool the agent actually declares -- a misspelled name
   here would otherwise be a silently ignored override, so `prepare` refuses
   the run instead (`unknown_tool_risk_declaration`), naming the exact key
   that matched nothing.
2. **Framework metadata**, when a framework genuinely exposes an authoritative
   side-effect flag. No adapter AgentCheck currently supports does; the tier
   exists in the model so one that does can use it without a new contract.
3. **Inference.** The OpenAI Agents SDK and PydanticAI carry no risk field of
   their own, so an undeclared axis is inferred from the tool's **name and
   description**, recorded with its own confidence, and never treated as
   authoritative anywhere a hard verdict depends on it.
4. **Unknown.** A tool no rule matches stays unclassified rather than guessed,
   resolving to a conservative `False` that is explicitly *not* a claim the
   tool is safe.

Name inference is wrong in both directions, and neither is hypothetical:

- `find_user_id_by_email` in tau-bench is a pure lookup, and is read as
  state-changing because of the tokens in its name.
- Persistence names such as `write`, `save_draft`, `store_result`,
  `persist_state`, and `append_log` are inferred state-changing but not
  destructive. They receive the state-changing fault family, while coverage
  remains `UNKNOWN` because inference is not authoritative.
- Generic dispatch names such as `bash`, `execute_python`, and
  `execute_command` remain read-only (`UNKNOWN`, not a confirmed `False`)
  because their names do not reveal their effects. They receive no fault
  family unless declared.

Declaring one axis does not upgrade the other's authority: declaring only
`destructive` on a tool leaves `state_changing` exactly as inferred (or
unknown) as it was. Where a declaration disagrees with inference or framework
metadata, the declaration wins and the disagreement is recorded rather than
silently discarded — see `spec.tool_risk` and
`agentcheck/inspect/risk_authority.py`.

A tool whose risk is inferred or unknown is reported that way rather than
hidden: behavioral coverage marks it `unknown` with the reason
`risk_metadata_not_authoritative`. An absent fault family is not evidence that
a tool is safe to retry.

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

For the retry rule, an earlier matching call with no recorded outcome cannot
establish that a later call was safe to repeat: the assertion is `INCONCLUSIVE`
and names the missing evidence. An observed ambiguous timeout followed by a
matching retry remains a violation even if the retry's own result is missing.
Missing unrelated or later outcomes do not by themselves make this rule
inconclusive. This tightens evaluation of incomplete public run records; it
does not rewrite artifacts, require optional actions or change call identity.

## Representative arguments and authored requests

The fixture pack's `tools.<name>.arguments` supplies representative input
values. If `user_request` replaces the generated request, those samples do not
prove which exact argument values that independent prose requires. A valid call
can use defaults or an alternative representation without contradicting the
tool's schema. AgentCheck does not infer that equivalence from tool names,
parameter names, or natural-language instructions.

Newly generated positive, fault and confirmed-action cases keep the argument
comparison and its observed/expected values, but give that sample comparison
separate, non-authoritative provenance. A mismatch is `INCONCLUSIVE`, not a
hard `FAIL` and not `PASS`. A wrong identifier is not certified either. Schema
violations, declared ordering and confirmation rules, duplicate actions, and
known timeout retries retain their independent authority. Explicitly authored
exact argument contracts and generated requests that enumerate exact values
are unchanged.

This changes generation semantics: generator compatibility version is **2**.
Existing frozen suites retain their recorded assertions and fingerprints; they
are not rewritten or silently reinterpreted. Generate and review a new suite to
use the corrected provenance. Do not create a trusted baseline from known
misclassified results. This change does not drop confirmation policy from
schema-boundary cases or invent missing supporting-tool fixtures.

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
