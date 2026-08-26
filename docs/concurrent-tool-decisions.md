# Concurrent tool decisions

A model response can carry several tool calls at once. AgentCheck can tell
that they were *decided* together; it does not *execute* them concurrently.
This document is about that gap, why it exists, and what each half actually
buys you.

## Three different things named "concurrency"

1. **Concurrent decision / launch semantics.** Whether two tool calls were
   chosen in the same model response, before either produced a result. This is
   a fact about the recorded run, derived from event linkage
   (`agentcheck/evaluate/launch.py`), and AgentCheck has understood it since
   the launch-group work that added `LaunchAnalysis.same_launch_group` and
   `.observed_before`.
2. **Simulated execution scheduling.** The order `ToolGateway` actually
   services calls the adapter hands it. Today this is strictly serial and
   synchronous, regardless of launch grouping: two calls decided together
   still run one after another inside the gateway.
3. **Real wall-clock concurrent execution.** Actually dispatching two tool
   invocations at once (multiple OS threads, an event loop running them
   interleaved, or `max_function_tool_concurrency > 1` in the OpenAI Agents
   SDK). AgentCheck does not do this.

(1) and (2) are already distinct today: a scenario can represent "these two
were decided together" without them ever running out of order. (3) is the one
this milestone deliberately did not add.

## Why (3) is not enabled

`ToolGateway` (`agentcheck/runner/tool_gateway.py`) is a single synchronous
object with mutable, unsynchronized per-call state: a `_used_fixtures` set
that fixture matching depends on, and sequence counters that canonical event
ordering depends on. Two callers entering it concurrently — real threads, or
an event loop actually interleaving coroutines rather than awaiting them one
at a time — would make fixture selection and event sequencing depend on
whichever call happened to reach the gateway first. That is exactly the
scheduling-dependent, non-deterministic harness behaviour AgentCheck exists to
avoid: a suite's outcome would stop being a pure function of the target, the
seed, and the fixtures, and would start depending on OS thread scheduling.

`max_function_tool_concurrency` stays `1` in the OpenAI Agents adapter, and
the custom `ToolRuntime` stays synchronous, for this reason. Raising the
concurrency cap without first making the gateway's internal state safe under
concurrent access would not add a real capability — it would add flakiness.
Making the gateway safe under real concurrent access (locking, or per-launch-
group isolated fixture pools) is future work, not something this milestone
does quietly under a version bump.

## What AgentCheck can test today

Everything below is deterministic: no thread scheduling, no sleeps, no
random interleaving, bounded generation.

- **`same_launch_group(a, b)`** — were two attempts decided in the same model
  response.
- **`observed_before(a, b)`** — was `a`'s result recorded before the model
  response that launched `b`. `False` for same-stage attempts (neither could
  have informed the other); `None` when the run does not record enough to
  say — never treated as evidence of either answer.
- **`ordering`** (a declared policy or generated constraint) — fails when a
  same-stage pair violates a declared "observe X before deciding Y" rule.
  A prerequisite and its dependent action launched together is exactly this
  case, and it already fails correctly: see
  `test_ordering_fails_when_both_calls_were_decided_together`.
- **`no_same_stage_duplicate_action`** — fails when the *same* tool, with
  identical arguments, is decided twice in one launch group (`delete_user()`
  called twice in one response). Structurally unambiguous, unlike a repeat
  across two later reasoning turns (which `no_duplicate_side_effect` already
  covers, and which this rule does not fire on).
- **Unrelated same-stage calls are not, by themselves, a finding.** Two reads
  decided together, or one destructive tool decided alongside an unrelated
  read, produce no failure unless a declared/authoritative rule actually
  says something about the relationship. Sharing a launch group is evidence,
  not a verdict.

## What AgentCheck cannot test yet

- **Real concurrent execution effects**: a genuine race between two
  in-flight calls, one mutating state the other reads mid-flight. The
  simulated world only ever sees calls one at a time.
- **Two different destructive tools targeting the same resource**, launched
  together, evaluated for whether that pairing itself is safe. Doing this
  honestly requires a declared resource/entity link between the two tools,
  which no current contract expresses; inferring one from names would be
  exactly the kind of semantic guessing this milestone's tool-risk work was
  written to stop doing. This stays `UNKNOWN`/unsupported rather than
  invented.
- **Retry-after-concurrent-mutation**: "action A retried after concurrent
  action B already changed the state A depends on" needs state versioning
  the simulated world does not yet have. Not built here.
- **Async custom-agent tool calls.** `ToolRuntime` is synchronous by
  contract; a custom agent that wants to issue tool calls from concurrent
  tasks has no supported path today. This is documented as unsupported
  rather than fabricated, per the same reasoning as (3) above — the
  underlying gateway is not yet safe for it.

If a scenario or policy needs one of the unsupported cases above, the honest
answer is `INCONCLUSIVE`/generation refusing to invent the case, not a
guessed verdict.
