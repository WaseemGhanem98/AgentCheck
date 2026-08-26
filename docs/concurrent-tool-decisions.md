# Concurrent tool decisions

A model response can carry several tool calls at once. AgentCheck has always
been able to tell that they were *decided* together. As of this milestone it
can also let the OpenAI Agents and PydanticAI adapters dispatch them as
genuinely concurrent tasks without that concurrency corrupting fixture
assignment, invocation-index bookkeeping, or simulated world state. This
document explains what changed, what still hasn't, and why the boundary sits
where it does.

## Three different things named "concurrency"

1. **Concurrent decision / launch semantics.** Whether two tool calls were
   chosen in the same model response, before either produced a result. A fact
   about the recorded run, derived from event linkage
   (`agentcheck/evaluate/launch.py`).
2. **Dispatch concurrency.** Whether the framework actually runs the
   resulting tool-call coroutines as more than one concurrently-scheduled
   task. Both supported adapters do this via plain asyncio tasks
   (`asyncio.create_task` + `asyncio.gather`/`asyncio.wait`) — never real OS
   threads, confirmed by reading each pinned SDK's own tool-execution code.
   The OpenAI Agents SDK does this once its `max_function_tool_concurrency`
   allows more than one in flight (now 8, up from 1). PydanticAI does this
   **by default**, for every tool call, with no equivalent setting to opt out
   of — that was true before this milestone too, and it was previously
   unaudited.
3. **Completion/interleaving semantics.** Which outcomes actually finish
   first, and what each call's committed effects see. This milestone forces
   this to follow *decision* order, not real completion order, wherever it
   would otherwise matter (see below) — it does not model genuine race
   interleavings beyond that.

These are not the same thing, and conflating them is exactly the mistake this
milestone set out to avoid. (1) and (2) already existed independently before
this work; what changed is that (2) is now something AgentCheck's own
bookkeeping stays correct under, instead of something only made safe by
accident (`max_function_tool_concurrency=1`) or left unaudited (PydanticAI).

## How dispatch concurrency was made safe

`ToolGateway` (`agentcheck/runner/tool_gateway.py`) is now split into a
deterministic **plan** phase and a **commit** phase:

- `plan_batch(calls)` decides everything that must not depend on execution
  order — tool-call and retry budget consumption, invocation-index
  assignment, fixture selection and consumption, the resulting status — for
  every call in a batch, strictly in the order the calls are given. Two
  reservations from one batch can be committed in either order without
  changing any of that; it was already decided.
- `commit(reservation)` applies the fixture's world-state effects and records
  the attempt/outcome for one already-planned reservation.

Both adapters now use a hook that sees a model response's full, ordered list
of tool calls *before* the framework dispatches any of them as a task (the
OpenAI adapter's `on_llm_end`; PydanticAI's `_CapturingModel.request`). That
hook calls `plan_batch` immediately, in the model's own emission order, so
which reservation a call gets never depends on how the framework later
schedules the coroutine that services it.

Because dispatch concurrency here is cooperative asyncio, not real OS
threads, a synchronous method with no internal `await` (like `commit`) always
runs to completion without another task's code interleaving inside it. What
still needed forcing into a fixed order is *which reservation commits before
which* when two calls in one batch touch overlapping simulated state (see
"State conflicts" below) — `agentcheck/runner/launch_barrier.py`'s
`LaunchBarrier` does that with a small `asyncio.Event`-based utility, gating
each reservation's commit until every earlier-ranked one has finished. No
thread locks (there is no real thread contention to lock against), no
sleeps, no timing assumptions.

## What AgentCheck now supports

- **Real concurrent tool-call dispatch** in the OpenAI Agents adapter
  (`max_function_tool_concurrency=8`, a small explicit bound rather than the
  SDK's unbounded default) and in PydanticAI (its own default, now made
  safe rather than merely unaudited).
- **Deterministic fixture/invocation-index assignment** regardless of which
  task's coroutine the event loop happens to run first.
- **Deterministic world-state commit order**: two same-stage calls whose
  fixtures both mutate simulated state commit in decision order, every time —
  proven by running the same scripted batch many times and checking the
  final state is identical, not by inspecting one run.
- **`same_launch_group(a, b)` / `observed_before(a, b)`** — unchanged from
  before this milestone, and still correct under real concurrent dispatch.
- **`ordering`** and **`no_same_stage_duplicate_action`** — both verdicts are
  proven independent of completion order: repeated runs of the same
  same-stage scenario produce the same verdict regardless of which
  concurrently-dispatched call's task happens to finish first.
- **Two same-stage calls racing for one single-use fixture fail closed**: the
  second is refused at plan time (deterministically, by decision order), not
  raced for at commit time.

## What AgentCheck still cannot do

- **Custom Python agents have no launch-group concurrency, sync or async.**
  A custom `start`/`resume` pair may now be a coroutine function pair, given
  an `AsyncToolRuntime` whose `call()` can be `await`-ed (see
  `docs/custom-agents.md`). That closes a *different* gap — an agent whose own
  loop needs a real `await` no longer has to fake a synchronous boundary — but
  it does not add dispatch concurrency: `AsyncToolRuntime.call` has no
  internal `await` of its own, so it always runs to completion before the next
  call begins, even under `asyncio.gather`/`create_task`. Neither `ToolRuntime`
  has a hook analogous to the two adapters' above — a custom agent's own
  orchestration decides when to call it, and if that orchestration reaches a
  call through real OS threads instead of asyncio, fixture and
  invocation-index ordering are not guaranteed. This is deliberately left
  unsupported rather than guessed at: see the docstrings on
  `_GatewayToolRuntime` and `_AsyncGatewayToolRuntime` in
  `agentcheck/adapters/custom.py`. More fundamentally, a custom agent's model
  calls are never observed by AgentCheck at all (no `model_request`/
  `model_response` events), so there is no evidence from which to derive that
  two tool calls were *decided together* — the fact same-stage/launch-group
  semantics are built on for the other two adapters. Inventing that fact for a
  custom agent would be exactly the guessed verdict this project does not
  produce, so it stays unsupported regardless of how the calls are dispatched.
- **Two different destructive tools targeting the same resource**, launched
  together, evaluated for whether that pairing itself is safe. Doing this
  honestly requires a declared resource/entity link between the two tools,
  which no current contract expresses; inferring one from names would be
  exactly the kind of semantic guessing this project's tool-risk work exists
  to stop doing. This stays `UNKNOWN`/unsupported rather than invented.
- **Retry-after-concurrent-mutation**: "action A retried after concurrent
  action B already changed the state A depends on" needs state versioning
  the simulated world does not have. Not built here.
- **Genuine race-condition effects**: a real interleaving where one call
  observes another's *partial* mutation mid-flight. The simulated world has
  no partial states to observe — a commit either has happened or hasn't.

If a scenario or policy needs one of the unsupported cases above, the honest
answer is `INCONCLUSIVE`/generation refusing to invent the case, not a
guessed verdict.
