# Changelog

Notable changes per release. Dates are release dates.

Two version numbers matter in this project and they move independently:

- the **release version** below, which identifies the distribution, and
- `agentcheck.generate.GENERATOR_COMPATIBILITY_VERSION`, which identifies
  generation *semantics* and is what suite identity is built on.

A release that does not change generation semantics leaves every suite
fingerprint where it was. That is stated for each release under **Suite
identity**.

## 0.4.1 (2026-08-28)

One fix, found validating an independent PydanticAI target
(`slack-samples/bolt-python-starter-agent`) with a dedicated,
framework-isolated `--python` interpreter -- exactly the setup the CLI's
own `--python` help text recommends.

**Worker-python probe required both framework SDKs regardless of
adapter.** `_probe_worker_python()` unconditionally imported `agents`
(the OpenAI Agents SDK) in the candidate worker interpreter, so a
`pydantic_ai`- or `custom`-adapter target using an isolated interpreter
that only had `pydantic-ai` installed (or neither framework, for
`custom`) failed closed with a misleading "could not import AgentCheck
and its runtime dependencies (pydantic, openai-agents)" error -- even
though that adapter needs no such package. This directly contradicted the
project's own separate-extras guarantee ("evaluating one framework never
installs the other"). Fixed: the probe now imports only the framework
package the *configured* adapter actually needs.

**Suite identity:** unaffected. `GENERATOR_COMPATIBILITY_VERSION` stays
`1`; this fix changes worker-interpreter diagnostics only, not inspection,
generation, or evaluation semantics. No target's `spec_id` moves.

## 0.4.0 (2026-08-27)

One new capability, four fixes found by hands-on reliability sweeps and real
third-party target testing, and one audit that confirmed an existing boundary
rather than adding one.

**Custom agents: async `start()`/`resume()` support.** Custom agents
previously rejected any coroutine `start()`/`resume()` outright at preflight,
even one that only needed a real `await` for its own provider client and
never touched concurrent dispatch. `start()`/`resume()` may now both be
coroutine functions, given an `AsyncToolRuntime` whose `call()` is awaited
instead of called directly; the two methods must still agree on shape
(mixing sync and async is refused as `mismatched_turn_method_concurrency`).
This does **not** add concurrent tool-call dispatch or same-stage/launch-group
semantics for custom agents: `AsyncToolRuntime.call` has no internal await of
its own, so even `asyncio.gather`/`create_task`-dispatched calls still
execute strictly in scheduling order (verified by running the same
gather-based scenario 25 times with the order never varying). A custom
agent's model calls are never observed by AgentCheck, so there is no evidence
from which to infer that two tool calls were decided together — see
`docs/concurrent-tool-decisions.md` and `docs/custom-agents.md` for the exact
boundary.

### Fixed

- **A misspelled `agentcheck.json` `tool_risk` key was a silent no-op.**
  Nothing cross-checked that every key in a declared `tool_risk` block
  actually named a real tool, so a typo (`"cancle_order"` instead of
  `"cancel_order"`) silently fell back to inferred risk with no error and no
  warning. A misspelled or otherwise unmatched key now refuses the run before
  any scenario executes (`unknown_tool_risk_declaration`), naming the exact
  key.
- **`init --force` silently wiped fields it did not control**, most notably
  `environment_allowlist` (a security-relevant setting with no CLI flag of
  its own to restore it), by rebuilding `agentcheck.json` from scratch on
  every re-run. `--force` now reads the existing file first and merges
  `adapter`/`entrypoint` on top of it.
- **PydanticAI controlled-model evaluation exhausted its retry budget for any
  non-object `output_type`** (`bool`, `int`, ...), from three compounding
  bugs: schema derivation only handled `BaseModel` output types, the
  target's own `retries=` setting was never propagated, and the controlled
  model's reply was built from a schema computed once at construction time
  rather than the framework's actual per-request negotiated contract
  (PydanticAI wraps a scalar `output_type` in a single-key object before
  accepting a reply for it). All three fixed; reproduced against the SDK's
  own `roulette_wheel.py` example before and after.
- **`openai-agents` and `pydantic-ai-slim` version pins widened to a verified
  range** (`openai-agents>=0.20,<0.23`, `pydantic-ai-slim>=2.32,<2.36`) after
  each pin was found stale against the frameworks' actual current PyPI
  releases during real-world example testing. Both adapters' full test
  suites were run against the floor and ceiling of their new ranges in
  separate scratch virtualenvs with identical private-attribute surfaces
  confirmed at each end, not just the pin bumped.
- **A non-standard virtualenv name broke replay-manifest source inventory.**
  A target-local virtualenv under any name other than the two hardcoded ones
  (e.g. `.test-venv`) could survive the fixed-name exclusion list and then
  hit the dot-prefix path-safety refusal as a hard failure. Virtualenvs are
  now detected by their `pyvenv.cfg` marker instead of a fixed name list; a
  dot-prefixed directory that is not a genuine virtualenv still fails closed
  exactly as before.
- **Wrong-SDK-type targets now get a "did you mean" suggestion** naming the
  likely correct adapter, which also surfaced and fixed an independent bug:
  `OpenAIAgentsAdapter.inspect()` raised a bare `TypeError` for a wrong-type
  target, discarding the detail before any suggestion could reach a user.
- **Committed `.claude/` project-tooling metadata entered the source
  inventory**, triggering the safe-relative-path refusal on targets that
  ship it; it now joins the established tooling-exclusion list, while
  ambiguous `.hooks/` remains fail-closed by design.
- **Persistence-shaped tool names (`write`, `save`, `store`, `persist`,
  `append`) fell into the weakest read-only/unknown risk-inference bucket.**
  They now infer state-changing (not automatically destructive); generic
  dispatch names such as `bash` are deliberately left unknown by name alone,
  since a bare dispatch name does not itself reveal an effect. **Breaking for
  any target with a matching, undeclared tool name:** the resolved
  `state_changing` boolean is hashed into `spec_id` (`ToolDefinition` is
  hashed whole), so a target whose tools include one of these names with no
  explicit `tool_risk` declaration gets a new `spec_id`, and any frozen suite
  generated before this fix is rejected as stale until regenerated — the same
  category of break as 0.2.1's risk-marker fix. A target that declares
  `tool_risk` explicitly for that tool, or has no matching tool name at all,
  is unaffected.
- **Documentation drift**: `README.md` and `docs/fault-testing.md` still
  described persistence-shaped names as read-only after the fix above;
  corrected to match the actual inference rules.

### Audited, no defect found

- **PydanticAI multi-agent delegation**: confirmed no `handoff`/`delegate`/
  `as_tool` method exists anywhere in the pinned SDK's `Agent` surface
  (unlike the OpenAI Agents SDK's distinct `Handoff` type). This is a
  documentation and audit clarification only — it does not add multi-agent
  support to the PydanticAI adapter, which remains out of scope.

**Suite identity:** `GENERATOR_COMPATIBILITY_VERSION` stays `1`;
`agentcheck/domain/` and `agentcheck/generate/` are unchanged since 0.3.0, so
no scenario or suite fingerprint moves as a direct result of this release.
**One fix does move `spec_id`, narrowly:** the persistence-verb risk-inference
fix above changes the resolved `state_changing` value — and therefore
`ToolDefinition`, which is hashed whole into `spec_id` — for any target with
an undeclared tool named `write`/`save`/`store`/`persist`/`append` (or a
compound name containing one of those tokens). Such a target's existing
frozen suite is correctly rejected as stale until regenerated; every other
target, and every other fix in this release, leaves `inspect()`/`spec_id`
unaffected — each corrects run-time behavior (preflight refusal wording,
controlled-model reply construction, source inventory) rather than spec
derivation.

## 0.3.0 (2026-08-26)

Three accumulated milestones since 0.2.1: developer-declared tool risk with
an explicit authority precedence, deterministic concurrent simulated tool
execution for both supported frameworks, and safe PydanticAI dependency
injection. A pre-release audit added a cross-milestone integration test
(declared risk + concurrent dispatch + dependency injection composed on one
target) and a further real-world validation pass found no release-blocking
defects, including a real third-party dependency client (an async video-API
SDK) and two previously-unexercised OpenAI Agents patterns (dynamic tool
approval callbacks, handoff input filters) that were confirmed already
correctly refused rather than silently allowed.

**Suite identity:** `GENERATOR_COMPATIBILITY_VERSION` stays `1` across this
entire release. No default-generation behavior changed for a target that
declares no risk and uses no new opt-in policy pack; where a target
deliberately opts into new behavior (`derived_tool_risk_v1` against a
destructive tool), only that target's own suite identity moves, as intended.

**PydanticAI dependency injection / `RunContext[Deps]`:** previously rejected
outright at preflight. Real-world validation found this blocked most
realistic PydanticAI examples, including both official ones AgentCheck's own
prior campaign had tried. Now supported: AgentCheck never constructs a real
dependency object (`deps_type` is a static type annotation the framework
never instantiates or runtime-validates -- confirmed against the pinned SDK),
and always passes a fail-closed `_InertDependencies` placeholder as `deps=`
when it drives a run. The reconstructed tool invoker never reads `ctx.deps`;
if anything ever did, attribute access on the placeholder raises immediately
rather than returning fabricated data or reaching a live object. `ctx` itself
was already excluded from the tool's declared JSON Schema by PydanticAI's own
schema generation, so no schema-handling code needed to change. A newly
identified, narrower gap -- a callable `Agent(validation_context=...)`, which
the framework itself invokes with a `RunContext` -- gets its own new
preflight rejection; a non-callable value is accepted.

Validated against 8 real public PydanticAI examples (pydantic/pydantic-ai
`41e503c91ee33a53c4d9529bd1a6425fff69dfca`): `weather_agent.py`,
`data_analyst.py`, `roulette_wheel.py`, and `rag.py` now inspect, prepare, and
run correctly end to end (including a cross-turn prerequisite chain and
same-stage concurrent duplicate calls); `bank_support.py`, `sql_gen.py`, and
`flight_booking.py` are still correctly refused, for unrelated, pre-existing
reasons (dynamic instructions / output validators reading dependencies, not
dependency injection itself); `medical_agent_delegation.py`'s multi-agent
delegation stays out of scope, as this project's PydanticAI adapter does not
support multi-agent topology at all. No milestone-related defects found; zero
provider calls, zero original handler executions across every target.

**Suite identity:** unaffected. `inspect()` (and therefore `spec_id`) is
unchanged by this milestone; only `preflight()` and how `agent.run()` is
invoked changed. `GENERATOR_COMPATIBILITY_VERSION` stays `1`.

**Concurrent execution safety:** `ToolGateway` is split into a deterministic
plan phase (`plan_batch`/`plan_one` -- decides budget consumption, invocation
index, fixture selection and consumption, and resulting status, strictly in
the order calls are given) and a commit phase (`commit` -- applies world-state
effects and records the attempt/outcome for one already-planned reservation).
A new `agentcheck.runner.launch_barrier.LaunchBarrier` forces a batch's
commits into decision order under real concurrent dispatch.

The OpenAI Agents adapter now dispatches a model response's tool calls as
genuinely concurrent asyncio tasks (`max_function_tool_concurrency` moves
from 1 to a bounded 8), planning the batch deterministically at `on_llm_end`
before any task is created. The PydanticAI adapter -- which already dispatches
concurrently by default, unaudited until now -- gets the same treatment via
`_CapturingModel.request`. Both are proven, by running the same scripted
same-stage batch many times, to produce identical fixture assignment and
world-state commit order regardless of which task the event loop happens to
run first. Custom Python agents remain synchronous and explicitly documented
as unsupported for concurrent dispatch.

Fixed a real, pre-existing defect surfaced while adding the first PydanticAI
test to exercise state effects: `ToolOutcome.state_transition_ids` stored raw
gateway transition IDs instead of the canonical run-scoped IDs `CanonicalRun`
validation expects, so any PydanticAI run that actually produced a state
transition failed to construct. Also fixed a regression introduced during
this milestone's own development: `ToolGateway.commit` had stopped chaining
`BudgetExceeded` as the `__cause__` of its raised `ToolCallBlockedError`,
which the custom adapter's termination mapping depends on to report
`MAX_TOOL_CALLS`/`WALL_CLOCK_TIMEOUT` instead of a generic `ADAPTER_ERROR`.

Real-world validation against public OpenAI Agents SDK and PydanticAI
examples plus tau-bench's retail tool schemas found no concurrency-related
defects: fixture assignment and launch-group evidence matched decision order
in every target, and no original tool handler executed.

**Suite identity:** unaffected. This milestone changes execution/dispatch
behaviour, not generation semantics -- `GENERATOR_COMPATIBILITY_VERSION`
stays `1`, and no scenario or suite fingerprint moves.

**Developer-declared risk:** `agentcheck.json` gains an optional `tool_risk`
block declaring `state_changing`/`destructive` per tool, independently per
axis. Precedence is fixed: developer declaration > genuine framework metadata
(no current adapter exposes this) > inference > unknown. A declared axis
always wins; an undeclared axis falls through to inference on its own,
without upgrading its authority. Every tool's resolved risk now carries its
provenance in `spec.tool_risk` (`ToolRiskAssertion`/`RiskAxis`), separate from
`ToolDefinition` so provenance does not move `spec_id` for a target that
declares nothing.

Fixed two related defects found while wiring this in: the PydanticAI and
custom-agent `prepare()` methods rebuilt each tool's runtime risk from
`spec.capabilities` — a second, independent name-based classifier — instead
of trusting the already-resolved `ToolDefinition`. This silently discarded a
developer's declaration at the exact point the runtime invoker enforces risk,
for any tool whose name carried no lexical signal of its own. Both now read
the resolved value directly.

**Concurrency:** a new `no_same_stage_duplicate_action` rule/oracle,
building on the existing launch-group/observed-before analysis, flags the
same tool called twice with identical arguments in one model response — a
structurally unambiguous case, unlike a cross-turn retry, which
`no_duplicate_side_effect` already covers and this rule does not fire on.
`max_function_tool_concurrency` stays `1` and `ToolGateway` stays
synchronous: see `docs/concurrent-tool-decisions.md` for why real concurrent
execution is not yet safe to enable, and what is and is not tested today.

**Suite identity:** `GENERATOR_COMPATIBILITY_VERSION` stays `1`. Default
generation for a target that declares no risk and uses no new policy pack is
byte-identical to before. `derive_tool_risk_pack`'s own version
(`DERIVED_TOOL_RISK_VERSION`) moves `1` -> `2`: opt-in and additive, so a
target that does not list `derived_tool_risk_v1`, or declares no destructive
tool, is unaffected; one that does now asserts something new for its
destructive tools, so a freshly generated suite earns its own identity. A
suite already frozen under version `1` keeps validating as recorded.

## 0.2.1

Six defects found by running 0.2.0 against public third-party agents. Five were
silent: the suite generated, ran, and passed while asserting nothing.

**Suite identity:** `GENERATOR_COMPATIBILITY_VERSION` stays `1`. Two suites that
were equivalent before remain byte-identical after, which is the documented
condition for leaving it alone. Suites whose *content* genuinely changed — a spec
with a zero-argument state-changing tool, or one large enough to have been
truncated — move their own `suite_id`, as they should.

**Breaking for PydanticAI targets:** the risk-marker fix changes the declared
tool surface, and `spec_id` hashes that surface. A PydanticAI target now inspects
under a different `spec_id`, so an existing frozen suite is rejected with
"re-run agentcheck generate". Regenerating is the right action: the previous
suite contained no fault cases at all. OpenAI Agents targets keep their
`spec_id`; verified by measurement.

### Fixed

- **Zero-argument tools received no fault family.** A schema declaring no
  parameters has one in-contract call — the empty one — but that was read as
  "no valid call can be constructed", so arity rather than declared risk decided
  which irreversible actions were tested. In the official OpenAI customer-support
  sample, `cancel_trip()` produced one scenario carrying no fixtures, no
  trajectory constraints and no output criteria; it could not fail. That target
  goes from 30 cases to 44, and `cancel_trip` from 1 to 8 including the
  ambiguous-timeout duplicate-side-effect case.
- **A run that attempted no tool was scored FAIL.** `ControlledModel` never
  chooses a tool by design, so every zero-argument tool on every target produced
  a guaranteed failure under the documented offline configuration, and `gate`
  returned exit 1. Confirmed by controlled experiment: same suite, same tool,
  FAIL under `ControlledModel` and PASS under a scripted model that calls it.
  Such a run is now `INCONCLUSIVE`. A call with the wrong arguments, a call to
  another tool, or too many calls all remain `FAIL`. `gate` still refuses the
  run, at exit 3 instead of exit 1.
- **PydanticAI tools carried no risk markers.** The adapter hardcoded
  `state_changing=False, destructive=False` and never called the shared
  classifier the OpenAI adapter uses, so no PydanticAI target received any fault
  family for any tool. It surfaced as a single `inspect` report contradicting
  itself — a summary of zero state-changing actions above a capability listing
  describing the same tool as destructive. Measured on a PydanticAI target: 13
  cases and 0 fault cases before, 25 and 11 after.
- **The fault family was bounded by the wrong cap.** Generation broke on
  `_MAX_POSITIVE_SCENARIOS_PER_SPEC` (24), so the documented
  `MAX_FAULT_VARIANT_SCENARIOS` (64) was unreachable. On tau-bench,
  `modify_user_address` lost its entire fault family while 38 scenarios of the
  real cap were still unused: 90 cases and 5 fault-covered tools before, 96 and
  6 after.
- **Declared output-rule parameters were discarded.** `_apply_output_rule`
  hardcoded `"parameters": {}`, so a policy declaring `success_terms` attached
  without error and did nothing. Since the evaluator gates the hard
  fabricated-success verdict on exactly that parameter, 0.2.0's
  degraded-evidence capability had no reachable configuration on a generated
  suite. With the fix and a declared pack, an agent answering "Refund completed"
  after an error moves from `INCONCLUSIVE` to `FAIL`; without a pack it stays
  `INCONCLUSIVE`, because the authority is the developer's declaration.

### Added

- Handoff expectations are declarable. The evaluator already implemented five
  deterministic handoff checks over recorded `HANDOFF` events, and the
  generation lint already listed them as supported, but no policy rule kind
  produced one — the only route was hand-written suite JSON. `required_handoff`,
  `forbidden_handoff`, `max_handoffs`, `no_handoff_loop` and
  `handoff_before_tool` now reach those evaluators. On the official
  `customer_service` example, a pack that was previously rejected at parse time
  attaches 26 constraints.

### Documentation

- Corrected the claim that fault generation follows a tool's *declared* risk.
  It does for custom Python agents, which state it. For the OpenAI Agents SDK
  and PydanticAI it is **inferred from the tool name**, reported as inferred and
  never treated as authoritative. `docs/fault-testing.md` asserted names were
  not trusted while citing, as justification, a real case where a name was
  trusted and read wrongly. Both now say which adapter establishes risk how, and
  that an unclassified tool receives no fault family — so an absent fault family
  is not evidence a tool is safe.

## 0.2.0

**Suite identity:** unchanged. `GENERATOR_COMPATIBILITY_VERSION` stays `1`, so
suites frozen under 0.1.x keep their `suite_id` and scenario fingerprints and
load without regeneration.

### Added

- `agentcheck gate <target>` — a single CI entry point that answers whether a
  change should block the build. It reports separate exit codes for a
  behavioural failure (1), a target that is not certifiable (2), and an
  inconclusive run (3), so a broken harness cannot be read as a clean pass.
  See `docs/ci-gate.md`.
- Degraded-evidence fault cases. Suites now generate scenarios for `empty`,
  `malformed`, `partial`, and `stale` tool results, bounded by
  `MAX_FAULT_VARIANT_SCENARIOS = 64`. See `docs/fault-testing.md`.
- Policy rule kinds `ordering`, `max_retries`, `max_tool_calls`, and
  `max_model_turns`, each mapping onto a trajectory constraint the evaluator
  already implements. See `docs/behavioral-policies.md`.

### Changed

- `no_fabricated_success` now treats degraded tool results as unreliable
  evidence, not just hard failures. An agent that receives a truncated or stale
  payload and reports clean success is a FAIL where it previously passed.
  `acknowledges_tool_error` is deliberately unchanged: degraded evidence is not
  a tool error.
- Suite provenance records generation semantics rather than the release
  version. Publishing a new version no longer re-identifies otherwise-identical
  suites.
- The missing-entrypoint error names the file it looked for and the command
  that fixes it.

### Fixed

- Policy rule parameters were discarded when constraints were built, so
  `max_retries: 2` and `max_retries: 5` were indistinguishable and collapsed
  into one constraint.
- `ordering` constraints were dropped by a stale lint allowlist before reaching
  the evaluator, so a declared ordering rule silently did not run.

### Compatibility

No change to the interception boundary, fail-closed behaviour, isolation, the
network policy, source binding, or verdict semantics. The new degraded-evidence
FAIL is the one behavioural scoring change: a suite regenerated under 0.2.0 can
report failures that 0.1.x did not surface. Existing frozen suites are
unaffected until regenerated.

## 0.1.1 and earlier

Released before this changelog was kept. See the git history.
