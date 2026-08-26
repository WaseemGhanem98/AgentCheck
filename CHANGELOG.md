# Changelog

Notable changes per release. Dates are release dates.

Two version numbers matter in this project and they move independently:

- the **release version** below, which identifies the distribution, and
- `agentcheck.generate.GENERATOR_COMPATIBILITY_VERSION`, which identifies
  generation *semantics* and is what suite identity is built on.

A release that does not change generation semantics leaves every suite
fingerprint where it was. That is stated for each release under **Suite
identity**.

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
