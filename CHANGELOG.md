# Changelog

Notable changes per release. Dates are release dates.

Two version numbers matter in this project and they move independently:

- the **release version** below, which identifies the distribution, and
- `agentcheck.generate.GENERATOR_COMPATIBILITY_VERSION`, which identifies
  generation *semantics* and is what suite identity is built on.

A release that does not change generation semantics leaves every suite
fingerprint where it was. That is stated for each release under **Suite
identity**.

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
