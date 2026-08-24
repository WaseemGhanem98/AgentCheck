# Declared behavioral coverage

AgentCheck reports which behavioral obligations a suite structurally exercises
and which obligations remain missing. The report answers a bounded question:

> Given this inspected specification and these scenarios, which explicitly
> represented behaviors have both a stimulus and an observable check?

It does not guess business meaning from tool names or prose. A tool named
`delete_user` is not treated as destructive merely because of its name.
Destructive, state-changing, confirmation, and prerequisite coverage is only
reported when the relevant semantics are authoritative in the inspected
contract or explicitly represented by a scenario or selected policy contract.
Unknown metadata stays unknown.

## Reading the statuses

Within each behavioral family, AgentCheck creates one requirement for each
declared tool or explicitly represented tool relationship. Requirements are
not created per scenario or execution, and multiple scenarios for the same
family and subject contribute evidence to the same requirement.

- `COVERED`: the suite contains the explicit stimulus and a supported,
  non-vacuous oracle or trajectory constraint needed to check it.
- `PARTIAL`: the behavior is represented, but the check is optional,
  potentially vacuous, lacks enough fixture outcomes, or otherwise proves less
  than full structural coverage.
- `MISSING`: the behavior is applicable from a declared contract, but no
  adequate scenario represents it.
- `UNKNOWN`: AgentCheck lacks authoritative metadata needed to decide whether
  the risk is applicable.
- `UNSUPPORTED`: the contract expresses a behavior that the current evaluator
  cannot observe reliably.

The denominator for a family is its applicable requirements:
`COVERED + PARTIAL + MISSING`. Unknown and unsupported requirements are shown
but excluded from that denominator. Empty families are omitted; a family that
contains only unknown or unsupported requirements shows zero applicable and is
never presented as fully covered. AgentCheck deliberately does not combine
unrelated families into a marketing-style overall percentage.

For example, a simulated timeout fixture alone shows that the suite can deliver
a timeout, but it does not show what response is required. That is partial
coverage until a supported output or trajectory oracle makes the expected
behavior observable. A no-duplicate constraint with only one response-capable fixture is
also partial: a second call would exhaust the fixture rather than prove the
duplicate-action behavior. Retry coverage similarly accounts for the number of
reachable configured outcomes, fixture priority and ambiguity, declared input
schemas, and scenario resource budgets.

Prerequisite relationships are not inferred from tool or fixture names in
arbitrary custom scenarios. The generator convention is recognized only with a
known generated-action source tag, exactly one explicit `tool:<name>` focal
tag consistent with the declared focal behavior, and a prerequisite fixture
from that convention. AgentCheck can report controlled prerequisite and focal
calls, but its current oracle does not prove their order. Prerequisite success
therefore remains partial, and the corresponding ordering requirement is
`UNSUPPORTED`.

## Structural coverage versus observed execution

Declared behavioral coverage describes suite design. It does not claim that a
model actually took a branch during a run. The run verdict, trace, assertion
evidence, and the separate “Action paths exercised” output describe what the
agent actually did.

This distinction matters for optional calls. A scenario can contain a tool
fixture and pass a constraint vacuously because the agent never called the tool.
Such a scenario does not become full structural coverage merely because its run
verdict was `PASS`, and structural coverage does not rewrite any
`PASS`/`FAIL`/`INCONCLUSIVE`/`INFRA_ERROR` verdict.

Infrastructure failures are also separate. A worker wall-clock timeout is an
`INFRA_ERROR`; only an explicit simulated tool-timeout outcome participates in
tool-timeout coverage.

## Denominator and source binding

Coverage is derived deterministically from:

- the validated `AgentSpec`, including the authority provenance of metadata;
- validated scenario fixtures, fault injections, outcome variants, output
  criteria and trajectory constraints;
- a reference scenario set that defines the requirement universe;
- the actual selected scenario set that supplies evidence; and
- the frozen-suite fingerprint when a frozen suite was used.

For unbounded generation, the actual and reference sets are the same full
lint-clean generated suite. Bounded generation retains the full pre-selection
set as its ephemeral reference while the selected cases supply evidence. A
runtime selection likewise uses the full lint-clean in-memory suite as its
reference. Requirements come from that reference set, while `COVERED` evidence
comes only from the actual set. Excluding a case can therefore make its
behavior missing; it does not silently shrink the denominator. A later run of
a persisted pruned v1 frozen suite cannot reconstruct excluded documents and
is reported as `AVAILABLE_SCENARIOS_ONLY`. Invalid scenarios are not
represented as behavioral coverage.

The document records actual and reference counts and digests. Each digest hashes
a canonically sorted multiset of
`(scenario_id, scenario_fingerprint)` records. It is independent of input
ordering, preserves duplicate multiplicity, and binds both display identity
and behavioral content.

The document fingerprint is an unkeyed checksum over its redacted coverage
content. It can detect stale or corrupted content, but it is not
authentication. Binding checks independently recompute the spec,
actual-scenario, reference-scenario, and frozen-suite identities when those
source documents are available.

Coverage-sensitive tool identifiers or input schemas that cannot survive
mandatory artifact redaction or truncation losslessly are reported
`UNKNOWN`, never `COVERED`. The normalized spec ID is a display/source hint,
not proof of semantic identity; when full source inputs are available,
verification re-derives coverage from those inputs.

For a stored selected run, embedded coverage can retain the digest of the full
reference universe. The loader can verify the coverage checksum and surviving
actual scenario documents, but it cannot independently reconstruct excluded
scenario fingerprints. When an older summary lacks embedded coverage and only
selected scenarios remain, AgentCheck derives
`AVAILABLE_SCENARIOS_ONLY` coverage. That explicitly means the denominator is
limited to surviving documents; the HTML report displays the limitation.

Coverage analysis is pure contract inspection. It does not import or invoke a
target handler, run a gateway, apply fixture effects, or mutate a real or
simulated world. Tool names and descriptions remain identifiers and evidence;
they never classify destructive, state-changing, prerequisite, or authorization
semantics.

## Where the report appears

The machine-readable document is written under
`summary.json.behavioral_coverage` and rendered in the offline HTML report.
The CLI prints the same family counts and uncovered subjects after `generate`
and `test`. Collections are bounded before they reach the artifact boundary;
omitted-item counts remain explicit.

## Compatibility

Behavioral coverage is derived report metadata. It adds no field to `Scenario`,
`FrozenSuite`, replay manifests, baselines, or review records. Existing frozen
suite bytes and fingerprints therefore do not change, and this feature does not
alter replay semantics. Replay remains source-bound re-execution of recorded
inputs and harness behavior; it is not deterministic provider-output replay.

Custom Python agents participate through the same `AgentSpec` and scenario
contracts as the native adapters. Custom tool risk flags are authoritative when
declared by the integration. Framework metadata that an adapter could only
infer from a name or description is retained as unknown for risk-applicability
purposes rather than promoted into a fact.
