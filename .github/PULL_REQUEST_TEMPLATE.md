## What this changes

<!-- What and why. Prefer the reason over the diff summary. -->

## Safety invariants

AgentCheck's value is that its verdicts can be trusted, so these are checked on
every PR. Tick what applies, and say so plainly if something does not.

- [ ] No original target tool handler executes during a simulated evaluation.
- [ ] Unknown tools still fail closed — no synthesized tool results.
- [ ] No real mutations; only the simulated world changes.
- [ ] Worker isolation and the network-denied default are intact.
- [ ] `INCONCLUSIVE` / `INFRA_ERROR` are not collapsed into `PASS`.
- [ ] No safety or wall-clock budget was widened to make CI pass.

## Contracts

- [ ] No fingerprint, serialized contract, or stored artifact shape changed.
      *(If one did, describe the compatibility impact — suite fingerprints
      include the generator version, so they are easy to move by accident.)*

## Adapters (skip if untouched)

- [ ] Interception happens **before** the original handler.
- [ ] The framework version gate is unchanged, or the widening is backed by
      evidence that the new version was actually verified.
- [ ] A missing optional extra raises an actionable `AdapterDependencyError`.

## Verification

<!-- Commands you ran and what they reported. Tests must be offline,
     credential-free, and cost nothing. -->
