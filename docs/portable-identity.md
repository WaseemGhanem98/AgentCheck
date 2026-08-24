# Portable target identity

`spec_id` names the agent under test. It must describe *what* is being tested,
not *where the checkout happens to live*, so that a suite or baseline produced
on a developer machine is still valid in CI.

```
/home/waseem/refund-agent          agentspec-9f2c…
/home/runner/work/refund-agent/…   agentspec-9f2c…   same agent, same identity
```

## What identity is derived from

Semantic inputs — a change here changes `spec_id`:

- the declared agent name;
- the framework and its verified version;
- the declared model identity;
- a digest of the authoritative instructions;
- every declared tool, including its full input schema and its declared
  `state_changing` / `destructive` / `replaceable` flags;
- the handoff graph for a multi-agent target, by structural position;
- the **entrypoint relative to the target root**, canonicalized.

Location metadata — deliberately excluded:

- the absolute checkout path, and therefore the username, the temporary
  directory, the OS root, and the CI workspace root;
- how the same relative path was spelled (`./agent.py`, `agent//agent.py`);
- the host path separator, since the locator is normalized to POSIX form;
- whether the checkout was reached through a symlink.

Two files that share a basename under different relative directories remain
different targets, so portability never collapses distinct agents together.

Evidence keeps the real absolute location for diagnostics. Evidence never
influences identity.

## What is *not* part of identity

Declared policy packs are not read out of the agent, so they are not part of
`spec_id`. They are bound separately: the replay spec binding and the
evaluation baseline each record `policy_pack_ids`, and a run whose declared
packs differ is refused by that binding. Folding them into `spec_id` would
claim the inspected agent had changed when it had not.

## Artifacts created before this contract

Earlier releases hashed the absolute entrypoint path into `spec_id`, so those
artifacts are bound to the directory that produced them.

An inspection records `legacy_spec_id`, the identity the *same* inspection
would have produced under that older contract. A frozen suite, replay manifest
or baseline carrying a legacy identity is accepted only when this inspection
reproduces that value — which can happen only at the original path with an
unchanged behavioral surface. That is precisely the evidence the old contract
required, so nothing is weakened and no equivalence is invented.

Carried to any other directory, a legacy artifact is refused. The refusal
offers the migration path conditionally, because a relocated legacy artifact
and a genuine change to the behavioral contract are indistinguishable from the
recorded identity alone:

> If this artifact predates portable target identity it is bound to the
> directory that created it, and regenerating it from this checkout will
> produce a portable identity.

Regenerating from the current checkout produces a portable identity. Nothing is
rewritten in place and no stored artifact is migrated silently.

## Source integrity is unchanged

Portable identity answers "is this the same agent contract?". It never answers
"is this the same source?". Those bindings are untouched:

- the replay manifest still records `entrypoint_digest` and the bounded
  `file_set`, both content hashes, and still refuses a changed source;
- frozen suites still verify their own fingerprint, so a hand-edited
  `spec_id` invalidates the document;
- declared policy packs are still bound explicitly;
- `INCONCLUSIVE` and `INFRA_ERROR` remain distinct from `PASS`.

Making identity portable widened *where* a trusted artifact may be used. It did
not widen *what* counts as the same source.

## Compatibility

`spec_id` values produced by the evaluation pipeline change with this release,
because the absolute path is no longer an input. `agentcheck.agent_spec.v1`
gains one optional field, `legacy_spec_id`; specs written before it still load.
No other serialized contract gains or loses a field, and no fingerprint
algorithm changes. Suites, baselines, manifests and reports generated from this
release forward are portable; earlier ones behave exactly as they do today, at
the location that produced them.
