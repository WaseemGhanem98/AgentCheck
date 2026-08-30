# Environment containment research spike

This repository-only spike asks two independent questions before AgentCheck
changes any production containment architecture:

1. Can a maintained environment boundary execute a frozen hostile Python
   target while blocking and independently observing every relevant attempted
   external effect, with zero real mutation?
2. Can the same boundary still exercise meaningful agent action paths and
   preserve enough semantic evidence for policy evaluation?

Containment alone is not success. A perfectly isolated run that exercises no
meaningful action path is a failed product experiment.

## Status and non-claims

Milestone 1 freezes measurement instruments only. It does not implement a
production sandbox, add an `EnvironmentProvider` interface, change the current
worker, or demonstrate that arbitrary Python is safe. The current AgentCheck
worker remains trusted-code, same-user process isolation and must not execute
`targets/adversarial/adversarial_target.py`.

The spike makes no paid provider calls, uses no real credentials or services,
and adds no CLI command. Existing `inspect`, `generate`, `test`, `replay`, and
other command behavior is unchanged. Static inspection remains useful and is
always reported as lower-bounded (`exhaustive: false`).

## Frozen layout and trust boundary

```text
environment_containment/
├── README.md                         host-owned experiment contract
├── catalog.json                      frozen hostile/control denominator
├── measurement.py                   host-side measurement contract
├── semantic_ground_truth.json        frozen behavioral oracle
└── targets/
    ├── adversarial/
    │   └── adversarial_target.py      explicitly armed hostile code only
    └── behavioral/
        ├── agentcheck.json
        └── behavioral_target.py       deterministic offline custom agent
```

Only a target leaf may enter a future sandbox. Catalog, ground truth,
measurement code, sentinels, result writers, provider configuration, and
project-control metadata stay outside it. Neither leaf contains `.agents`
metadata or an `__init__.py`. The adversarial leaf deliberately has no
`agentcheck.json` or exported `agent`, preventing accidental execution through
today's ordinary AgentCheck workflow.

The hostile source is inert on import and requires both an explicit attack ID
and an arming value. Those friction controls prevent accidents; they are not a
security boundary. Contract tests parse the file without importing or running
it. A future provider must use a committed, digest-pinned copy mounted or baked
read-only, plus private writable scratch.

## Evidence dimensions

Every result keeps these dimensions separate:

- **Contained:** could the execution affect the real world?
- **Observed:** did an independent trusted channel see the attempted action?
- **Semantically understood:** does frozen authoritative ground truth explain
  what the action means?
- **Policy-verifiable:** is the trusted execution and semantic evidence enough
  to decide the policy?

An observed, blocked action with unknown meaning remains policy
`INCONCLUSIVE`. No aggregate score combines containment and behavioral
evidence.

## Adversarial catalog

`catalog.json` freezes 27 hostile cases and two non-escape controls before any
provider implementation. It covers filesystem/data, process/IPC, network,
credential/provider, control-plane, and resource-pressure surfaces. Some cases
are explicit bounded matrices of closely related sub-attempts; every listed
sub-attempt must execute and be observed for that case to count as executed.

The target receives only synthetic, harness-created paths, endpoints, process
IDs, and credential canaries. It contains no real service hostname, credential,
or host-specific path. Resource cases also contain target-side hard bounds so a
provider failure does not create an unbounded workstation or runner load.

The catalog is provider-neutral. Provider code may map symbolic inputs and
observation requirements to a maintained runtime, but may not delete, rewrite,
split, merge, or silently skip cases to improve the denominator. An unsupported
mandatory case remains explicitly unexecuted and prevents a favorable complete
containment conclusion.

### Threat-model inputs and limits

The bounded catalog is informed by primary, maintained security references:

- [NIST SP 800-190, Application Container Security Guide](https://csrc.nist.gov/pubs/sp/800/190/final),
  including container runtime, host OS, image, registry, orchestration, and
  countermeasure concerns;
- the [MITRE ATT&CK Containers matrix](https://attack.mitre.org/matrices/enterprise/containers/),
  including execution, persistence, escape-to-host, credential access,
  discovery, and resource-impact techniques; and
- the [gVisor security model](https://gvisor.dev/docs/architecture_guide/security/),
  especially its System API/ABI, side-channel, host-backed resource, network
  policy, and resource-controller boundaries.

These sources guide category selection; they do not make 27 cases exhaustive
and do not prove a provider secure. The catalog intentionally excludes exploit
payloads for specific kernel/runtime vulnerabilities. It measures bounded
security properties with synthetic canaries and controlled endpoints.

Each catalog record carries the frozen definition and empty result slots:

- `attack_id`, `category`, `technique`, `security_property_under_test`
- `expected_outcome`, `observability_requirement`, `mutation_sentinel`
- `actual_outcome`, `evidence_references`, `limitations`

`actual_outcome` starts `null`; `evidence_references` starts empty. Milestone 1
therefore contains no claimed attack result.

## Trusted observation

Target stdout/stderr is bounded, untrusted diagnostic material. It cannot prove
that an attempt happened, that a block came from the provider, or that a
mutation did not occur. A future run needs independent evidence appropriate to
the case, such as:

- parent-created canary state outside target mounts;
- provider/runtime trace records configured before target start;
- a trusted controlled-endpoint counter;
- provider/network-policy postconditions proving uncontrolled egress stayed
  unavailable;
- parent-observed process, container, mount, network, and teardown state.

Observer setup failure, dropped trace events, missing sentinels, or cleanup
failure makes evidence incomplete. Target-authored evidence never upgrades a
case to observed.

The catalog freezes every target-side subattempt ID and its permitted trusted
observer kinds. A case counts as executed and observed only when every frozen
subattempt has independent evidence. Absence evidence is separate: unchanged
canaries, zero-hit endpoints, source bindings, and teardown postconditions
cannot by themselves prove that an attempt occurred. Unchecked absence remains
`null`, not a favorable zero.

Each relevant effect also has its own frozen expected value and trusted
evidence kinds. Attempt evidence, the overall postcondition, and effect evidence
are role-bound; a trace placed in the wrong field cannot satisfy another role.
Resource cases additionally require a typed provider-limit-enforced fact from
the independent resource controller. Merely reaching the fixture's safety cap
is inconclusive.

Environment identity is also role-bound. A favorable ordinary case has exactly
one `primary_attempt` phase: every attempt, postcondition, effect, and typed
phase-completion receipt must bind to that same run, trial, and environment
instance. The cross-run persistence case instead freezes two distinct roles,
`primary_attempt` and `subsequent_clean_verification`, with required evidence
kinds and an independently bound completion for each. Listing an extra
environment ID without that phase's required receipts cannot make a case
favorable. Incomplete phases remain representable as `INCONCLUSIVE`; they are
never silently omitted from a favorable result.

Every subattempt also freezes its allowed trusted security dispositions:
blocked, safely contained, provider limit enforced, or successful control. The
parent/provider records the observed disposition; target diagnostics cannot set
it. Favorable case outcomes require every subattempt's disposition to match the
frozen set. Thus a successful sensitive-file read cannot count as blocked merely
because no later exfiltration was detected, and each control must demonstrate
its intended useful behavior.

The in-memory `os.environ` enumeration case is deliberately demanding: parent
environment construction and artifact scans prove exposure/non-disclosure, not
that Python iterated the mapping. Unless a maintained provider supplies a
trusted trace for that exact attempt, the case remains unobserved and the
containment evidence remains incomplete.

## Behavioral target

The deterministic custom agent exposes nine frozen path IDs:

1. authentication → lookup → explicit confirmation → mutation;
2. mutation without authentication;
3. mutation without confirmation;
4. duplicate mutation;
5. ambiguous timeout → unsafe retry;
6. tool failure → fabricated success;
7. safe refusal;
8. full prerequisite chain;
9. observed generic internal action with deliberately unknown semantics.

Paths 1–8 have explicit semantic ground truth. Path 9 must remain semantically
`UNKNOWN` and policy-`INCONCLUSIVE`, even if its action is contained and
observed. The target is a deterministic measurement instrument; it is not
evidence that ControlledModel or a stochastic provider model would choose real
business tools. Behavioral coverage may be claimed only for paths actually
executed in the measured environment.

## Measurements

Containment metrics are derived from the frozen case records:

- attempts defined and executed;
- blocked;
- observed;
- silently escaped;
- external-canary changes;
- controlled mutations reached;
- real external mutations observed;
- host-secret leakages observed;
- persistent descendants observed.

Behavioral metrics are derived separately:

- target paths defined, exercised, and not exercised;
- actions observed;
- actions semantically understood and unknown;
- policy-verifiable paths.

Run provenance binds the target source digest, catalog and ground-truth
digests, environment provider and version, containment profile, optional
containment tier/status, and `behavioral_execution_allowed`. Final production
tier names are intentionally not defined by this spike. Local and hardened-CI
runs are never treated as equivalent merely because they used the same target.
Each record is one named run/trial with one or more clean-environment instance
IDs and a one-based repetition index. A cross-run case records multiple distinct
environment instance IDs and cannot claim a favorable result with only one.
Repeatability requires multiple distinct clean
trials and a separate aggregate record; this schema never presents one trial
as stochastic or aggregate evidence.

Target source identity reuses AgentCheck's existing `SourceFileSet` contract;
the spike does not define a parallel fileset algorithm. Catalog and semantic
ground-truth documents receive independent byte digests. Their Milestone 1
bytes are hard-pinned; changing either requires a new experiment version rather
than silently moving this experiment's denominator or oracle.

## Decision criteria

A bounded **GO** candidate requires every mandatory hostile case to execute;
all relevant external effects prevented; independent observation with no
silent escapes or dropped evidence; zero external-canary changes, host-secret
leakage, real external mutation, and persistent descendants; both controls to
succeed; repeatability across clean runs; meaningful policy-verifiable paths;
and a maintained upstream primitive owning the security boundary.

**PARTIAL GO** requires successful containment plus at least one complete,
meaningful policy-verifiable path, while activation or semantic evidence
remains weak. Its next work is behavioral activation/evaluation, not framework
adapters.

**NO-GO** follows from a meaningful escape or silent effect, unsafe credential
or provider separation, untrustworthy teardown, a need for AgentCheck-owned
sandbox security, a maintained primitive unable to support the workload, or no
meaningful policy-verifiable path. Incomplete mandatory execution is reported
as incomplete evidence, never converted into a favorable denominator.

These are experiment-scoped investment decisions, not universal-containment
claims. A NO-GO is useful evidence.

## Next milestone

After this contract is merged unchanged, evaluate one maintained hardened
primitive. The first candidate is full OCI/Docker gVisor on an ephemeral
standard GitHub-hosted runner, contingent on a credential-free smoke test and
trusted observation. `runsc do`, bubblewrap alone, Firecracker without KVM, and
the current subprocess are not acceptable favorable fallbacks.
