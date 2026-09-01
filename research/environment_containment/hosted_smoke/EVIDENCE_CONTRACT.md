# Harmless hosted maintained-runtime smoke evidence contract

Status: **UNEXECUTED REVIEW CANDIDATE**. This candidate does not establish
containment and must be independently reviewed before the branch is pushed or
the workflow is run.

## Purpose and boundary

The candidate answers one dependency question only: can a fresh standard
GitHub-hosted Ubuntu 24.04 VM install the frozen Docker, gVisor, and OCI inputs,
run one benign full-OCI container through an explicitly configured `systrap`
runtime, prove that runtime from the host, and tear the container down without
changing a host-side sentinel?

It does not run AgentCheck, a target repository, a hostile case, a model, a
provider, or a real integration. It does not test the frozen M2 catalog. A
successful smoke would not prove containment, observation completeness,
semantic authority, policy correctness, zero mutation outside the one exact
sentinel, or an accepted `EnvironmentProvider`.

## Review and invocation boundary

`.github/workflows/maintained-runtime-smoke.yml` has only
`workflow_dispatch`, top-level `permissions: {}`, one GitHub-hosted
`ubuntu-24.04` job, and shell steps. It uses no action, checkout, repository
content, token, secret, credential, privileged GitHub environment, or automatic
trigger. The candidate must remain unpushed and unexecuted until a fresh
independent reviewer accepts the exact committed tree.

Any future input refresh is a source change requiring renewed review. The
workflow never follows a moving release alias or changes a digest after a
download fails.

## Frozen input authority

`frozen-inputs.json` is the machine-readable source of truth for every value
duplicated into the no-checkout workflow. Its exact sources and checks are:

- Docker's HTTPS signing key, exact key-file SHA-256, primary fingerprint, and
  signing-subkey fingerprint;
- an exact SHA-256-pinned Docker Noble `InRelease` plus its detached inline
  signature verification, and the exact SHA-256-pinned amd64 `Packages.gz` it
  authenticates;
- exact `containerd.io`, Docker CLI, and Docker Engine versions, filenames,
  sizes, architectures, and SHA-256 values from that signed package index;
- gVisor `release-20260817.0`, its full commit, exact archive size, GitHub asset
  SHA-256, official storage SHA-512, checksum-file SHA-256, exact six-file
  archive manifest, and every extracted member SHA-512;
- the exact linux/amd64 Docker Official Image platform-manifest digest and
  config digest. The human-readable image version is metadata, never a tag used
  for execution.

The workflow verifies each checksum, fingerprint, signed-index relationship,
package control field, archive member, installed file, and version before use.
It tells `runsc install` both `download-sidecars=NEVER` and
`require-sidecars=ALWAYS`; a missing sidecar cannot trigger a mutable download.

## Required host-produced evidence

The log must contain `AC_SMOKE_EVIDENCE_V1` records for all of the following,
and the reviewer must also retain the GitHub `Set up job` runner-image block:

1. actual `ImageOS`, `ImageVersion`, runner OS/architecture, Ubuntu release,
   and kernel;
2. initial Docker client/server versions and exact installed package versions;
3. final Docker client/server versions, both exactly `29.7.2`;
4. exact gVisor release output, archive/member checks, complete-sidecar policy,
   Docker runtime registration, runtime path, and `systrap` argument;
5. exact OCI manifest reference, image OS/architecture, config digest, and
   registry `RepoDigest`;
6. live container ID, explicit `HostConfig.Runtime=ac-smoke-runsc`, host PID,
   exact `/proc/<pid>/exe` path, and a SHA-512 matching one frozen gVisor
   executable;
7. no bind mount, `network=none`, read-only root, non-privileged state, benign
   fixed stdout, and container exit code zero;
8. automatic container removal, zero remaining frozen-gVisor processes,
   runtime unregistration, image removal, frozen gVisor file removal, temporary
   download/extraction directory removal, and unchanged sentinel SHA-256.

In-container `dmesg` is neither invoked nor accepted as runtime identity.
Target stdout would be untrusted in M2; here the fixed benign stdout is only a
liveness receipt and never substitutes for host-side runtime proof.

## Fail-closed result

Any download, key, signature, metadata, package, version, archive, sidecar,
runtime registration, `systrap`, OCI identity, container, host-process,
sentinel, exit, removal, residual-process, runtime-unregistration, or image-
cleanup mismatch makes the job fail. This includes a remaining temporary
download/extraction directory. Cleanup runs on every exit and cannot turn an
earlier error into success. There is no retry through another runtime, no
standalone runsc execution mode, and no trusted-subprocess path.

## Interpretation

- **Candidate not run:** all hosted/runtime outcomes are **NOT PROVEN**.
- **Job failure:** the smoke is incomplete/failed; it is never favorable
  evidence and does not authorize a fallback.
- **Job success after exact-tree review:** only the bounded dependency smoke is
  observed. A separate independent evidence/security review is still required
  before changing `AC-P1-3` or considering hostile M2 execution.
