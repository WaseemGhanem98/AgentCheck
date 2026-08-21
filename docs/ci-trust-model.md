# CI trust model

> **If you are about to make this repository public, read this first.**
> Normal CI currently runs on a self-hosted runner that is a maintainer's
> personal workstation. That is safe while this repository is private. It stops
> being safe the moment anyone can open a pull request.

## What runs, and on which interpreter

The full suite runs once, on the primary interpreter. Every other supported
version runs the cross-version suite defined in `tests/compat_manifest.py`.

| | Py3.10 | Py3.11 | Py3.12 |
|---|---|---|---|
| Full behavioural suite (964 tests) | | | ✅ |
| Domain models, serialization, fingerprints | ✅ | ✅ | ✅ |
| Worker / process isolation | ✅ | ✅ | ✅ |
| ToolGateway, fail-closed, unknown tools | ✅ | ✅ | ✅ |
| Prerequisites, confirmation, `followup_turns` | ✅ | ✅ | ✅ |
| Replay, source integrity | ✅ | ✅ | ✅ |
| OpenAI Agents adapter | ✅ | ✅ | ✅ |
| PydanticAI adapter | ✅ | ✅ | ✅ |
| CLI end-to-end, target import, package boundary | ✅ | ✅ | ✅ |
| Network containment | ✅ | ✅ | ✅ |
| Packaging, extras, clean install | | | ✅ (checks job) |
| Capability inference, generation, selection, shrink, reporting logic | | | ✅ |

The split is by **where the interpreter can reach**, not by how slow a file is.
The package contains no `sys.version_info` branching, so what differs between
3.10 and 3.12 is the boundary with the interpreter: the recursive
`TypeAliasType` models in `agentcheck/domain/base.py` that every contract sits
on, subprocess worker launch, import machinery, third-party SDK internals, and
the `argparse` console entry point. All of those run everywhere. Pure logic over
already-constructed objects runs once.

The cross-version suite is 381 of 964 tests across 18 files. Measured on the
self-hosted runner it takes about 8 minutes per interpreter, against 21 for the
full suite. It is not a smoke test.

`tests/agentcheck/test_compat_manifest.py` fails if a category loses its files,
if a listed file disappears, or if a listed file stops containing the symbols
that made it evidence — and if `ci.yml` stops reading the list from the manifest.

## Post-merge

A pull request runs all three interpreters. A push to `main` runs only the
primary, plus the full checks job.

`pull_request` events already build and test the **merge result**, not the
branch tip, so post-merge CI is not re-testing an untested tree. What it still
catches is a merge race: `main` advancing between the moment PR CI ran and the
moment the merge landed. That risk is real but interpreter-independent — a
semantic conflict breaks on 3.12 exactly as it breaks on 3.10 — so re-running
the other two versions after merge buys nothing and costs about 45 minutes of a
serial queue.

## Where CI runs today

| Workflow | Runner | Trigger | Why |
|---|---|---|---|
| `ci.yml` | `[self-hosted, Linux, X64]` (`agentcheck-local`) | `push`, `pull_request` | Every branch here is written by the owner, and GitHub-hosted minutes are billed against the account while the repository is private. |
| `agentcheck-example.yml` | `ubuntu-latest` | `workflow_dispatch` | Deliberately GitHub-hosted — see below. |

## Why the example workflow stays GitHub-hosted

`agentcheck-example.yml` is not this repository's CI. It is a template meant to
be copied into a project that uses AgentCheck. Pointing it at `self-hosted`
would hand every reader a workflow that cannot run anywhere but one specific
machine, which defeats its only purpose.

It also costs nothing: it is `workflow_dispatch` only, so it never runs unless
someone triggers it by hand.

Do not "fix" it to match `ci.yml`.

## What makes the self-hosted arrangement safe right now

Two things, and only two:

1. **The repository is private.** Only the owner can push a branch or open a
   pull request, so all executed code is trusted.
2. **Each self-hosted job carries a trusted-context guard:**

   ```yaml
   if: >-
     github.event_name == 'push'
     || (github.event_name == 'pull_request'
         && github.event.pull_request.head.repo.full_name == github.repository)
   ```

   It is an allowlist, not `event_name != 'pull_request'`. A deny-list would
   admit any trigger added to `on:` later without anyone noticing.

   The guard is on **every** job, because a job-level `if` is not inherited and
   one unguarded job is enough to undo the whole arrangement.

`scripts/check_workflow_safety.py` proves this holds by evaluating each guard
against synthetic push, same-repository, fork, `pull_request_target`,
`workflow_run`, `workflow_dispatch`, and `issue_comment` contexts. It runs in
CI and in the test suite. It also fails if a `ci.yml` job drifts *back* to a
GitHub-hosted label, because that spends billed minutes silently — nothing
breaks, the bill just grows.

## Why the answer is simply "move to GitHub-hosted"

Two facts, both verified against GitHub's own documentation rather than assumed,
decide this:

**1. Self-hosted runners are the wrong tool for a public repository.** GitHub's
security hardening guide states that "self-hosted runners should almost never be
used for public repositories", because "anyone who can fork the repository and
open a pull request ... are able to compromise the self-hosted runner
environment, including gaining access to secrets and the `GITHUB_TOKEN`".

**2. The cost objection disappears when the repository is public.** Standard
GitHub-hosted runners are **free with no minute allowance to exhaust** for public
repositories — "GitHub Actions usage is free for self-hosted runners and for
public repositories that use standard GitHub-hosted runners". (Larger runners are
always billed, including on public repositories; standard runners are what this
project needs.)

The only reason CI is self-hosted today is that hosted minutes *are* billed for
**private** repositories, and this account exhausted them. Publication removes
that constraint entirely. There is no trade-off to weigh: going public makes the
secure option also the free one.

Sources:
- Security hardening for GitHub Actions — <https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions>
- About billing for GitHub Actions — <https://docs.github.com/en/billing/managing-billing-for-your-products/about-billing-for-github-actions>

## What must change before this repository becomes public

The guard does not fail open, so publication would not instantly expose the
runner: a fork's pull request is **skipped**, not executed.

That is the problem. A public project where external pull requests get no CI at
all is not a working open-source project, and the obvious "fix" — relaxing the
guard so contributor PRs run — is exactly the change that turns a workstation
into a public code-execution surface.

So, before publication:

1. **Move `ci.yml`'s `tests` and `checks` jobs back to GitHub-hosted runners**
   (`runs-on: ubuntu-latest`), or to properly isolated ephemeral self-hosted
   runners that are destroyed after each job.
2. **Remove the `if:` guards** once no job targets `agentcheck-local`, since they
   exist only to protect that runner.
3. **Restore `actions/setup-python`** in place of the `uv`-provisioned
   interpreters if hosted runners are used. The `uv` approach exists because
   `actions/setup-python`'s interpreters hard-code
   `/opt/hostedtoolcache/...` libpython paths that do not exist on a
   self-hosted runner. On a hosted runner that constraint disappears.
4. **Re-check pytest concurrency.** `-n 2` is measured for a 14-core
   workstation. A hosted runner has far fewer cores, and at `-n 2` there the
   suite starved its own scenario budget: 10 failed / 920 passed on Python 3.11
   and 3.12 while 3.10 passed, from a commit that passes 924/924 locally.
   Scenarios came back `INFRA_ERROR` instead of real verdicts. The fix is fewer
   pytest workers or sharding across jobs — **never** a larger scenario budget,
   which would trade away the signal that detects a starved run.
5. **Do not attach any self-hosted runner to this repository**, and do
   not use `pull_request_target` anywhere.
6. **Enable private vulnerability reporting** in the repository's Security
   settings. GitHub only offers it on public repositories, so it cannot be
   turned on before publication — and `SECURITY.md` directs researchers to it.
7. **Scrub the runner references.** This document and `ci.yml` name
   `agentcheck-local` because that is operationally accurate today. Once CI moves
   to hosted or ephemeral runners those names are stale, and a public repository
   should not carry the label of a machine it no longer uses.
8. **Budget for the minutes.** Actions minutes are free on public
   repositories, which is what makes step 1 affordable. While this repository
   is private they are billed: the single bootstrap CI run consumed roughly 84
   minutes of job time, most of it three full-suite jobs at 23–29 minutes each,
   and the account then declined to start further hosted jobs.

### The prepared replacement

`.github/workflows/ci-public.yml.disabled` is the finished public workflow. It is
deliberately **not** a `.yml` file, so GitHub ignores it and it cannot run while
this repository is private and hosted minutes are billed.

Publication is then two file operations:

```bash
git rm .github/workflows/ci.yml
git mv .github/workflows/ci-public.yml.disabled .github/workflows/ci.yml
```

Everything else in the checklist above is already satisfied by that file: hosted
runners, no self-hosted label, no trusted-context guards, `actions/setup-python`
instead of uv, and pytest concurrency reduced for a smaller machine.

`tests/agentcheck/test_workflow_safety.py::test_publication_checklist_is_discoverable`
asserts this document exists and that `ci.yml` carries the warning inline, so
the decision cannot quietly disappear.

## The rule that does not change

**Arbitrary code from a public pull request must never execute on
`agentcheck-local`,** whatever else is convenient.
