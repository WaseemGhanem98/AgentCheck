# CI trust model

> **If you are about to make this repository public, read this first.**
> Normal CI currently runs on a self-hosted runner that is a maintainer's
> personal workstation. That is safe while this repository is private. It stops
> being safe the moment anyone can open a pull request.

## Where CI runs today

| Workflow | Runner | Trigger | Why |
|---|---|---|---|
| `ci.yml` | `[self-hosted, Linux, X64]` (`agentlens-local`) | `push`, `pull_request` | Every branch here is written by the owner, and GitHub-hosted minutes are billed against the account while the repository is private. |
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
2. **Remove the `if:` guards** once no job targets `agentlens-local`, since they
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
5. **Do not attach `agentlens-local` to this repository's runner list**, and do
   not use `pull_request_target` anywhere.
6. **Budget for the minutes.** Actions minutes are free on public
   repositories, which is what makes step 1 affordable. While this repository
   is private they are billed: the single bootstrap CI run consumed roughly 84
   minutes of job time, most of it three full-suite jobs at 23–29 minutes each,
   and the account then declined to start further hosted jobs.

`tests/agentcheck/test_workflow_safety.py::test_publication_checklist_is_discoverable`
asserts this document exists and that `ci.yml` carries the warning inline, so
the decision cannot quietly disappear.

## The rule that does not change

**Arbitrary code from a public pull request must never execute on
`agentlens-local`,** whatever else is convenient.
