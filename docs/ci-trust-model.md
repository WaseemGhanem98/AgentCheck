# CI trust model

AgentCheck treats pull-request code as untrusted. All active test, quality, and
dependency-review jobs run on ephemeral GitHub-hosted runners with a read-only
`GITHUB_TOKEN`, no repository secrets, and no privileged environment.

## Workflow routing

| Event | Runner | Trust boundary |
|---|---|---|
| Pull request from this repository | GitHub-hosted | Read-only; ordinary PR review applies. |
| Pull request from a fork | GitHub-hosted | Read-only token, no secrets, no environment. |
| Push to `main` | GitHub-hosted | Trusted source, but no additional token permission. |
| Published GitHub Release | GitHub-hosted | Separate build and OIDC publish jobs; `pypi` environment approval required. |

PR workflows use `pull_request`, never `pull_request_target`. External action
references are pinned to immutable commit SHAs, and checkout does not persist
credentials. `scripts/check_workflow_safety.py` enforces these repository-side
controls.

For code and runtime changes, the test matrix runs the full suite on Python 3.12
and the compatibility manifest on Python 3.10 and 3.11. Those process-heavy
invocations use one pytest worker on standard hosted runners because AgentCheck
scenarios already execute in child processes. A fail-closed scope check skips
that matrix only when every changed path is Markdown or under `docs/assets/`.
The documentation-only path still runs workflow-safety and secret checks plus
focused documentation consistency tests. Missing or unresolvable comparison
SHAs select full validation. Product scenario timeouts and wall-clock budgets
are not widened for CI.

## Self-hosted runners

No self-hosted runner is registered with this public repository, and public
workflows must remain GitHub-hosted. A fork can propose a workflow that targets
any repository-level runner; job guards, CODEOWNERS, and branch rules operate
too late to prevent that pre-merge execution.

If a repository-level self-hosted runner is ever attached accidentally, remove
it under **Settings → Actions → Runners**. Do not rely on a YAML `if`
condition to protect a personal machine from untrusted pull-request code.

## Fork approval and secrets

Set **Settings → Actions → General → Approval for running fork pull request
workflows** to require approval for all external contributors. Review workflow
changes before approving a run. This is
defense-in-depth; GitHub-hosted isolation, read-only permissions, and the absence
of secrets are still required.

No AgentCheck test requires a provider credential. Fork PRs must never receive
repository secrets, write tokens, deployment environments, or PyPI publishing
authority.

## Release boundary

`.github/workflows/release.yml` runs only for a published, non-prerelease GitHub
Release. Its build job has `contents: read`; its publish job receives only
`id-token: write`, downloads the exact validated artifacts, and uses the `pypi`
environment for Trusted Publishing. It has no API token or manual publication
trigger.

Protect `main`, `v*` tags, the release workflow, `pyproject.toml`, and version
files with CODEOWNER review and GitHub rulesets. Configure the `pypi` environment
for owner approval and `v*` deployment tags before the next release.
