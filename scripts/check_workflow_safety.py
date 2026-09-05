#!/usr/bin/env python3
"""Statically enforce AgentCheck's public GitHub Actions trust boundary."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


WORKFLOW_DIRECTORY = Path(".github/workflows")
FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
HOSTED_RUNNERS = frozenset({"ubuntu-latest", "ubuntu-24.04"})
FORBIDDEN_WORKFLOW_REFERENCE = re.compile(r"self-hosted|agentcheck-local", re.I)


def _workflow_paths() -> list[Path]:
    return sorted(WORKFLOW_DIRECTORY.glob("*.yml")) + sorted(
        WORKFLOW_DIRECTORY.glob("*.yaml")
    )


def _triggers(workflow: Mapping[Any, Any]) -> set[str]:
    # PyYAML 1.1 resolves an unquoted `on:` key to boolean True.
    raw = workflow.get("on", workflow.get(True))
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(item) for item in raw}
    if isinstance(raw, dict):
        return {str(item) for item in raw}
    return set()


def _hosted_runner(runs_on: Any) -> bool:
    return isinstance(runs_on, str) and runs_on in HOSTED_RUNNERS


def _safe_top_level_permissions(value: Any) -> bool:
    return value in ({}, {"contents": "read"})


def _iter_steps(workflow: Mapping[Any, Any]) -> Iterator[tuple[str, str, Mapping[str, Any]]]:
    for job_name, job in (workflow.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        for index, step in enumerate(job.get("steps") or [], start=1):
            if isinstance(step, dict):
                yield str(job_name), f"step {index}", step


def _read_only_permissions(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        permission in {"read", "none"} for permission in value.values()
    )


def release_artifact_gate_failures(release: Mapping[Any, Any]) -> list[str]:
    """Keep the exact-wheel gate fail-closed, not merely present by name."""
    failures: list[str] = []
    jobs = release.get("jobs") or {}
    build = jobs.get("build") or {}
    publish = jobs.get("publish") or {}
    # This is a guard for the reviewed two-job pipeline, not a universal shell
    # verifier. Inherited shells/workdirs can bypass the otherwise exact step.
    if set(jobs) != {"build", "publish"}:
        failures.append("release.yml must retain only its reviewed build and publish jobs")
    for label, item in (("workflow", release), ("build", build), ("publish", publish)):
        if "defaults" in item:
            failures.append(f"release.yml:{label} must not override inherited run defaults")
    steps = build.get("steps") or []
    gates = [i for i, step in enumerate(steps) if step.get("id") == "qualify_artifacts"]
    uploads = [i for i, step in enumerate(steps)
               if str(step.get("uses", "")).startswith("actions/upload-artifact@")]
    builds = [i for i, step in enumerate(steps)
              if str(step.get("run", "")).strip() == "python -m build"]
    audits = [i for i, step in enumerate(steps)
              if "python scripts/check_distribution.py" in
              str(step.get("run", "")).splitlines()]
    if not all(len(items) == 1 for items in (gates, uploads, builds, audits)):
        return ["release.yml must have one build/audit/qualification/upload chain"]
    gate, upload = steps[gates[0]], steps[uploads[0]]
    command = (
        'python -I scripts/check_release_artifact.py \\\n'
        '  --dist-dir dist --version "${RELEASE_TAG#v}" --source-sha "$RELEASE_SHA"'
    )
    if str(gate.get("run", "")).strip() != command:
        failures.append("release.yml must execute the exact artifact qualification command")
    if gate.get("env") != {
        "RELEASE_SHA": "${{ github.sha }}",
        "RELEASE_TAG": "${{ github.event.release.tag_name }}",
    }:
        failures.append("release.yml qualification must bind the release source and version")
    if not (builds[0] < audits[0] < gates[0] == uploads[0] - 1 == len(steps) - 2):
        failures.append("release.yml must qualify after audit, immediately before final upload")
    for label, item in (("build", build), ("qualification", gate), ("upload", upload),
                        ("publish", publish)):
        if item.get("continue-on-error", False) is not False:
            failures.append(f"release.yml:{label} must not ignore failure")
    for label, item in (("qualification", gate), ("upload", upload)):
        if any(key in item for key in ("if", "shell", "working-directory")):
            failures.append(f"release.yml:{label} must retain default successful-step gating")
    if build.get("timeout-minutes") != 15:
        failures.append("release.yml build must retain its 15 minute budget")
    if upload.get("with", {}).get("path") != "dist/":
        failures.append("release.yml must upload the qualified dist directory")
    if upload.get("with", {}).get("name") != "python-package-distributions":
        failures.append("release.yml qualified artifact name changed")
    if upload.get("with", {}).get("if-no-files-found") != "error":
        failures.append("release.yml missing distributions must fail upload")
    if upload.get("with", {}).get("overwrite", False) is not False:
        failures.append("release.yml qualified artifact producer must not overwrite another artifact")
    if publish.get("needs") != "build" or publish.get("if") != (
        "${{ github.event.release.prerelease == false }}"
    ):
        failures.append("release.yml publish must depend on a successful non-prerelease build")
    publication = publish.get("steps") or []
    if len(publication) != 2 or not (
        str(publication[0].get("uses", "")).startswith("actions/download-artifact@")
        and publication[0].get("with") == {
            "name": "python-package-distributions", "path": "dist/",
        }
        and str(publication[1].get("uses", "")).startswith("pypa/gh-action-pypi-publish@")
        and all("run" not in step and "if" not in step
                and step.get("continue-on-error", False) is False for step in publication)
    ):
        failures.append("release.yml publish must download unchanged artifacts without rebuilding")
    return failures


def main() -> int:
    failures: list[str] = []
    paths = _workflow_paths()
    if not paths:
        print("FAIL: no active workflow files found")
        return 1

    for path in paths:
        text = path.read_text(encoding="utf-8")
        workflow = yaml.safe_load(text)
        if not isinstance(workflow, dict):
            failures.append(f"{path.name}: workflow is not a mapping")
            continue
        triggers = _triggers(workflow)

        if "pull_request_target" in triggers:
            failures.append(f"{path.name}: pull_request_target is forbidden")
        if not _safe_top_level_permissions(workflow.get("permissions")):
            failures.append(
                f"{path.name}: top-level permissions must be empty or exactly "
                "contents: read"
            )
        if re.search(r"\$\{\{\s*secrets\.", text) or "PYPI_API_TOKEN" in text:
            failures.append(f"{path.name}: workflow references a long-lived secret")
        if FORBIDDEN_WORKFLOW_REFERENCE.search(text):
            failures.append(f"{path.name}: workflow references a workstation runner")

        for job_name, job in (workflow.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            if not _hosted_runner(job.get("runs-on")):
                failures.append(
                    f"{path.name}:{job_name} must use an ephemeral GitHub-hosted runner"
                )
            is_publish_job = path.name == "release.yml" and job_name == "publish"
            if "environment" in job and not is_publish_job:
                failures.append(
                    f"{path.name}:{job_name} uses a privileged environment"
                )
            permissions = job.get("permissions")
            if (
                not is_publish_job
                and permissions is not None
                and not _read_only_permissions(permissions)
            ):
                failures.append(
                    f"{path.name}:{job_name} grants unnecessary write permission"
                )

        for job_name, step_label, step in _iter_steps(workflow):
            uses = step.get("uses")
            if not isinstance(uses, str) or uses.startswith("./"):
                continue
            action, separator, ref = uses.rpartition("@")
            if not separator or not FULL_COMMIT_SHA.fullmatch(ref):
                failures.append(
                    f"{path.name}:{job_name}:{step_label} action {uses!r} is not "
                    "pinned to a full commit SHA"
                )
            if action == "actions/checkout":
                options = step.get("with") or {}
                if not isinstance(options, dict) or options.get("persist-credentials") is not False:
                    failures.append(
                        f"{path.name}:{job_name}:{step_label} checkout must set "
                        "persist-credentials: false"
                    )

    ci_path = WORKFLOW_DIRECTORY / "ci.yml"
    if ci_path not in paths:
        failures.append("ci.yml is missing")
    else:
        ci = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
        if _triggers(ci) != {"push", "pull_request"}:
            failures.append("ci.yml must run for push and pull_request only")
        ci_text = ci_path.read_text(encoding="utf-8")
        if "tests -q -n 1" not in ci_text:
            failures.append("ci.yml must serialize the process-heavy full suite")
        if '"${compat[@]}" -q -n 1' not in ci_text:
            failures.append("ci.yml must serialize the compatibility suite")

        jobs = ci.get("jobs") or {}
        scope = jobs.get("scope") or {}
        tests = jobs.get("tests") or {}
        checks = jobs.get("checks") or {}
        if scope.get("outputs") != {
            "expensive": "${{ steps.classify.outputs.expensive }}"
        }:
            failures.append("ci.yml:scope must expose the classifier output")
        if tests.get("needs") != "scope":
            failures.append("ci.yml:tests must depend on scope")
        if tests.get("if") != "needs.scope.outputs.expensive == 'true'":
            failures.append("ci.yml:tests must run only for full validation")
        if checks.get("needs") != "scope":
            failures.append("ci.yml:checks must depend on scope")

        scope_steps = scope.get("steps") or []
        classify = next(
            (
                step
                for step in scope_steps
                if isinstance(step, dict) and step.get("id") == "classify"
            ),
            {},
        )
        classify_run = classify.get("run", "")
        for required in (
            "expensive=true",
            "git cat-file -e",
            "git diff --no-renames --name-only -z",
            "*.md|docs/assets/*",
            "GITHUB_OUTPUT",
        ):
            if required not in classify_run:
                failures.append(f"ci.yml:scope classifier is missing {required!r}")

        scope_checkout = next(
            (
                step
                for step in scope_steps
                if isinstance(step, dict)
                and str(step.get("uses", "")).startswith("actions/checkout@")
            ),
            {},
        )
        scope_options = scope_checkout.get("with") or {}
        if scope_options.get("fetch-depth") != 0:
            failures.append("ci.yml:scope checkout must fetch complete comparison history")

        if "needs.scope.outputs.expensive == 'false'" not in ci_text:
            failures.append("ci.yml must run focused checks for documentation-only changes")

    release_path = WORKFLOW_DIRECTORY / "release.yml"
    if release_path not in paths:
        failures.append("release.yml is missing")
    else:
        release = yaml.safe_load(release_path.read_text(encoding="utf-8"))
        jobs = release.get("jobs") or {}
        build = jobs.get("build") or {}
        publish = jobs.get("publish") or {}
        if _triggers(release) != {"release"}:
            failures.append("release.yml must be triggered only by a GitHub Release")
        if build.get("permissions") != {"contents": "read"}:
            failures.append("release.yml:build must have only contents: read")
        if publish.get("permissions") != {"id-token": "write"}:
            failures.append("release.yml:publish must have only id-token: write")
        environment = publish.get("environment")
        environment_name = (
            environment.get("name") if isinstance(environment, dict) else environment
        )
        if environment_name != "pypi":
            failures.append("release.yml:publish must use the pypi environment")
        failures.extend(release_artifact_gate_failures(release))

    disabled = WORKFLOW_DIRECTORY / "ci-public.yml.disabled"
    if disabled.exists():
        failures.append("stale disabled public CI workflow must be removed")

    if failures:
        print("FAIL: workflow trust or supply-chain controls are incomplete")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    job_count = sum(
        len((yaml.safe_load(path.read_text(encoding="utf-8")).get("jobs") or {}))
        for path in paths
    )
    print(
        f"PASS: {job_count} job(s) use GitHub-hosted runners; PR workflows are "
        "read-only, secret-free, environment-free, and SHA-pinned"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
