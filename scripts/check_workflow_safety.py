#!/usr/bin/env python3
"""Prove that no untrusted context can reach the self-hosted runner.

It also holds normal CI on that runner. Moving a job back to a GitHub-hosted
label costs billed minutes silently -- nothing fails, the bill just grows -- so
the placement is asserted here rather than left to review.

The self-hosted runner is a personal machine, so a job that runs on it must
execute only repository-controlled code. GitHub does not enforce that: a
job-level ``if`` is not inherited, so one unguarded job is enough to run a
fork's branch here. This script reads the workflows and fails when a
self-hosted job would be reachable from an untrusted context.

It evaluates the guard expressions rather than pattern-matching them, because
the interesting mistakes are logical -- an ``||`` that should be ``&&``, a
deny-list that admits a trigger nobody thought about -- and a substring check
cannot see those.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


WORKFLOW_DIRECTORY = Path(".github/workflows")
SELF_HOSTED_LABEL = "self-hosted"

# Workflows whose jobs must run on the self-hosted runner. Anything else in the
# directory is an example or an on-demand workflow and is exempt; see the
# module docstring in each.
NORMAL_CI_WORKFLOWS = frozenset({"ci.yml"})

# Contexts a workflow can be triggered from. ``trusted`` means the ref being
# checked out is controlled by this repository.
CONTEXTS: dict[str, dict[str, Any]] = {
    "push to main": {
        "trusted": True,
        "github": {
            "event_name": "push",
            "repository": "owner/repo",
            "event": {},
        },
    },
    "pull_request from a branch in this repository": {
        "trusted": True,
        "github": {
            "event_name": "pull_request",
            "repository": "owner/repo",
            "event": {"pull_request": {"head": {"repo": {"full_name": "owner/repo"}}}},
        },
    },
    "pull_request from a fork": {
        "trusted": False,
        "github": {
            "event_name": "pull_request",
            "repository": "owner/repo",
            "event": {
                "pull_request": {"head": {"repo": {"full_name": "attacker/repo"}}}
            },
        },
    },
    "pull_request_target from a fork": {
        "trusted": False,
        "github": {
            "event_name": "pull_request_target",
            "repository": "owner/repo",
            "event": {
                "pull_request": {"head": {"repo": {"full_name": "attacker/repo"}}}
            },
        },
    },
    "workflow_dispatch": {
        "trusted": False,
        "github": {
            "event_name": "workflow_dispatch",
            "repository": "owner/repo",
            "event": {},
        },
    },
    "workflow_run": {
        "trusted": False,
        "github": {
            "event_name": "workflow_run",
            "repository": "owner/repo",
            "event": {},
        },
    },
    "issue_comment": {
        "trusted": False,
        "github": {
            "event_name": "issue_comment",
            "repository": "owner/repo",
            "event": {},
        },
    },
}


class ExpressionError(RuntimeError):
    """The guard uses syntax this checker does not model."""


def _tokenize(source: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char.isspace():
            index += 1
            continue
        if source.startswith("&&", index) or source.startswith("||", index):
            tokens.append(source[index : index + 2])
            index += 2
            continue
        if source.startswith("==", index) or source.startswith("!=", index):
            tokens.append(source[index : index + 2])
            index += 2
            continue
        if char in "()":
            tokens.append(char)
            index += 1
            continue
        if char == "!":
            tokens.append(char)
            index += 1
            continue
        if char == "'":
            end = source.index("'", index + 1)
            tokens.append(source[index : end + 1])
            index = end + 1
            continue
        start = index
        while index < len(source) and (source[index].isalnum() or source[index] in "._-"):
            index += 1
        if index == start:
            raise ExpressionError(f"unexpected character {char!r} in guard")
        tokens.append(source[start:index])
    return tokens


class _Parser:
    """Recursive descent over the subset of expressions these guards use."""

    def __init__(self, tokens: list[str], context: dict[str, Any]) -> None:
        self.tokens = tokens
        self.position = 0
        self.context = context

    def peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self) -> str:
        token = self.tokens[self.position]
        self.position += 1
        return token

    def parse(self) -> Any:
        value = self.parse_or()
        if self.position != len(self.tokens):
            raise ExpressionError("trailing tokens in guard")
        return value

    def parse_or(self) -> Any:
        left = self.parse_and()
        while self.peek() == "||":
            self.take()
            right = self.parse_and()
            left = bool(left) or bool(right)
        return left

    def parse_and(self) -> Any:
        left = self.parse_equality()
        while self.peek() == "&&":
            self.take()
            right = self.parse_equality()
            left = bool(left) and bool(right)
        return left

    def parse_equality(self) -> Any:
        left = self.parse_unary()
        while self.peek() in ("==", "!="):
            operator = self.take()
            right = self.parse_unary()
            left = (left == right) if operator == "==" else (left != right)
        return left

    def parse_unary(self) -> Any:
        if self.peek() == "!":
            self.take()
            return not bool(self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Any:
        token = self.peek()
        if token is None:
            raise ExpressionError("guard ended unexpectedly")
        if token == "(":
            self.take()
            value = self.parse_or()
            if self.take() != ")":
                raise ExpressionError("unbalanced parenthesis in guard")
            return value
        token = self.take()
        if token.startswith("'"):
            return token[1:-1]
        if token in ("true", "false"):
            return token == "true"
        return self._lookup(token)

    def _lookup(self, path: str) -> Any:
        # An absent property is the empty string in GitHub expressions, which is
        # falsy and never equal to a non-empty literal. Modelling it any other
        # way would make a guard look safer than it is.
        current: Any = self.context
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return ""
        return current


def evaluate(expression: str, context: dict[str, Any]) -> bool:
    text = expression.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2]
    return bool(_Parser(_tokenize(text), context).parse())


def _is_self_hosted(runs_on: Any) -> bool:
    if isinstance(runs_on, str):
        return runs_on == SELF_HOSTED_LABEL
    if isinstance(runs_on, list):
        return SELF_HOSTED_LABEL in runs_on
    if isinstance(runs_on, dict):
        labels = runs_on.get("labels", [])
        if isinstance(labels, str):
            return labels == SELF_HOSTED_LABEL
        return SELF_HOSTED_LABEL in (labels or [])
    return False


def _triggers(workflow: Mapping[Any, Any]) -> set[str]:
    # PyYAML resolves an unquoted `on:` key to the boolean True.
    raw = workflow.get("on", workflow.get(True))
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return set(raw)
    if isinstance(raw, dict):
        return set(raw)
    return set()


def main() -> int:
    failures: list[str] = []
    checked = 0

    for path in sorted(WORKFLOW_DIRECTORY.glob("*.yml")) + sorted(
        WORKFLOW_DIRECTORY.glob("*.yaml")
    ):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        triggers = _triggers(workflow)
        for job_name, job in (workflow.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            if not _is_self_hosted(job.get("runs-on")):
                continue
            checked += 1
            guard = job.get("if")
            if guard is None:
                failures.append(
                    f"{path.name}:{job_name} runs self-hosted with no trusted-context guard"
                )
                continue
            for label, context in CONTEXTS.items():
                event = context["github"]["event_name"]
                # Only contexts the workflow can actually be triggered from can
                # reach a job, but an untrusted trigger that is not declared is
                # still worth reporting if the guard would admit it.
                try:
                    allowed = evaluate(str(guard), {"github": context["github"]})
                except ExpressionError as error:
                    failures.append(f"{path.name}:{job_name} guard not understood: {error}")
                    break
                if allowed and not context["trusted"]:
                    reachable = event in triggers
                    detail = "declared trigger" if reachable else "undeclared trigger"
                    failures.append(
                        f"{path.name}:{job_name} would run for {label} ({detail})"
                    )
            if "pull_request_target" in triggers:
                failures.append(
                    f"{path.name}:{job_name} is self-hosted in a workflow that uses "
                    "pull_request_target"
                )

    for name in sorted(NORMAL_CI_WORKFLOWS):
        path = WORKFLOW_DIRECTORY / name
        if not path.exists():
            failures.append(f"{name} is missing; normal CI placement cannot be checked")
            continue
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job_name, job in (workflow.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            if not _is_self_hosted(job.get("runs-on")):
                failures.append(
                    f"{name}:{job_name} runs on {job.get('runs-on')!r} rather than the "
                    "self-hosted runner, which spends billed GitHub minutes"
                )

    if not checked:
        print("FAIL: no self-hosted job found; this checker would pass vacuously")
        return 1
    if failures:
        print("FAIL: workflow trust or runner placement is wrong")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"PASS: {checked} self-hosted job(s) reject every untrusted context, "
        "and normal CI stays off GitHub-hosted runners"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
