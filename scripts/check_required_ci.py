#!/usr/bin/env python3
"""Resolve the whole CI workflow into one required status.

Branch protection needs a check name that always reports. The expensive test
matrix cannot be that name: it is skipped by design on documentation-only pull
requests, and a skipped job never reports, so requiring it leaves those pull
requests waiting for a status that will never arrive.

This collapses the workflow into a single answer. The rule it applies is the one
that is easy to get wrong by hand:

* the classifier and the quality job must have *succeeded* in every scope --
  neither is ever skipped, so anything else is a real problem;
* the matrix must have succeeded when the change was classified expensive;
* the matrix may be skipped only when the change was classified cheap;
* a failed or cancelled matrix is never an acceptable skip, in either scope.

Anything it cannot classify is treated as a code change, so a broken or missing
classification demands the full suite rather than waving the change through.

Reads ``needs.<job>.result`` values and the classifier output from the
environment. Standard library only: this runs before any project install.
"""

from __future__ import annotations

import os
import sys


SUCCESS = "success"
SKIPPED = "skipped"

# GitHub reports one of these for `needs.<job>.result`.
KNOWN_RESULTS = frozenset({SUCCESS, SKIPPED, "failure", "cancelled"})


def _decide(
    scope: str, tests: str, checks: str, expensive: str
) -> tuple[bool, list[str]]:
    problems: list[str] = []

    # Neither of these is conditional, so "not success" always means broken.
    if scope != SUCCESS:
        problems.append(
            f"scope (change classification) reported {scope or 'nothing'!r}; "
            "expected 'success'"
        )
    if checks != SUCCESS:
        problems.append(
            f"checks (quality, packaging, extras, workflow trust) reported "
            f"{checks or 'nothing'!r}; expected 'success'"
        )

    # An unrecognised classification is a code change. Failing closed here is
    # what stops a broken classifier from silently buying a free pass.
    cheap = expensive == "false"
    if expensive not in {"true", "false"}:
        problems.append(
            f"scope produced expensive={expensive or 'nothing'!r}, which is neither "
            "'true' nor 'false'; treating this as a code change"
        )

    if tests == SUCCESS:
        pass
    elif tests == SKIPPED:
        if not cheap:
            problems.append(
                "tests were skipped on a change that requires the full matrix; "
                "a skipped suite never satisfies this gate outside a "
                "documentation-only change"
            )
    elif tests in KNOWN_RESULTS:
        # failure / cancelled -- never acceptable, including on a docs-only run
        # where the matrix was not expected to contribute anything.
        problems.append(
            f"tests reported {tests!r}; a failed or cancelled matrix is never "
            "treated as an acceptable skip"
        )
    else:
        problems.append(f"tests reported the unrecognised result {tests or 'nothing'!r}")

    return not problems, problems


def main() -> int:
    scope = os.environ.get("SCOPE_RESULT", "")
    tests = os.environ.get("TESTS_RESULT", "")
    checks = os.environ.get("CHECKS_RESULT", "")
    expensive = os.environ.get("EXPENSIVE", "")

    print(
        "required CI inputs: "
        f"scope={scope or '(unset)'} tests={tests or '(unset)'} "
        f"checks={checks or '(unset)'} expensive={expensive or '(unset)'}"
    )

    ok, problems = _decide(scope, tests, checks, expensive)
    if ok:
        shape = "documentation-only" if expensive == "false" else "full"
        print(f"PASS: every required job reported an acceptable result ({shape} scope)")
        return 0

    print("FAIL: required CI did not pass")
    for problem in problems:
        print(f"  - {problem}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
