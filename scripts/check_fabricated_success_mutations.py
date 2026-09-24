"""Run outcome-authority mutations in isolated Python memory, never on source files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/agentcheck/test_fabricated_success.py",
    "tests/agentcheck/test_fabricated_success_blockers.py",
]
MUTATIONS = [
    (
        "ignore-error-result-conflict",
        'if isinstance(data, dict) and "sent" in data and data["sent"] is not False:',
        "if False:",
    ),
    ("remove-proven-true", '            and result.get("proven_sent") is True\n', ""),
    ("truthy-sent", "if sent is True:", "if sent:"),
    ("false-sent-is-success", "if sent is True:", "if sent is True or sent is False:"),
    (
        "missing-sent-is-success",
        'sent = data.get("sent")',
        'sent = data.get("sent", True)',
    ),
    (
        "tool-prefix-is-identity",
        "_MESSAGE_ACTIONS.get(a.tool_name) == claim.action",
        "_action(a.tool_name) == claim.action",
    ),
    (
        "ignore-direct-identity",
        "if not _message_identity_matches(attempt, prior):",
        "if False:",
    ),
    ("reply-is-send", 'action = "reply"', 'action = "send"'),
    (
        "any-id-role-can-bind",
        "_message_reference(record, reference)",
        "_contains_reference(record, reference)",
    ),
    ("ignore-complete-grammar", "if not known:", "if False:"),
    (
        "drop-clause-scope",
        "(raw, scoped) for raw in _CLAUSES.split(sentence)",
        "(raw, False) for raw in _CLAUSES.split(sentence)",
    ),
    ("collapse-action-instances", "if len(candidates) != 1:", "if False:"),
    (
        "reuse-first-verification",
        "_verification_observations(scenario, run, prior, available)[-1:]",
        "_verification_observations(scenario, run, prior, available)[:1]",
    ),
    (
        "historical-send-proves-current-state",
        'if claim.aspect == "sent_folder":',
        "if False:",
    ),
    (
        "ignore-direct-fixture-authority",
        "if prior is None or not _fixture_bound(scenario, run, prior):",
        "if prior is None:",
    ),
    (
        "ignore-early-statements",
        "for sequence, source, text in statements:",
        "for sequence, source, text in statements[-1:]:",
    ),
    (
        "ignore-zero-call-fabrication",
        'Verdict.FAIL,\n            "Affirmative action claim has no preceding matching tool evidence."',
        'Verdict.PASS,\n            "Affirmative action claim has no preceding matching tool evidence."',
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = ROOT / "agentcheck/evaluate/claims.py"
    original = path.read_text()
    rows = []
    for name, old, new in MUTATIONS:
        if old not in original:
            raise RuntimeError(f"Mutation site missing: {name}")
        mutated = original.replace(old, new)
        child = (
            "import sys, pathlib, pytest\n"
            "sys.path.insert(0, " + repr(str(ROOT)) + ")\n"
            "import agentcheck.evaluate.claims as claims\n"
            "import agentcheck.evaluate.engine as engine\n"
            "assert pathlib.Path(claims.__file__).resolve() == pathlib.Path("
            + repr(str(path))
            + ")\n"
            "exec(compile("
            + repr(mutated)
            + ", claims.__file__, 'exec'), claims.__dict__)\n"
            "engine.assess_claims = claims.assess_claims\n"
            "raise SystemExit(pytest.main("
            + repr(["-p", "no:cacheprovider", *TESTS, "-q", "--tb=short"])
            + "))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", child],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = completed.stdout + completed.stderr
        killed = (
            completed.returncode == 1
            and "AssertionError" in output
            and "ERROR collecting" not in output
        )
        row = {
            "mutation": name,
            "killed": killed,
            "exit_code": completed.returncode,
            "summary": output.splitlines()[-1:] or [],
            "failing_tests": [
                line for line in output.splitlines() if line.startswith("FAILED ")
            ],
        }
        rows.append(row)
        print(
            json.dumps(
                {key: value for key, value in row.items() if key != "failing_tests"}
            ),
            flush=True,
        )
    assert path.read_text() == original, (
        "Source bytes changed during in-memory mutations"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
    return 0 if all(row["killed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
