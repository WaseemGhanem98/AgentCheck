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
    "tests/agentcheck/test_fabricated_success_attribution.py",
    "tests/agentcheck/test_fabricated_success_evidence.py",
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
        "bypass-action-identity",
        "_MESSAGE_ACTIONS.get(a.tool_name) == claim.action",
        "a.tool_name in _MESSAGE_ACTIONS",
    ),
    ("bypass-channel-identity", '"channel" in r and r["channel"] != "email"', "False"),
    ("ignore-recipient-identity", "if not _recipients_agree(records):", "if False:"),
    ("omit-draft-identity", '    "draft_id",\n', ""),
    (
        "cross-verification-identity-bypass",
        "        and _identity_agrees(\n            [action.arguments, prior.result, verification.arguments, proof.result]\n        )",
        "        and True",
    ),
    (
        "any-id-role-can-bind",
        'keys = ("message_id",)',
        'keys = ("message_id", "thread_id")',
    ),
    (
        "reuse-first-verification",
        "_verification_observations(scenario, run, prior, available)[-1:]",
        "_verification_observations(scenario, run, prior, available)[:1]",
    ),
    (
        "skip-unknown-verification",
        "            observations.append(o)",
        "            if _verification_authoritative(scenario, run, prior, o):\n                observations.append(o)",
    ),
    (
        "select-stale-state",
        "latest = max(o.sequence for o in observations)",
        "latest = min(o.sequence for o in observations)",
    ),
    (
        "ignore-current-contradiction",
        "        if not authoritative or type(state) is not bool:",
        "        if state is False:\n            continue\n        if not authoritative or type(state) is not bool:",
    ),
    (
        "discard-missing-current-state",
        '        state = data.get("in_sent") if isinstance(data, dict) else None',
        '        if isinstance(data, dict) and "in_sent" not in data:\n            continue\n        state = data.get("in_sent") if isinstance(data, dict) else None',
    ),
    (
        "ignore-equal-authority-conflict",
        "if len(values) != 1 or None in values:",
        "if False:",
    ),
    (
        "configured-phrase-is-proof",
        '    if claim.polarity == "abstention":',
        '    if claim.declared:\n        return result(Verdict.PASS, "Mutant treats configured vocabulary as proof.")\n    if claim.polarity == "abstention":',
    ),
    (
        "allow-double-negative-subject",
        "{negative_subject}){argument}(?:\\s+(?:{positive_aux}))?",
        "{negative_subject}){argument}(?:\\s+(?:{negative_aux}|{positive_aux}))?",
    ),
    (
        "collapse-quoted-speech",
        "            clauses.append(span)",
        '            clauses.extend(SpeechSpan(raw, "factual") for raw in _CLAUSES.split(span.text))',
    ),
    (
        "drop-conditional-scope",
        "SpeechSpan(raw, kind) for raw in _CLAUSES.split(proposition)",
        'SpeechSpan(raw, "factual") for raw in _CLAUSES.split(proposition)',
    ),
    ("collapse-action-instances", "if len(candidates) != 1:", "if False:"),
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
    ("ignore-complete-grammar", "if not known:", "if False:"),
    (
        "ignore-channel-ambiguity",
        "if claim.channel is None and any(",
        "if False and any(",
    ),
    (
        "verification-reference-without-proof",
        "Verdict.PASS if proof is not None else Verdict.INCONCLUSIVE,",
        "Verdict.PASS,",
    ),
]
MUTATIONS.extend(
    [
        (
            "ignore-state-envelope",
            "            and _state_envelope_coherent(data)\n",
            "",
        ),
        (
            "remove-read-only-verification",
            "        and not lookup.state_changing\n",
            "",
        ),
        (
            "remove-verification-operation",
            '        and data.get("operation") == proof.tool_name\n',
            "",
        ),
        (
            "discover-state-by-tool-name",
            "        data = observed.result",
            '        if observed.tool_name not in {"move_message", "verify_sent_message", "send_email", "send_mail"}:\n            continue\n        data = observed.result',
        ),
        (
            "discover-state-by-message-id-only",
            '_OBJECT_KEYS = ("message_id", "draft_id", "client_message_id")',
            '_OBJECT_KEYS = ("message_id",)',
        ),
        (
            "object-identity-without-operation-authority",
            "            and protocol\n",
            "",
        ),
        (
            "collapse-reported-ownership",
            "                if not conditional and re.search(",
            "                if False and re.search(",
        ),
    ]
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = ROOT / "agentcheck/evaluate/claims.py"
    original = path.read_text()
    rows = []
    for name, old, new in MUTATIONS:
        if original.count(old) != 1:
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
            "classification": "KILLED"
            if killed
            else "SURVIVED"
            if completed.returncode == 0
            else "ERROR",
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
