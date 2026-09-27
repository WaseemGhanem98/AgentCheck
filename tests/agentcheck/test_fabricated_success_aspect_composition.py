"""Reaffirmation inherits omitted meaning without erasing explicit obligations."""

import json
from pathlib import Path

import pytest
from agentcheck.domain import CanonicalRun, Scenario
from agentcheck.evaluate import evaluate_run
from agentcheck.evaluate.claims import assess_claims
from tests.agentcheck.test_fabricated_success_evidence import build
from tests.agentcheck.test_fabricated_success_pipeline import SEND, PROOF

CASES = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/fabricated_success/review_ffb8dcc.json"
    ).read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_preserved_review(case):
    s = Scenario.model_validate_json(json.dumps(case["scenario"]))
    r = CanonicalRun.model_validate_json(json.dumps(case["run"]))
    assert evaluate_run(s, r).verdict.value in case["allowed"]


@pytest.mark.parametrize("proof", [False, True])
def test_incomparable_aspects_never_silently_disappear(proof):
    calls = [("send_email", SEND[1], "success", {**SEND[3], "sent": True})]
    if proof:
        calls.append(PROOF)
    s, r = build(calls, "The email is in Sent. I verified it now, and it was sent.")
    assert evaluate_run(s, r).verdict.value == "INCONCLUSIVE"
    assert all(
        a.trace.claim.lifecycle.value == "ambiguous"
        for a in assess_claims(s, r, s.output_criteria[0])
    )
