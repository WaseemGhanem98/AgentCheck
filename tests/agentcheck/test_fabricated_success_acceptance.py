"""Independent acceptance witnesses: reaffirmation preserves proposition semantics."""

import json
from pathlib import Path

import pytest
from agentcheck.domain import CanonicalRun, Scenario
from agentcheck.evaluate import evaluate_run
from agentcheck.evaluate.claims import assess_claims

CASES = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/fabricated_success/review_5d4d1fe.json"
    ).read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_independent_acceptance(case):
    s = Scenario.model_validate_json(json.dumps(case["scenario"]))
    r = CanonicalRun.model_validate_json(json.dumps(case["run"]))
    assert evaluate_run(s, r).verdict.value in case["allowed"]
    claims = assess_claims(s, r, s.output_criteria[0])
    history = {a.trace.claim.claim_id: a.trace.claim for a in claims}
    if case["id"] == "different-aspects-ambiguous":
        assert all(a.trace.claim.lifecycle.value == "ambiguous" for a in claims), (
            "Distinct proposition aspects must leave the confirmation target ambiguous"
        )
    for a in claims:
        c = a.trace.claim
        if (
            c.relation
            and c.relation.value == "confirms"
            and c.lifecycle.value == "active"
        ):
            assert c.aspect == history[c.antecedents[0]].aspect
