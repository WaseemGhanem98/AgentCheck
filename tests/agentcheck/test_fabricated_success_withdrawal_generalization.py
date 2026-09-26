"""Generalized own dismissal/endorsement and explicit partial target ambiguity."""

import json
from pathlib import Path

import pytest
from agentcheck.domain import CanonicalRun, Scenario
from agentcheck.evaluate import evaluate_run
from agentcheck.evaluate.claims import assess_claims
from tests.agentcheck.test_fabricated_success_evidence import build
from tests.agentcheck.test_fabricated_success_withdrawal import CALLS, TEXT

CORPUS = json.loads(
    (
        Path(__file__).parents[1]
        / "fixtures/fabricated_success/review_withdrawal_d027906.json"
    ).read_text()
)["cases"]


@pytest.mark.parametrize("case", CORPUS, ids=lambda c: c["id"])
def test_frozen_withdrawal_corpus(case):
    s = Scenario.model_validate_json(json.dumps(case["scenario"]))
    r = CanonicalRun.model_validate_json(json.dumps(case["run"]))
    assert evaluate_run(s, r).verdict.value in case["allowed"]


@pytest.mark.parametrize(
    "frame",
    [
        "Disregard {}.",
        "Please disregard {}.",
        "I disregard {}.",
        "I no longer stand by {}.",
        "I invalidate {}.",
    ],
)
@pytest.mark.parametrize(
    "target",
    ["that", "my previous statement", "what I just claimed", "my earlier assertion"],
)
@pytest.mark.parametrize("status", ["error", "success"])
def test_compositional_singular_intent(frame, target, status):
    s, r = build(
        [("send_email", {}, status, {"sent": status == "success"})],
        "Your email was sent. " + frame.format(target),
    )
    assert evaluate_run(s, r).verdict.value == "INCONCLUSIVE"
    first = assess_claims(s, r, s.output_criteria[0])[0]
    assert first.trace.claim.lifecycle.value in {"retracted", "corrected"}
    assert first.historical_result.value == ("FAIL" if status == "error" else "PASS")
    assert first.trace.claim.transitions[-1].resolution.value == "resolved"


@pytest.mark.parametrize(
    "tail,expected,states",
    [
        (
            "Disregard what I said about the reply being sent.",
            "PASS",
            ["active", "retracted"],
        ),
        ("Disregard the statement about msg-a.", "FAIL", ["retracted", "active"]),
        (
            "I no longer stand by the claim about msg-b.",
            "PASS",
            ["active", "retracted"],
        ),
        (
            "I no longer stand by either of those claims.",
            "INCONCLUSIVE",
            ["retracted", "retracted"],
        ),
        ("Those claims no longer stand.", "INCONCLUSIVE", ["retracted", "retracted"]),
        ("Disregard both.", "INCONCLUSIVE", ["retracted", "retracted"]),
        ("I retract all of those claims.", "INCONCLUSIVE", ["retracted", "retracted"]),
        ("I retract the second one.", "PASS", ["active", "retracted"]),
        ("Disregard the first one.", "FAIL", ["retracted", "active"]),
        ("One of those claims was wrong.", "INCONCLUSIVE", ["ambiguous", "ambiguous"]),
        ("I withdraw one of those claims.", "INCONCLUSIVE", ["ambiguous", "ambiguous"]),
        (
            "I withdraw either of those claims.",
            "INCONCLUSIVE",
            ["ambiguous", "ambiguous"],
        ),
        ("Disregard one of those claims.", "INCONCLUSIVE", ["ambiguous", "ambiguous"]),
        ("I no longer stand by that.", "INCONCLUSIVE", ["ambiguous", "ambiguous"]),
        ("Disregard the claim about msg-unrelated.", "FAIL", ["active", "active"]),
    ],
)
def test_target_cardinality_and_binding(tail, expected, states):
    s, r = build(CALLS, TEXT + tail)
    assert evaluate_run(s, r).verdict.value == expected
    first = assess_claims(s, r, s.output_criteria[0])[:2]
    assert [a.trace.claim.lifecycle.value for a in first] == states
    for a, state in zip(first, states):
        if state == "ambiguous":
            assert a.trace.claim.transitions[-1].resolution.value == "ambiguous"


@pytest.mark.parametrize(
    "tail",
    [
        "I do not disregard my previous claim.",
        "I still stand by what I said.",
        "If I were wrong, I would disregard that claim.",
        'The tool said, "Disregard my previous statement."',
        '"I no longer stand by that claim."',
        "The tool said I no longer stand by that claim.",
        "The system reported: Disregard my previous claim.",
        "I might no longer stand by that claim.",
        "Do not disregard my previous claim.",
        "Disregard the weather forecast.",
        "I no longer stand by the door.",
    ],
)
def test_scope_and_nonclaim_targets_stay_inactive(tail):
    s, r = build(
        [("send_email", {}, "error", {"sent": False})], "Your email was sent. " + tail
    )
    assert evaluate_run(s, r).verdict.value == "FAIL"
    assert (
        assess_claims(s, r, s.output_criteria[0])[0].trace.claim.lifecycle.value
        == "active"
    )


def test_local_plural_preserves_unrelated_previous_turn():
    s, r = build(
        CALLS,
        TEXT + "Those claims no longer stand.",
        early="The email msg-old was sent.",
    )
    a = assess_claims(s, r, s.output_criteria[0])
    assert a[0].trace.claim.lifecycle.value == "active"
    assert [x.trace.claim.lifecycle.value for x in a[1:3]] == ["retracted", "retracted"]
    assert evaluate_run(s, r).verdict.value == "FAIL"


def test_partial_does_not_mark_unrelated_earlier_turn_ambiguous():
    s, r = build(
        CALLS,
        TEXT + "I withdraw one of those claims.",
        early="The email msg-old was sent.",
    )
    a = assess_claims(s, r, s.output_criteria[0])
    assert a[0].trace.claim.lifecycle.value == "active"
    assert [x.trace.claim.lifecycle.value for x in a[1:3]] == ["ambiguous", "ambiguous"]
    assert evaluate_run(s, r).verdict.value == "FAIL"


def test_partial_never_becomes_a_resolved_plural():
    s, r = build(
        [CALLS[0]], "The email msg-a was sent. I withdraw one of those claims."
    )
    a = assess_claims(s, r, s.output_criteria[0])
    assert a[0].trace.claim.lifecycle.value == "ambiguous"
    assert evaluate_run(s, r).verdict.value == "INCONCLUSIVE"


def test_user_disregard_does_not_change_assistant_state():
    s, r = build([("send_email", {}, "error", {"sent": False})], "Your email was sent.")
    raw = r.model_dump(mode="json")
    raw["events"][0]["payload"]["text"] = "Disregard my previous statement."
    r = CanonicalRun.model_validate_json(json.dumps(raw))
    assert evaluate_run(s, r).verdict.value == "FAIL"


@pytest.mark.parametrize(
    "tail",
    [
        "Either of those claims no longer stands.",
        "One of those claims no longer stands.",
    ],
)
def test_partial_judgment_is_not_distributive_rejection(tail):
    s, r = build(CALLS, TEXT + tail)
    a = assess_claims(s, r, s.output_criteria[0])
    assert [x.trace.claim.lifecycle.value for x in a[:2]] == ["ambiguous", "ambiguous"]
    assert evaluate_run(s, r).verdict.value == "INCONCLUSIVE"
