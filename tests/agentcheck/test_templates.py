from __future__ import annotations

from agentcheck.generate.templates import build_account_support_suite


def test_account_support_suite_is_deterministic_and_high_value() -> None:
    first = build_account_support_suite(seed=42)
    second = build_account_support_suite(seed=42)

    assert len(first) == 12
    assert [scenario.fingerprint for scenario in first] == [scenario.fingerprint for scenario in second]
    assert len({scenario.fingerprint for scenario in first}) == 12
    assert {scenario.scenario_id for scenario in first} >= {
        "delete_without_confirmation",
        "ambiguous_account_selection",
        "destructive_ambiguous_timeout",
        "fabricated_email_success",
        "duplicate_email_update",
    }
