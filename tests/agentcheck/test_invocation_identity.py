"""Duplicate and retry detection must compare calls semantically.

Grounded in a real historical bug: the OpenAI Agents SDK computed tool
invocation identity without excluding ``None`` optional fields, so two
logically identical invocations produced different identities. Maintainers
fixed it in openai/openai-agents-python#4289 and added a regression test.

AgentCheck's duplicate-side-effect check compared raw arguments and shared the
flaw. That is a false negative in a safety check: a second destructive call
that merely spells an optional field as ``null`` looked like a different action,
so a repeated side effect went unreported.
"""

from __future__ import annotations

from agentcheck.evaluate.engine import _invocation_identity, _semantic_arguments


def test_omitted_and_explicit_null_optional_are_the_same_call() -> None:
    """The historical bug: these request identical work and must collide."""

    assert _invocation_identity(
        "cancel_flight", {"confirmation_number": "ABC123"}
    ) == _invocation_identity(
        "cancel_flight", {"confirmation_number": "ABC123", "reason": None}
    )


def test_normalization_reaches_nested_values() -> None:
    assert _invocation_identity(
        "book_flight", {"passenger": {"seat": "1A"}, "extras": [{"bag": "carry_on"}]}
    ) == _invocation_identity(
        "book_flight",
        {"passenger": {"seat": "1A", "meal": None}, "extras": [{"bag": "carry_on", "note": None}]},
    )


def test_key_order_does_not_change_identity() -> None:
    assert _invocation_identity(
        "update_email", {"account_id": "a1", "new_email": "x@example.com"}
    ) == _invocation_identity(
        "update_email", {"new_email": "x@example.com", "account_id": "a1"}
    )


def test_genuinely_different_calls_stay_distinct() -> None:
    """Normalizing must not collapse actions that really differ.

    Over-matching would be worse than the original bug: it would hide a second,
    legitimately different side effect behind the first.
    """

    base = _invocation_identity("cancel_flight", {"confirmation_number": "ABC123"})

    assert base != _invocation_identity("cancel_flight", {"confirmation_number": "XYZ789"})
    assert base != _invocation_identity("book_flight", {"confirmation_number": "ABC123"})
    # An optional field carrying a real value is a different request than omitting it.
    assert base != _invocation_identity(
        "cancel_flight", {"confirmation_number": "ABC123", "reason": "weather"}
    )


def test_false_is_preserved_because_it_is_a_value_not_an_absence() -> None:
    """Only ``None`` means "not supplied"; falsey values are real arguments."""

    assert _semantic_arguments({"force": False}) == {"force": False}
    assert _semantic_arguments({"count": 0, "note": ""}) == {"count": 0, "note": ""}
    assert _semantic_arguments({"force": None}) == {}


def test_identity_is_stable_across_calls() -> None:
    """Deterministic: the same arguments always yield the same identity string."""

    arguments = {"b": [3, None, 1], "a": {"z": None, "y": 2}}

    assert _invocation_identity("t", arguments) == _invocation_identity("t", arguments)
