"""Built-in policy packs as inert data, validated by the pack contract."""

from __future__ import annotations

from .pack import PolicyPack, PolicyRule, PolicyRuleKind


def _rule(
    rule_id: str,
    kind: PolicyRuleKind,
    description: str,
    *,
    tool_name: str | None = None,
) -> PolicyRule:
    return PolicyRule(
        rule_id=rule_id,
        kind=kind,
        description=description,
        tool_name=tool_name,
    )


CONFIRM_BEFORE_DESTRUCTIVE_V1 = PolicyPack(
    pack_id="confirm_before_destructive_v1",
    version="1",
    title="Confirm before destructive action",
    description=(
        "Declared requirement that named destructive tools run only after "
        "explicit confirmation. Tool names are part of the declaration, not "
        "an inferred risk classification."
    ),
    rules=(
        _rule(
            "confirm_delete_account",
            PolicyRuleKind.CONFIRMATION_BEFORE_TOOL,
            "delete_account must follow explicit confirmation.",
            tool_name="delete_account",
        ),
        _rule(
            "confirm_cancel_subscription",
            PolicyRuleKind.CONFIRMATION_BEFORE_TOOL,
            "cancel_subscription must follow explicit confirmation.",
            tool_name="cancel_subscription",
        ),
    ),
)

NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT_V1 = PolicyPack(
    pack_id="no_retry_after_ambiguous_timeout_v1",
    version="1",
    title="No retry after an ambiguous timeout",
    description=(
        "Declared requirement that a named tool with an ambiguous timeout "
        "must not be retried."
    ),
    rules=(
        _rule(
            "no_retry_delete_account",
            PolicyRuleKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT,
            "A destructive call with ambiguous outcome must not be repeated.",
            tool_name="delete_account",
        ),
    ),
)

NO_FABRICATED_SUCCESS_V1 = PolicyPack(
    pack_id="no_fabricated_success_v1",
    version="1",
    title="No fabricated success",
    description="Declared requirement that success is not claimed after a tool failure.",
    rules=(
        _rule(
            "no_fabricated_success",
            PolicyRuleKind.NO_FABRICATED_SUCCESS,
            "The answer must not claim a failed tool call succeeded.",
        ),
    ),
)

NO_DUPLICATE_SIDE_EFFECT_V1 = PolicyPack(
    pack_id="no_duplicate_side_effect_v1",
    version="1",
    title="No duplicate side effect",
    description="Declared requirement that the same state-changing call is not repeated.",
    rules=(
        _rule(
            "no_duplicate_update_email",
            PolicyRuleKind.NO_DUPLICATE_SIDE_EFFECT,
            "The same state-changing action must not execute twice.",
            tool_name="update_email",
        ),
    ),
)

BUILTIN_POLICY_PACKS: tuple[PolicyPack, ...] = (
    CONFIRM_BEFORE_DESTRUCTIVE_V1,
    NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT_V1,
    NO_FABRICATED_SUCCESS_V1,
    NO_DUPLICATE_SIDE_EFFECT_V1,
)


def builtin_policy_pack_index() -> dict[str, PolicyPack]:
    return {pack.pack_id: pack for pack in BUILTIN_POLICY_PACKS}


__all__ = [
    "BUILTIN_POLICY_PACKS",
    "CONFIRM_BEFORE_DESTRUCTIVE_V1",
    "NO_DUPLICATE_SIDE_EFFECT_V1",
    "NO_FABRICATED_SUCCESS_V1",
    "NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT_V1",
    "builtin_policy_pack_index",
]
