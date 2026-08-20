from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from agentcheck.redaction import DEFAULT_REDACTED_KEYS, redact_text, sanitize_value


# These are evaluation resource metrics, not authentication material. AgentLens
# already exempts its plural token counters; AgentCheck adds its one singular
# schema field so versioned scenarios remain valid after privacy filtering.
_NON_SECRET_AGENTCHECK_KEYS = frozenset({"token_budget"})
_MAX_ARTIFACT_DEPTH = 20
_MAX_ARTIFACT_ITEMS = 100
_MAX_ARTIFACT_NODES = 100_000


def _redact_artifact(
    value: Any,
    *,
    keys: tuple[str, ...],
    additional_keys: tuple[str, ...],
    depth: int,
    seen: set[int],
    nodes: list[int],
) -> Any:
    if depth > _MAX_ARTIFACT_DEPTH:
        return "[MAX_DEPTH]"
    if nodes[0] <= 0:
        return "[TRUNCATED_NODES]"
    nodes[0] -= 1
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in seen:
            return "[CYCLE]"
        seen.add(value_id)
        try:
            result: dict[str, Any] = {}
            for index, (raw_key, item) in enumerate(value.items()):
                if index >= _MAX_ARTIFACT_ITEMS:
                    result["[TRUNCATED_ITEMS]"] = True
                    break
                # A singleton placeholder lets AgentLens own key normalization
                # and sensitivity classification.
                if (
                    isinstance(raw_key, str)
                    and raw_key in _NON_SECRET_AGENTCHECK_KEYS
                ):
                    safe_key = str(sanitize_value(raw_key, redacted_keys=keys))
                    marker = None
                else:
                    safe_pair = sanitize_value({raw_key: None}, redacted_keys=keys)
                    safe_key, marker = next(iter(safe_pair.items()))
                result[safe_key] = (
                    marker
                    if marker == "[REDACTED]"
                    else _redact_artifact(
                        item,
                        keys=keys,
                        additional_keys=additional_keys,
                        depth=depth + 1,
                        seen=seen,
                        nodes=nodes,
                    )
                )
            return result
        finally:
            seen.discard(value_id)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        value_id = id(value)
        if value_id in seen:
            return "[CYCLE]"
        seen.add(value_id)
        try:
            result_list = [
                _redact_artifact(
                    item,
                    keys=keys,
                    additional_keys=additional_keys,
                    depth=depth + 1,
                    seen=seen,
                    nodes=nodes,
                )
                for item in value[:_MAX_ARTIFACT_ITEMS]
            ]
            if len(value) > _MAX_ARTIFACT_ITEMS:
                result_list.append("[TRUNCATED_ITEMS]")
            return result_list
        finally:
            seen.discard(value_id)
    return sanitize_value(value, redacted_keys=keys)


def redact_artifact(value: Any, *, additional_keys: tuple[str, ...] = ()) -> Any:
    """Apply AgentLens redaction without one shared budget truncating a whole suite.

    AgentLens telemetry intentionally has a small per-event node budget.
    Evaluation artifacts contain collections of independent cases, so this
    layer retains its key/text rules while applying a larger, depth- and
    item-bounded artifact budget.
    """

    keys = tuple(dict.fromkeys((*DEFAULT_REDACTED_KEYS, *additional_keys)))
    return _redact_artifact(
        value,
        keys=keys,
        additional_keys=additional_keys,
        depth=0,
        seen=set(),
        nodes=[_MAX_ARTIFACT_NODES],
    )


def redact_log_text(value: str) -> str:
    """Redact free-form worker diagnostics before they enter artifacts or console output."""

    return redact_text(value)


def redact_model_text(value: str, *, max_chars: int = 4_000) -> str:
    """Bound and redact untrusted model output before it is stored or displayed."""

    if max_chars < 1:
        return ""
    return redact_log_text(value)[:max_chars]
