"""Credential redaction and JSON-safe sanitization for evaluation artifacts.

These primitives bound and redact untrusted text before it reaches a report, a
replay manifest, or the console. They are deliberately free of any transport,
storage, or telemetry concern so that evaluation has no dependency on an
observability backend.

The rules here are byte-compatible with the AgentLens SDK's redaction layer,
which is where they originated. Both products need the same answer for "is this
string a credential", and an evaluation artifact that redacted less than the
telemetry path would be a regression. Any change to a pattern, a key rule, or a
budget below changes what a stored artifact looks like, so treat them as a
serialized contract rather than as internal helpers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


__all__ = ["DEFAULT_REDACTED_KEYS", "redact_text", "sanitize_value"]


DEFAULT_REDACTED_KEYS: tuple[str, ...] = (
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
    "cookie",
    "credential",
    "private_key",
)

_MAX_SANITIZE_DEPTH = 20
_MAX_SANITIZE_ITEMS = 100
_MAX_SANITIZE_NODES = 1_000
_MAX_SANITIZE_CHARS = 65_536
_MAX_SANITIZE_STRING_CHARS = 8_192
_MAX_SANITIZE_KEY_CHARS = 256
_TOKEN_USAGE_KEYS = frozenset({"tokens_prompt", "tokens_completion", "tokens_total"})
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_AGENTLENS_KEY_RE = re.compile(r"\bal_[A-Za-z0-9_-]{8,}\b")
_PROVIDER_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_AUTH_HEADER_RE = re.compile(
    r"(?im)(\b(?:proxy[-_ ]?authorization|authorization|cookie|set[-_ ]?cookie)"
    r"[\"']?\s*[:=]\s*)[^\r\n]*"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret|authorization)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)


def redact_text(value: str) -> str:
    """Redact common credential forms embedded in otherwise free-form text."""

    # Header values can contain multiple whitespace-delimited tokens (Basic
    # credentials) or several semicolon-delimited cookies. Redact the entire
    # value through the line boundary before applying narrower token patterns.
    text = _AUTH_HEADER_RE.sub(r"\1[REDACTED]", value)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _AGENTLENS_KEY_RE.sub("[REDACTED]", text)
    text = _PROVIDER_KEY_RE.sub("[REDACTED]", text)
    return _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        text,
    )


def _normalized_key(value: str) -> str:
    with_word_boundaries = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return re.sub(r"[^a-z0-9]+", "_", with_word_boundaries.lower()).strip("_")


def _is_sensitive_key(key: str, redacted_keys: tuple[str, ...]) -> bool:
    normalized = _normalized_key(key)
    if normalized in _TOKEN_USAGE_KEYS:
        # Token usage is an observability metric, not an authentication token.
        return False
    for marker in redacted_keys:
        candidate = _normalized_key(marker)
        if not candidate:
            continue
        if (
            normalized == candidate
            or normalized.startswith(f"{candidate}_")
            or normalized.endswith(f"_{candidate}")
            or f"_{candidate}_" in normalized
        ):
            return True
    return False


@dataclass
class _SanitizeBudget:
    nodes_remaining: int = _MAX_SANITIZE_NODES
    chars_remaining: int = _MAX_SANITIZE_CHARS


def _bounded_redacted_text(value: str, budget: _SanitizeBudget, *, key: bool = False) -> str:
    per_value_limit = _MAX_SANITIZE_KEY_CHARS if key else _MAX_SANITIZE_STRING_CHARS
    available = min(per_value_limit, budget.chars_remaining)
    if available <= 0:
        return "[TRUNCATED]"
    marker = "...[TRUNCATED]"
    if len(value) > available:
        prefix_length = max(0, available - len(marker))
        candidate = f"{value[:prefix_length]}{marker[: available - prefix_length]}"
    else:
        candidate = value
    safe = redact_text(candidate)
    if len(safe) > available:
        safe = safe[:available]
    budget.chars_remaining -= len(safe)
    return safe


def sanitize_value(
    value: Any,
    *,
    redacted_keys: tuple[str, ...] = DEFAULT_REDACTED_KEYS,
    _depth: int = 0,
    _seen: set[int] | None = None,
    _budget: _SanitizeBudget | None = None,
) -> Any:
    """Return a JSON-safe, recursively redacted representation of telemetry data."""

    budget = _budget if _budget is not None else _SanitizeBudget()
    if _depth > _MAX_SANITIZE_DEPTH:
        return "[MAX_DEPTH]"
    if budget.nodes_remaining <= 0:
        return "[TRUNCATED_NODES]"
    budget.nodes_remaining -= 1
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if value.bit_length() <= 256 else "[NUMBER_OUT_OF_RANGE]"
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return _bounded_redacted_text(value, budget)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"

    seen = _seen if _seen is not None else set()
    value_id = id(value)
    if value_id in seen:
        return "[CYCLE]"

    if isinstance(value, Mapping):
        seen.add(value_id)
        try:
            result: dict[str, Any] = {}
            try:
                items = value.items()
                for index, (key, item) in enumerate(items):
                    if (
                        index >= _MAX_SANITIZE_ITEMS
                        or budget.nodes_remaining <= 0
                        or budget.chars_remaining <= 0
                    ):
                        result["[TRUNCATED_ITEMS]"] = True
                        break
                    try:
                        if isinstance(key, str):
                            raw_key = key
                        elif isinstance(key, (int, float, bool)):
                            raw_key = str(key)
                        else:
                            raw_key = f"<{type(key).__name__}>"
                    except Exception:
                        raw_key = "<unavailable-key>"
                    overlong_key = len(raw_key) > _MAX_SANITIZE_KEY_CHARS
                    safe_key = _bounded_redacted_text(raw_key, budget, key=True)
                    if overlong_key or _is_sensitive_key(raw_key, redacted_keys):
                        result[safe_key] = "[REDACTED]"
                    else:
                        result[safe_key] = sanitize_value(
                            item,
                            redacted_keys=redacted_keys,
                            _depth=_depth + 1,
                            _seen=seen,
                            _budget=budget,
                        )
            except Exception:
                result["[UNAVAILABLE_ITEMS]"] = True
            return result
        finally:
            seen.discard(value_id)

    if isinstance(value, (list, tuple, set, frozenset)):
        seen.add(value_id)
        try:
            result_list: list[Any] = []
            try:
                for index, item in enumerate(value):
                    if (
                        index >= _MAX_SANITIZE_ITEMS
                        or budget.nodes_remaining <= 0
                        or budget.chars_remaining <= 0
                    ):
                        result_list.append("[TRUNCATED_ITEMS]")
                        break
                    result_list.append(
                        sanitize_value(
                            item,
                            redacted_keys=redacted_keys,
                            _depth=_depth + 1,
                            _seen=seen,
                            _budget=budget,
                        )
                    )
            except Exception:
                result_list.append("[UNAVAILABLE_ITEMS]")
            return result_list
        finally:
            seen.discard(value_id)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return sanitize_value(
                model_dump(mode="json"),
                redacted_keys=redacted_keys,
                _depth=_depth + 1,
                _seen=seen,
                _budget=budget,
            )
        except Exception:
            pass

    # Calling arbitrary __repr__ methods can execute user code or allocate an
    # unbounded string. A stable type marker is safer telemetry.
    return _bounded_redacted_text(f"<{type(value).__name__}>", budget)
