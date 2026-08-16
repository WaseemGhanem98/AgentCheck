"""Consent-gated, non-authoritative LLM realization of display text.

Realization may rewrite ``title`` and ``description``, which are excluded from
``agentcheck.scenario.v1`` fingerprints, and may store display-only turn
phrasing beside the scenario. It never rewrites conversation turns, fixtures,
oracles, criteria, budgets, or dimension tags, and it cannot construct a
hard-failure oracle. Conversation turn text stays on v1 and participates in the
fingerprint; this slice deliberately does not introduce ``scenario.v2``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Literal, Protocol, Sequence

from pydantic import Field

from agentcheck.config import AgentCheckConfig
from agentcheck.domain import ContractModel, Scenario
from agentcheck.errors import ConfigurationError
from agentcheck.privacy import redact_log_text, redact_model_text


REALIZATION_PROVIDER = "openai"
REALIZATION_CREDENTIAL_ENV = "OPENAI_API_KEY"
MAX_REALIZATION_CALLS = 32
MAX_RETRIES = 2
MAX_PROMPT_CHARS = 4_000
MAX_RESPONSE_CHARS = 4_000
MAX_TITLE_CHARS = 500
MAX_DESCRIPTION_CHARS = 8_000
MAX_TURN_CHARS = 2_000
_DEFAULT_MODEL = "gpt-4o-mini"
_UNSAFE_PATH = re.compile(r"(?:^|[\s'\"`])(?:\.\.|/etc/|/tmp/|[A-Za-z]:\\)")
_UNSAFE_CODE = re.compile(
    r"\b(?:import\s+os|subprocess|eval\s*\(|exec\s*\(|__import__)\b",
    re.IGNORECASE,
)


class RealizationRequest(ContractModel):
    scenario_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    description: str = Field(default="", max_length=MAX_DESCRIPTION_CHARS)
    turns: tuple[str, ...] = Field(min_length=1)


class RealizationResult(ContractModel):
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    description: str = Field(default="", max_length=MAX_DESCRIPTION_CHARS)
    turns: tuple[str, ...] = ()


class RealizationRecord(ContractModel):
    """Non-authoritative display overlay. Not part of the scenario fingerprint."""

    source_kind: Literal["llm_inference"] = "llm_inference"
    inferred: Literal[True] = True
    authoritative: Literal[False] = False
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    description: str = Field(default="", max_length=MAX_DESCRIPTION_CHARS)
    turns: tuple[str, ...] = ()


class LlmRealizer(Protocol):
    def realize(self, request: RealizationRequest) -> RealizationResult:
        """Return untrusted display text or raise on provider failure."""


class DisabledRealizer:
    """Default realizer. Invoking it means consent was bypassed."""

    def realize(self, request: RealizationRequest) -> RealizationResult:
        raise ConfigurationError("LLM realization is disabled")


CompleteFn = Callable[[str], str]


class OpenAIChatRealizer:
    """OpenAI Chat Completions realizer. Network is used only after consent."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        complete: CompleteFn | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 1,
    ) -> None:
        if not api_key.strip():
            raise ConfigurationError(f"{REALIZATION_CREDENTIAL_ENV} is not set")
        if max_retries < 0 or max_retries > MAX_RETRIES:
            raise ConfigurationError(f"max_retries must be between 0 and {MAX_RETRIES}")
        self._api_key = api_key
        self._model = model
        self._complete = complete or (lambda prompt: complete_chat(
            prompt, api_key=api_key, model=model, timeout_seconds=timeout_seconds
        ))
        self._max_retries = max_retries
        self.calls = 0

    def realize(self, request: RealizationRequest) -> RealizationResult:
        prompt = _prompt_for(request)
        last_error = "provider returned no usable realization"
        attempts = self._max_retries + 1
        for _ in range(attempts):
            self.calls += 1
            try:
                raw = self._complete(prompt)
            except Exception as exc:
                last_error = str(exc)[:400]
                continue
            try:
                return parse_realization_payload(raw, expected_turns=len(request.turns))
            except ValueError as exc:
                last_error = str(exc)[:400]
                continue
        raise ConfigurationError(f"LLM realization failed: {last_error}")


def complete_chat(
    prompt: str,
    *,
    api_key: str,
    model: str,
    timeout_seconds: float = 20.0,
) -> str:
    """Single Chat Completions call. Tests replace this to forbid network."""

    import requests

    if len(prompt) > MAX_PROMPT_CHARS:
        raise ConfigurationError("realization prompt exceeds the size bound")
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0,
            "max_tokens": 800,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite only display wording. Return JSON with keys "
                        "title, description, turns. Do not invent tools, paths, "
                        "code, oracles, or secrets."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ConfigurationError("provider response was malformed") from exc
    if not isinstance(content, str):
        raise ConfigurationError("provider response was malformed")
    return content[: MAX_RESPONSE_CHARS + 1]


def require_realization_consent(config: AgentCheckConfig, *, realize: bool) -> None:
    """Fail closed unless CLI flag, config, and allowlisted credential all agree."""

    missing: list[str] = []
    if not realize:
        missing.append("the --realize flag")
    settings = config.llm_realization
    if settings is None or not settings.enabled:
        missing.append("llm_realization.enabled=true in agentcheck.json")
    if REALIZATION_CREDENTIAL_ENV not in config.environment_allowlist:
        missing.append(f"{REALIZATION_CREDENTIAL_ENV} in environment_allowlist")
    if missing:
        raise ConfigurationError(
            "LLM realization requires all of: --realize, "
            "llm_realization.enabled=true, and "
            f"{REALIZATION_CREDENTIAL_ENV} in environment_allowlist "
            f"(missing: {', '.join(missing)})"
        )
    if not (os.environ.get(REALIZATION_CREDENTIAL_ENV) or "").strip():
        raise ConfigurationError(
            f"LLM realization is consented but {REALIZATION_CREDENTIAL_ENV} is not set"
        )


def realization_settings(config: AgentCheckConfig) -> tuple[str, int, int]:
    settings = config.llm_realization
    model = _DEFAULT_MODEL if settings is None else settings.model
    max_calls = 8 if settings is None else settings.max_calls
    max_retries = 1 if settings is None else settings.max_retries
    if max_calls < 1 or max_calls > MAX_REALIZATION_CALLS:
        raise ConfigurationError(
            f"llm_realization.max_calls must be between 1 and {MAX_REALIZATION_CALLS}"
        )
    if max_retries < 0 or max_retries > MAX_RETRIES:
        raise ConfigurationError(
            f"llm_realization.max_retries must be between 0 and {MAX_RETRIES}"
        )
    return model, max_calls, max_retries


def parse_realization_payload(
    raw: str, *, expected_turns: int
) -> RealizationResult:
    text = raw.strip()
    if len(text) > MAX_RESPONSE_CHARS:
        raise ValueError("provider response exceeds the size bound")
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"provider response was not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("provider response must be a JSON object")
    allowed = {"title", "description", "turns"}
    extra = set(document) - allowed
    if extra:
        raise ValueError("provider response contained unsupported fields")
    title = document.get("title")
    description = document.get("description", "")
    turns = document.get("turns", [])
    if not isinstance(title, str) or not title.strip():
        raise ValueError("realized title must be a non-empty string")
    if not isinstance(description, str):
        raise ValueError("realized description must be a string")
    if not isinstance(turns, list) or not all(isinstance(item, str) for item in turns):
        raise ValueError("realized turns must be a list of strings")
    if turns and len(turns) != expected_turns:
        raise ValueError("realized turn count must match the original scenario")
    title = _clean_surface(title, max_chars=MAX_TITLE_CHARS, field="title")
    description = _clean_surface(
        description, max_chars=MAX_DESCRIPTION_CHARS, field="description", allow_empty=True
    )
    cleaned_turns = tuple(
        _clean_surface(item, max_chars=MAX_TURN_CHARS, field="turn") for item in turns
    )
    return RealizationResult(title=title, description=description, turns=cleaned_turns)


def apply_realization(
    scenario: Scenario,
    result: RealizationResult,
    *,
    provider: str,
    model: str,
) -> tuple[Scenario, RealizationRecord]:
    """Rewrite display fields only. Behavioral fingerprint must stay identical."""

    original_fingerprint = scenario.fingerprint
    original_dump = _behavioral_dump(scenario)
    payload = json.loads(scenario.model_dump_json())
    payload["title"] = result.title
    payload["description"] = result.description or None
    payload["fingerprint"] = ""
    updated = Scenario.model_validate_json(json.dumps(payload, ensure_ascii=False))
    if updated.fingerprint != original_fingerprint:
        raise ConfigurationError(
            "realization changed a fingerprint-relevant scenario field"
        )
    if _behavioral_dump(updated) != original_dump:
        raise ConfigurationError(
            "realization changed a fingerprint-relevant scenario field"
        )
    if updated.conversation_turns != scenario.conversation_turns:
        raise ConfigurationError("realization must not rewrite conversation turns")
    if updated.tool_fixtures != scenario.tool_fixtures:
        raise ConfigurationError("realization must not rewrite fixtures")
    if updated.dimension_tags != scenario.dimension_tags:
        raise ConfigurationError("realization must not rewrite dimension tags")
    if updated.oracle_provenance != scenario.oracle_provenance:
        raise ConfigurationError("realization must not rewrite oracles")
    record = RealizationRecord(
        provider=provider,
        model=model,
        title=result.title,
        description=result.description,
        turns=result.turns,
    )
    return updated, record


def realize_scenarios(
    scenarios: Sequence[Scenario],
    realizer: LlmRealizer,
    *,
    provider: str,
    model: str,
    max_calls: int,
) -> tuple[tuple[Scenario, RealizationRecord | None], ...]:
    if max_calls < 1 or max_calls > MAX_REALIZATION_CALLS:
        raise ConfigurationError(
            f"max_calls must be between 1 and {MAX_REALIZATION_CALLS}"
        )
    remaining = max_calls
    realized: list[tuple[Scenario, RealizationRecord | None]] = []
    for scenario in scenarios:
        if remaining <= 0:
            realized.append((scenario, None))
            continue
        request = RealizationRequest(
            scenario_id=scenario.scenario_id,
            title=scenario.title,
            description=scenario.description or "",
            turns=tuple(turn.content for turn in scenario.conversation_turns),
        )
        remaining -= 1
        try:
            result = realizer.realize(request)
            updated, record = apply_realization(
                scenario, result, provider=provider, model=model
            )
        except Exception:
            realized.append((scenario, None))
            continue
        realized.append((updated, record))
    return tuple(realized)


def _behavioral_dump(scenario: Scenario) -> str:
    return scenario.model_dump_json(
        exclude={"fingerprint", "scenario_id", "title", "description"}
    )


def _prompt_for(request: RealizationRequest) -> str:
    payload = json.dumps(
        {
            "title": request.title,
            "description": request.description,
            "turns": list(request.turns),
        },
        ensure_ascii=False,
        allow_nan=False,
    )
    text = (
        "Rewrite this AgentCheck scenario's display wording so it reads naturally. "
        "Keep the same intent. Return JSON {\"title\", \"description\", \"turns\"} "
        f"with exactly {len(request.turns)} turn string(s).\n{payload}"
    )
    return redact_log_text(text)[:MAX_PROMPT_CHARS]


def _clean_surface(
    value: str, *, max_chars: int, field: str, allow_empty: bool = False
) -> str:
    stripped = value.strip()
    redacted = redact_model_text(stripped, max_chars=max_chars)
    if redacted != stripped:
        raise ValueError(f"realized {field} contained a secret-shaped value")
    cleaned = redacted
    if not cleaned and not allow_empty:
        raise ValueError(f"realized {field} is empty after validation")
    if _UNSAFE_PATH.search(cleaned) or _UNSAFE_CODE.search(cleaned):
        raise ValueError(f"realized {field} contained disallowed content")
    if "\x00" in cleaned:
        raise ValueError(f"realized {field} contained a NUL byte")
    return cleaned


__all__ = [
    "MAX_REALIZATION_CALLS",
    "REALIZATION_CREDENTIAL_ENV",
    "REALIZATION_PROVIDER",
    "CompleteFn",
    "DisabledRealizer",
    "LlmRealizer",
    "OpenAIChatRealizer",
    "RealizationRecord",
    "RealizationRequest",
    "RealizationResult",
    "apply_realization",
    "complete_chat",
    "parse_realization_payload",
    "realize_scenarios",
    "realization_settings",
    "require_realization_consent",
]
