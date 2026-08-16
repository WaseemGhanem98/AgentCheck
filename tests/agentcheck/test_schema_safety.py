from __future__ import annotations

import urllib.request

import pytest
from jsonschema import ValidationError  # type: ignore[import-untyped]

from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator


@pytest.mark.parametrize(
    "reference",
    (
        "https://example.invalid/schema.json",
        "file:///etc/passwd",
        "relative-schema.json",
    ),
)
def test_external_schema_references_are_rejected_without_retrieval(
    reference: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieved: list[str] = []

    def tripwire(url: str, *args: object, **kwargs: object) -> None:
        del args, kwargs
        retrieved.append(url)
        raise AssertionError("schema retrieval must never run")

    monkeypatch.setattr(urllib.request, "urlopen", tripwire)

    with pytest.raises(UnsafeSchemaReference, match="local fragment"):
        offline_validator({"$ref": reference})

    assert retrieved == []


def test_local_schema_references_remain_supported_offline() -> None:
    validator = offline_validator(
        {
            "$defs": {"account_id": {"type": "string", "pattern": "^acct_[0-9]+$"}},
            "type": "object",
            "properties": {"account_id": {"$ref": "#/$defs/account_id"}},
            "required": ["account_id"],
        }
    )

    assert list(validator.iter_errors({"account_id": "acct_123"})) == []
    with pytest.raises(ValidationError):
        validator.validate({"account_id": "wrong"})
