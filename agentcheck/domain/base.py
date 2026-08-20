"""Shared primitives for versioned AgentCheck domain contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, Union

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict
from typing_extensions import Annotated, TypeAliasType


def _as_utc(value: datetime) -> datetime:
    """Normalize an already timezone-aware datetime to UTC."""

    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_as_utc)]

if TYPE_CHECKING:
    # Mypy cannot yet resolve TypeAliasType's recursive runtime definition.
    JsonValue: TypeAlias = Any
else:
    JsonValue = TypeAliasType(
        "JsonValue",
        Union[
            None,
            bool,
            int,
            float,
            str,
            list["JsonValue"],
            dict[str, "JsonValue"],
        ],
    )
JsonObject: TypeAlias = dict[str, JsonValue]

ContractT = TypeVar("ContractT", bound="ContractModel")


class ContractModel(BaseModel):
    """Strict base with stable, cross-process JSON serialization."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        allow_inf_nan=False,
    )

    def canonical_json(self, *, exclude: set[str] | None = None) -> str:
        """Return canonical JSON suitable for hashing and JSONL artifacts."""

        return json.dumps(
            self.model_dump(mode="json", exclude=exclude),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def content_hash(self, *, exclude: set[str] | None = None) -> str:
        payload = self.canonical_json(exclude=exclude).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    @classmethod
    def from_json(cls: type[ContractT], payload: str | bytes | bytearray) -> ContractT:
        return cls.model_validate_json(payload)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_hash(value: Any) -> str:
    """Hash an arbitrary JSON-compatible value with the contract encoding."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
