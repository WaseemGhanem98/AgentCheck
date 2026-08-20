"""Offline-only JSON Schema validation shared by AgentCheck components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import (  # type: ignore[import-untyped]
    Draft202012Validator,
    FormatChecker,
)
from referencing import Registry


class UnsafeSchemaReference(ValueError):
    """A schema could retrieve a resource outside its own JSON document."""


def ensure_local_schema_references(value: Any, *, path: str = "$") -> None:
    """Reject every non-fragment ``$ref``/``$dynamicRef`` before validation."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in {"$ref", "$dynamicRef"}:
                if not isinstance(item, str) or not item.startswith("#"):
                    raise UnsafeSchemaReference(
                        f"{child_path} must be a local fragment reference"
                    )
            ensure_local_schema_references(item, path=child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            ensure_local_schema_references(item, path=f"{path}[{index}]")


def offline_validator(
    schema: Mapping[str, Any],
    *,
    check_formats: bool = False,
) -> Draft202012Validator:
    """Build a Draft 2020-12 validator that cannot retrieve remote or local files."""

    ensure_local_schema_references(schema)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(
        schema,
        format_checker=FormatChecker() if check_formats else None,
        registry=Registry(),
    )


__all__ = [
    "UnsafeSchemaReference",
    "ensure_local_schema_references",
    "offline_validator",
]
