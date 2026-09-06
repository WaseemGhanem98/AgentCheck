"""Equality for already validated JSON values, not argument matching topology."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def json_values_equal(left: Any, right: Any) -> bool:
    """Keep JSON booleans distinct from numbers, including inside containers.

    Equal mathematical numbers (such as 1 and 1.0) remain equal. Objects are
    complete values and arrays are ordered; callers own any subset semantics.
    This is not validation or coercion of arbitrary Python objects.
    """

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(json_values_equal(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, list):
        return (
            isinstance(right, list)
            and len(left) == len(right)
            and all(json_values_equal(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)
