"""Deterministic, isolated world state for AgentCheck scenarios.

The simulator deliberately supports a very small effect language.  Fixtures may
set, delete, increment, or append values at dot-separated paths (or RFC 6901
style JSON-pointer paths).  State is always copied at the boundary so one case
cannot mutate another case's fixture data by aliasing a nested object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


class WorldStateError(ValueError):
    """Raised when a fixture asks for an invalid or impossible state effect."""


@dataclass(frozen=True)
class WorldTransition:
    """One observable mutation made by the world simulator."""

    operation: str
    path: str
    before: Any
    after: Any
    existed_before: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "path": self.path,
            "before": deepcopy(self.before),
            "after": deepcopy(self.after),
            "existed_before": self.existed_before,
        }


_MISSING = object()


def _decode_pointer_segment(value: str) -> str:
    # RFC 6901 requires ~1 and ~0 to be decoded in that order.
    return value.replace("~1", "/").replace("~0", "~")


def _path_segments(path: str | Sequence[str | int]) -> tuple[str | int, ...]:
    if isinstance(path, str):
        if not path:
            return ()
        if path.startswith("/"):
            return tuple(_decode_pointer_segment(item) for item in path[1:].split("/"))
        segments = tuple(path.split("."))
        if any(not item for item in segments):
            raise WorldStateError(f"invalid empty segment in world path {path!r}")
        return segments
    if isinstance(path, Sequence):
        sequence_segments = tuple(path)
        if any(not isinstance(item, (str, int)) for item in sequence_segments):
            raise WorldStateError("world path segments must be strings or integers")
        return sequence_segments
    raise WorldStateError("world path must be a string or sequence of segments")


def _display_path(path: str | Sequence[str | int]) -> str:
    if isinstance(path, str):
        return path
    return ".".join(str(item) for item in path)


def _list_index(segment: str | int, *, length: int, allow_end: bool = False) -> int:
    if isinstance(segment, bool):
        raise WorldStateError("boolean is not a valid list index")
    try:
        index = int(segment)
    except (TypeError, ValueError) as exc:
        raise WorldStateError(f"expected a list index, got {segment!r}") from exc
    maximum = length if allow_end else length - 1
    if index < 0 or index > maximum:
        raise WorldStateError(f"list index {index} is out of range")
    return index


class WorldSimulator:
    """Own an isolated mutable copy of one scenario's controlled state."""

    def __init__(self, initial_state: Mapping[str, Any] | None = None) -> None:
        if initial_state is not None and not isinstance(initial_state, Mapping):
            raise WorldStateError("initial world state must be a mapping")
        self._initial_state: dict[str, Any] = deepcopy(dict(initial_state or {}))
        self._state: dict[str, Any] = deepcopy(self._initial_state)

    @property
    def initial_state(self) -> dict[str, Any]:
        return deepcopy(self._initial_state)

    @property
    def state(self) -> dict[str, Any]:
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return a copy that callers may safely mutate."""

        return deepcopy(self._state)

    def reset(self) -> None:
        self._state = deepcopy(self._initial_state)

    def _restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore an internal transaction snapshot owned by the gateway."""

        self._state = deepcopy(dict(snapshot))

    def _lookup(self, segments: tuple[str | int, ...]) -> Any:
        current: Any = self._state
        for segment in segments:
            if isinstance(current, Mapping):
                key = str(segment)
                if key not in current:
                    return _MISSING
                current = current[key]
            elif isinstance(current, list):
                try:
                    index = _list_index(segment, length=len(current))
                except WorldStateError:
                    return _MISSING
                current = current[index]
            else:
                return _MISSING
        return current

    def get(self, path: str | Sequence[str | int], default: Any = _MISSING) -> Any:
        segments = _path_segments(path)
        value = self._lookup(segments)
        if value is _MISSING:
            if default is _MISSING:
                raise WorldStateError(
                    f"world path {_display_path(path)!r} does not exist"
                )
            return deepcopy(default)
        return deepcopy(value)

    def exists(self, path: str | Sequence[str | int]) -> bool:
        return self._lookup(_path_segments(path)) is not _MISSING

    def _parent(
        self,
        segments: tuple[str | int, ...],
        *,
        create_missing: bool,
    ) -> tuple[Any, str | int]:
        if not segments:
            raise WorldStateError("an effect path must not target the world root")
        current: Any = self._state
        for segment in segments[:-1]:
            if isinstance(current, dict):
                key = str(segment)
                if key not in current:
                    if not create_missing:
                        raise WorldStateError(
                            f"world path segment {key!r} does not exist"
                        )
                    current[key] = {}
                child = current[key]
            elif isinstance(current, list):
                child = current[_list_index(segment, length=len(current))]
            else:
                raise WorldStateError(
                    f"cannot traverse through non-container world value at {segment!r}"
                )
            if not isinstance(child, (dict, list)):
                raise WorldStateError(
                    f"cannot traverse through non-container world value at {segment!r}"
                )
            current = child
        return current, segments[-1]

    def set(
        self,
        path: str | Sequence[str | int],
        value: Any,
        *,
        create_missing: bool = True,
    ) -> WorldTransition:
        segments = _path_segments(path)
        parent, leaf = self._parent(segments, create_missing=create_missing)
        before: Any = _MISSING
        if isinstance(parent, dict):
            key = str(leaf)
            before = parent.get(key, _MISSING)
            parent[key] = deepcopy(value)
        elif isinstance(parent, list):
            index = _list_index(leaf, length=len(parent))
            before = parent[index]
            parent[index] = deepcopy(value)
        else:  # pragma: no cover - guarded by _parent
            raise WorldStateError("effect parent is not a mutable container")
        return WorldTransition(
            operation="set",
            path=_display_path(path),
            before=None if before is _MISSING else deepcopy(before),
            after=deepcopy(value),
            existed_before=before is not _MISSING,
        )

    def delete(self, path: str | Sequence[str | int]) -> WorldTransition:
        segments = _path_segments(path)
        parent, leaf = self._parent(segments, create_missing=False)
        if isinstance(parent, dict):
            key = str(leaf)
            if key not in parent:
                raise WorldStateError(
                    f"world path {_display_path(path)!r} does not exist"
                )
            before = parent.pop(key)
        elif isinstance(parent, list):
            before = parent.pop(_list_index(leaf, length=len(parent)))
        else:  # pragma: no cover - guarded by _parent
            raise WorldStateError("effect parent is not a mutable container")
        return WorldTransition(
            operation="delete",
            path=_display_path(path),
            before=deepcopy(before),
            after=None,
        )

    def increment(
        self,
        path: str | Sequence[str | int],
        delta: int | float = 1,
    ) -> WorldTransition:
        before = self.get(path)
        if (
            isinstance(before, bool)
            or not isinstance(before, (int, float))
            or isinstance(delta, bool)
            or not isinstance(delta, (int, float))
        ):
            raise WorldStateError("increment requires numeric current value and delta")
        after = before + delta
        transition = self.set(path, after, create_missing=False)
        return WorldTransition(
            operation="increment",
            path=transition.path,
            before=before,
            after=after,
        )

    def append(self, path: str | Sequence[str | int], value: Any) -> WorldTransition:
        segments = _path_segments(path)
        target = self._lookup(segments)
        if target is _MISSING:
            raise WorldStateError(f"world path {_display_path(path)!r} does not exist")
        if not isinstance(target, list):
            raise WorldStateError("append requires a list world value")
        before = deepcopy(target)
        target.append(deepcopy(value))
        return WorldTransition(
            operation="append",
            path=_display_path(path),
            before=before,
            after=deepcopy(target),
        )

    def apply_effect(self, effect: Mapping[str, Any]) -> WorldTransition:
        """Apply one fixture effect expressed as JSON-compatible data."""

        if not isinstance(effect, Mapping):
            raise WorldStateError("world effect must be a mapping")
        operation = effect.get("operation", effect.get("op"))
        path = effect.get("path")
        if not isinstance(operation, str) or not operation:
            raise WorldStateError("world effect requires an operation")
        if not isinstance(path, (str, list, tuple)):
            raise WorldStateError("world effect requires a path")
        normalized = operation.lower().strip()
        if normalized == "set":
            if "value" not in effect:
                raise WorldStateError("set effect requires a value")
            return self.set(
                path,
                effect["value"],
                create_missing=bool(effect.get("create_missing", True)),
            )
        if normalized in {"delete", "remove"}:
            return self.delete(path)
        if normalized in {"increment", "inc"}:
            return self.increment(path, effect.get("delta", effect.get("value", 1)))
        if normalized in {"append", "push"}:
            if "value" not in effect:
                raise WorldStateError("append effect requires a value")
            return self.append(path, effect["value"])
        raise WorldStateError(f"unsupported world effect operation {operation!r}")

    def apply_effects(
        self, effects: Sequence[Mapping[str, Any]]
    ) -> list[WorldTransition]:
        """Apply effects atomically and return their observable transitions."""

        before = self.snapshot()
        transitions: list[WorldTransition] = []
        try:
            for effect in effects:
                transitions.append(self.apply_effect(effect))
        except Exception:
            self._state = before
            raise
        return transitions
