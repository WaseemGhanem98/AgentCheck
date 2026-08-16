"""Small deterministic resource-budget tracker used by AgentCheck workers."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable


class BudgetExceeded(RuntimeError):
    """A case exceeded a declared resource limit."""

    def __init__(
        self, resource: str, limit: int | float, observed: int | float
    ) -> None:
        self.resource = resource
        self.limit = limit
        self.observed = observed
        super().__init__(
            f"{resource} budget exceeded: observed {observed}, limit {limit}"
        )


@dataclass(frozen=True)
class BudgetUsage:
    elapsed_seconds: float
    model_turns: int
    tool_calls: int
    retries: int
    tokens: int | None
    cost_usd: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "model_turns": self.model_turns,
            "tool_calls": self.tool_calls,
            "retries": self.retries,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
        }


_ALIASES: dict[str, tuple[str, ...]] = {
    "wall_time": (
        "wall_clock_seconds",
        "wall_time_seconds",
        "wall_clock_timeout_seconds",
        "timeout_seconds",
        "max_wall_time_seconds",
    ),
    "model_turns": ("max_model_turns", "model_turns"),
    "tool_calls": ("max_tool_calls", "tool_calls"),
    "retries": ("max_retries", "max_tool_retries", "retries"),
    "tokens": ("max_tokens", "token_budget", "tokens"),
    "cost_usd": ("max_cost_usd", "cost_budget_usd", "cost_usd"),
}


def _limit(source: Any, resource: str) -> int | float | None:
    for name in _ALIASES[resource]:
        if isinstance(source, Mapping) and name in source:
            value = source[name]
        elif hasattr(source, name):
            value = getattr(source, name)
        else:
            continue
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a non-negative number or null")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite number or null")
        return value
    return None


class BudgetTracker:
    """Track known usage without treating unavailable metrics as zero."""

    def __init__(
        self,
        budgets: Any | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        source = budgets or {}
        self._limits = {name: _limit(source, name) for name in _ALIASES}
        self._clock = clock
        self._started_at = clock()
        self._simulated_seconds = 0.0
        self._model_turns = 0
        self._tool_calls = 0
        self._retries = 0
        self._tokens: int | None = None
        self._cost_usd: float | None = None

    @property
    def limits(self) -> dict[str, int | float | None]:
        return dict(self._limits)

    def _enforce(self, resource: str, observed: int | float) -> None:
        limit = self._limits[resource]
        if limit is not None and observed > limit:
            raise BudgetExceeded(resource, limit, observed)

    def check_wall_time(self) -> float:
        elapsed = max(0.0, self._clock() - self._started_at) + self._simulated_seconds
        self._enforce("wall_time", elapsed)
        return elapsed

    def add_simulated_time(self, seconds: int | float) -> float:
        """Account for fixture latency without actually sleeping."""

        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or seconds < 0
        ):
            raise ValueError("simulated time must be a non-negative finite number")
        real_elapsed = max(0.0, self._clock() - self._started_at)
        observed = real_elapsed + self._simulated_seconds + float(seconds)
        self._enforce("wall_time", observed)
        self._simulated_seconds += float(seconds)
        return observed

    def consume_model_turn(self, count: int = 1) -> int:
        self.check_wall_time()
        self._model_turns = self._consume_count("model_turns", self._model_turns, count)
        return self._model_turns

    def consume_tool_call(self, count: int = 1) -> int:
        self.check_wall_time()
        self._tool_calls = self._consume_count("tool_calls", self._tool_calls, count)
        return self._tool_calls

    def consume_retry(self, count: int = 1) -> int:
        self.check_wall_time()
        self._retries = self._consume_count("retries", self._retries, count)
        return self._retries

    def _consume_count(self, resource: str, current: int, count: int) -> int:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{resource} increment must be a non-negative integer")
        observed = current + count
        self._enforce(resource, observed)
        return observed

    def add_tokens(self, count: int | None) -> int | None:
        self.check_wall_time()
        if count is None:
            return self._tokens
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("token usage must be a non-negative integer or null")
        observed = (self._tokens or 0) + count
        self._enforce("tokens", observed)
        self._tokens = observed
        return self._tokens

    def add_cost(self, cost_usd: int | float | None) -> float | None:
        self.check_wall_time()
        if cost_usd is None:
            return self._cost_usd
        if (
            isinstance(cost_usd, bool)
            or not isinstance(cost_usd, (int, float))
            or not math.isfinite(float(cost_usd))
            or cost_usd < 0
        ):
            raise ValueError("cost usage must be a non-negative finite number or null")
        observed = (self._cost_usd or 0.0) + float(cost_usd)
        self._enforce("cost_usd", observed)
        self._cost_usd = observed
        return self._cost_usd

    def snapshot(self) -> BudgetUsage:
        return BudgetUsage(
            elapsed_seconds=self.check_wall_time(),
            model_turns=self._model_turns,
            tool_calls=self._tool_calls,
            retries=self._retries,
            tokens=self._tokens,
            cost_usd=self._cost_usd,
        )
