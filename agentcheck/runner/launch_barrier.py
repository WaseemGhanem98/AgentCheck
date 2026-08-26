"""Deterministic commit ordering for one batch of concurrently dispatched calls.

``ToolGateway.plan_batch`` decides fixture ownership, invocation index, and
retry classification once, in call order -- none of that depends on when a
reservation is actually committed. Applying a fixture's world-state effects
is different: two reservations whose effects touch overlapping simulated
state would still leave the final world state depending on whichever one's
``commit`` call happens to run first, if nothing constrains that order.

``LaunchBarrier`` gates a batch's commits to happen in the same order they
were planned (``rank`` order), while leaving everything a participant does
*before* committing -- argument parsing, guardrails, attempt recording -- free
to interleave under real concurrent dispatch. It is built from plain
``asyncio.Event`` objects: no thread locks, no sleeps, no timing assumptions,
and no behaviour that depends on which OS thread or task the event loop
happens to run first. It only orders cooperative coroutines within one event
loop; it does not (and cannot) order real OS threads.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class LaunchBarrier:
    """Serializes a fixed number of ranked participants to rank order."""

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError("size must be non-negative")
        self._events = [asyncio.Event() for _ in range(size)]
        if size:
            self._events[0].set()

    @property
    def size(self) -> int:
        return len(self._events)

    async def wait_for_rank(self, rank: int) -> None:
        """Suspend until every rank before ``rank`` has called ``release``."""

        self._check_rank(rank)
        await self._events[rank].wait()

    def release(self, rank: int) -> None:
        """Allow the next rank to proceed. Idempotent for a given rank."""

        self._check_rank(rank)
        if rank + 1 < len(self._events):
            self._events[rank + 1].set()

    async def run_in_order(self, rank: int, action: Callable[[], T]) -> T:
        """Run ``action`` (a plain synchronous callable) once this rank's turn
        arrives, then release the next rank -- even if ``action`` raises."""

        await self.wait_for_rank(rank)
        try:
            return action()
        finally:
            self.release(rank)

    async def run_in_order_async(
        self, rank: int, action: Callable[[], Awaitable[T]]
    ) -> T:
        """Same as ``run_in_order``, for an ``action`` that is itself async."""

        await self.wait_for_rank(rank)
        try:
            return await action()
        finally:
            self.release(rank)

    def _check_rank(self, rank: int) -> None:
        if not isinstance(rank, int) or isinstance(rank, bool) or not (
            0 <= rank < len(self._events)
        ):
            raise ValueError(f"rank {rank!r} is out of range for a barrier of size {len(self._events)}")


__all__ = ["LaunchBarrier"]
