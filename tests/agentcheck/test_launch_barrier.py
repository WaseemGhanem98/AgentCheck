"""Adversarial coverage for LaunchBarrier: commit order must follow rank,
never whichever task happens to reach the barrier first.

Every test drives ordering with explicit ``asyncio.Event``/``asyncio.sleep(0)``
yields rather than real timing, so a fast or slow machine cannot change the
outcome and there is nothing here for a flaky sleep to depend on.
"""

from __future__ import annotations

import asyncio

import pytest

from agentcheck.runner import LaunchBarrier


def test_higher_rank_waits_even_when_it_reaches_the_barrier_first() -> None:
    """Task for rank 1 does its pre-work fast and arrives at the barrier well
    before rank 0 even starts -- rank 0 must still commit first."""

    order: list[int] = []
    barrier = LaunchBarrier(2)
    rank0_may_start = asyncio.Event()

    async def participant(rank: int, *, delay_start: bool) -> None:
        if delay_start:
            await rank0_may_start.wait()
        await barrier.wait_for_rank(rank)
        order.append(rank)
        barrier.release(rank)

    async def scenario() -> None:
        second = asyncio.create_task(participant(1, delay_start=False))
        # Let rank 1's task run all the way up to (and block on) the barrier
        # before rank 0 even begins.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        first = asyncio.create_task(participant(0, delay_start=True))
        rank0_may_start.set()
        await asyncio.gather(first, second)

    asyncio.run(scenario())
    assert order == [0, 1]


def test_barrier_orders_three_participants_regardless_of_arrival_order() -> None:
    order: list[int] = []
    barrier = LaunchBarrier(3)

    async def participant(rank: int, arrival_yields: int) -> None:
        for _ in range(arrival_yields):
            await asyncio.sleep(0)
        await barrier.wait_for_rank(rank)
        order.append(rank)
        barrier.release(rank)

    async def scenario() -> None:
        # Rank 2 arrives first, rank 0 last -- the reverse of commit order.
        await asyncio.gather(
            participant(2, arrival_yields=0),
            participant(1, arrival_yields=2),
            participant(0, arrival_yields=4),
        )

    asyncio.run(scenario())
    assert order == [0, 1, 2]


def test_run_in_order_releases_the_next_rank_even_if_the_action_raises() -> None:
    barrier = LaunchBarrier(2)
    order: list[str] = []

    def failing_action() -> None:
        order.append("rank0-ran")
        raise RuntimeError("boom")

    def second_action() -> None:
        order.append("rank1-ran")

    async def scenario() -> None:
        async def run_rank0() -> None:
            with pytest.raises(RuntimeError):
                await barrier.run_in_order(0, failing_action)

        async def run_rank1() -> None:
            await barrier.run_in_order(1, second_action)

        await asyncio.gather(run_rank0(), run_rank1())

    asyncio.run(scenario())
    assert order == ["rank0-ran", "rank1-ran"]


def test_run_in_order_async_supports_coroutine_actions() -> None:
    barrier = LaunchBarrier(2)
    order: list[int] = []

    async def make_action(rank: int):
        async def action() -> int:
            await asyncio.sleep(0)
            order.append(rank)
            return rank

        return action

    async def scenario() -> None:
        action1 = await make_action(1)
        action0 = await make_action(0)

        async def run(rank: int, action):
            return await barrier.run_in_order_async(rank, action)

        results = await asyncio.gather(run(1, action1), run(0, action0))
        assert results == [1, 0]

    asyncio.run(scenario())
    assert order == [0, 1]


def test_zero_size_barrier_accepts_no_ranks() -> None:
    barrier = LaunchBarrier(0)
    assert barrier.size == 0
    with pytest.raises(ValueError):
        asyncio.run(barrier.wait_for_rank(0))


def test_out_of_range_rank_is_rejected() -> None:
    barrier = LaunchBarrier(2)
    with pytest.raises(ValueError):
        asyncio.run(barrier.wait_for_rank(2))
    with pytest.raises(ValueError):
        barrier.release(-1)
