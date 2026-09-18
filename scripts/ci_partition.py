"""Pytest plugin: partition the complete collected suite across two CI jobs.

Load explicitly with ``-p scripts.ci_partition --ci-shard {1,2}``. Both jobs
collect everything, then retain whole files, preserving module fixtures and
parametrizations. No duration whitelist or filename discovery can omit new tests.
The same plugin and option reach xdist workers via pytest's normal -p mechanism.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest


SHARD_COUNT = 2


def partition_nodeids(nodeids: Sequence[str], shard: int) -> list[str]:
    if shard not in range(1, SHARD_COUNT + 1):
        raise ValueError("CI shard must be 1 or 2")
    if len(set(nodeids)) != len(nodeids):
        raise ValueError("duplicate collected node IDs prevent CI partitioning")
    files = sorted({nodeid.split("::", 1)[0] for nodeid in nodeids})
    if len(files) < SHARD_COUNT:
        raise ValueError("CI partition requires at least two collected test files")
    selected_files = set(files[shard - 1 :: SHARD_COUNT])
    return [nodeid for nodeid in nodeids if nodeid.split("::", 1)[0] in selected_files]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--ci-shard", type=int, choices=(1, 2), help="CI file shard (1 or 2)"
    )


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("ci_shard") is None:
        raise pytest.UsageError(
            "--ci-shard is required when loading scripts.ci_partition"
        )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    try:
        selected_ids = set(
            partition_nodeids(
                [item.nodeid for item in items], config.getoption("ci_shard")
            )
        )
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    selected = [item for item in items if item.nodeid in selected_ids]
    deselected = [item for item in items if item.nodeid not in selected_ids]
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
