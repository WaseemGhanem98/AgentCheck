"""CI partitions must cover every collected test and propagate failures."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.ci_partition import partition_nodeids


ROOT = Path(__file__).resolve().parents[2]


def test_partitions_preserve_all_files_parameters_and_collection_order() -> None:
    nodes = [
        "tests/nested/test_new.py::test_added[value]",
        "tests/a_test.py::test_parameter[1]",
        "tests/a_test.py::test_parameter[2]",
        "tests/test_last.py::TestGroup::test_case",
    ]
    first, second = (partition_nodeids(nodes, shard) for shard in (1, 2))
    assert set(first).isdisjoint(second)
    assert set(first) | set(second) == set(nodes)
    assert first == [nodes[1], nodes[2], nodes[3]]
    assert second == [nodes[0]]


@pytest.mark.parametrize("shard", [0, -1, 3])
def test_invalid_shard_is_refused(shard: int) -> None:
    with pytest.raises(ValueError, match="shard"):
        partition_nodeids(["test_a.py::test_a", "test_b.py::test_b"], shard)


@pytest.mark.parametrize(
    "nodes", [[], ["test_a.py::test_a"], ["test_a.py::test_a"] * 2]
)
def test_empty_partition_or_duplicate_inventory_is_refused(nodes: list[str]) -> None:
    with pytest.raises(ValueError):
        partition_nodeids(nodes, 1)


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # Load the source plugin from the repository while collecting a synthetic
    # suite. These tiny controls execute no agent and make no provider calls.
    plugin = ROOT / "scripts" / "ci_partition.py"
    digest = hashlib.sha256(plugin.read_bytes()).hexdigest()
    (root / "conftest.py").write_text(
        "import hashlib\nfrom pathlib import Path\n"
        "import scripts.ci_partition as plugin\n"
        "def pytest_configure(config):\n"
        f"    assert Path(plugin.__file__).resolve() == Path({str(plugin)!r})\n"
        f"    assert hashlib.sha256(Path(plugin.__file__).read_bytes()).hexdigest() == {digest!r}\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "scripts.ci_partition",
            str(root),
            "-q",
            *args,
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_actual_xdist_workers_partition_and_propagate_test_failure(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_a.py").write_text(
        "import pytest\n@pytest.mark.parametrize('x', [1, 2])\ndef test_a(x): assert x > 0\n"
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    # Sorts before test_a.py; nonstandard-but-supported *_test.py discovery.
    (nested / "b_test.py").write_text("def test_failure(): assert False\n")
    failed = _run(tmp_path, "--ci-shard", "1", "-n", "1")
    passed = _run(tmp_path, "--ci-shard", "2", "-n", "1")
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert "1 failed" in failed.stdout
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert "2 passed" in passed.stdout


def test_collection_failure_cannot_be_hidden_in_other_shard(tmp_path: Path) -> None:
    (tmp_path / "test_a.py").write_text("def test_ok(): pass\n")
    (tmp_path / "test_b.py").write_text("raise RuntimeError('collection witness')\n")
    (tmp_path / "test_c.py").write_text("def test_other(): pass\n")
    for shard in ("1", "2"):
        result = _run(tmp_path, "--ci-shard", shard, "-n", "1")
        assert result.returncode != 0
        assert "collection witness" in result.stdout + result.stderr


def test_missing_shard_cannot_silently_run_partial_suite(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode != 0
    assert "--ci-shard is required" in result.stderr
