from __future__ import annotations

import os
import shutil
import sqlite3
import stat
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

import agentcheck.application as application
from agentcheck.cli import _parser, main
from agentcheck.config import AgentCheckConfig
from agentcheck.domain import Verdict
from agentcheck.store import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_STORE_FILENAME,
    NullEvaluationStore,
    SqliteEvaluationStore,
    StoreConflictError,
    StoreError,
    StoreSchemaError,
    StoredRun,
    apply_migrations,
    default_store_relative_path,
    list_runs_readonly,
    open_evaluation_store,
    resolve_store_path,
)


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
EXPECTED_FAILURES = {
    "delete_without_confirmation",
    "ambiguous_account_selection",
    "destructive_ambiguous_timeout",
    "fabricated_email_success",
    "duplicate_email_update",
}
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX file modes required")
SEED = 1729
INSTRUCTION_PHRASE = "Ask for explicit confirmation before"


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "account_agent"
    shutil.copytree(
        EXAMPLE,
        target,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    return target


def _sample_run(**overrides: object) -> StoredRun:
    record = StoredRun(
        run_id="run-1",
        target="/tmp/target",
        git_revision="abc123",
        seed=SEED,
        spec_id="agentspec-test",
        suite_id="frozensuite-test",
        suite_fingerprint="sha256:deadbeef",
        fingerprints=("sha256:one", "sha256:two"),
        passed=7,
        failed=5,
        inconclusive=0,
        infra_error=0,
        case_count=12,
        finding_count=3,
        invalid_scenario_count=0,
        artifact_path=".agentcheck/runs/run-1",
        recorded_at="2026-01-01T00:00:00+00:00",
    )
    if not overrides:
        return record
    return replace(record, **overrides)  # type: ignore[arg-type]


def _dump(path: Path) -> str:
    connection = sqlite3.connect(os.fspath(path))
    try:
        return "\n".join(connection.iterdump())
    finally:
        connection.close()


def test_default_store_path_lives_in_the_artifacts_directory() -> None:
    config = AgentCheckConfig()
    assert default_store_relative_path(config) == f".agentcheck/{DEFAULT_STORE_FILENAME}"
    assert AgentCheckConfig().store_path is None


def test_store_path_rejects_unsafe_locations() -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        AgentCheckConfig(store_path="../escape.sqlite")
    with pytest.raises(ValueError, match="safe relative path"):
        AgentCheckConfig(store_path="/tmp/agentcheck.sqlite")
    with pytest.raises(ValueError, match="must not be empty"):
        AgentCheckConfig(store_path="   ")
    with pytest.raises(ValidationError):
        AgentCheckConfig.model_validate(
            {
                "schema_version": "agentcheck.config.v1",
                "store_path": "ok.sqlite",
                "unexpected": True,
            }
        )


def test_list_runs_readonly_does_not_create_or_migrate(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(StoreError, match="does not exist"):
        list_runs_readonly(missing)
    assert not missing.exists()

    path = tmp_path / DEFAULT_STORE_FILENAME
    SqliteEvaluationStore(path).record_run(_sample_run())
    listed = list_runs_readonly(path)
    assert [item.run_id for item in listed] == ["run-1"]
    assert listed[0].index_key() == _sample_run().index_key()


def test_null_store_is_a_noop() -> None:
    store = NullEvaluationStore()
    store.record_run(_sample_run())
    assert store.list_runs() == ()
    assert store.get_run("run-1") is None


def test_bootstrap_and_idempotent_migrations(tmp_path: Path) -> None:
    path = tmp_path / DEFAULT_STORE_FILENAME
    store = SqliteEvaluationStore(path)
    store.record_run(_sample_run())

    connection = sqlite3.connect(os.fspath(path), isolation_level=None)
    try:
        apply_migrations(connection)
        apply_migrations(connection)
        versions = [
            int(row[0])
            for row in connection.execute(
                "SELECT version FROM schema_version ORDER BY version"
            )
        ]
    finally:
        connection.close()

    assert versions == [CURRENT_SCHEMA_VERSION]
    loaded = store.get_run("run-1")
    assert loaded is not None
    assert loaded.index_key() == _sample_run().index_key()
    assert store.list_runs()[0].run_id == "run-1"


@POSIX_ONLY
def test_new_store_file_is_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / DEFAULT_STORE_FILENAME
    SqliteEvaluationStore(path).record_run(_sample_run())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_equivalent_duplicate_insert_is_idempotent(tmp_path: Path) -> None:
    store = SqliteEvaluationStore(tmp_path / DEFAULT_STORE_FILENAME)
    first = _sample_run()
    store.record_run(first)
    store.record_run(_sample_run(recorded_at="2026-02-02T00:00:00+00:00"))
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0].recorded_at == first.recorded_at


def test_incompatible_duplicate_is_refused(tmp_path: Path) -> None:
    store = SqliteEvaluationStore(tmp_path / DEFAULT_STORE_FILENAME)
    store.record_run(_sample_run())
    with pytest.raises(StoreConflictError, match="incompatible"):
        store.record_run(_sample_run(failed=99))
    assert store.get_run("run-1") is not None
    assert store.get_run("run-1").failed == 5  # type: ignore[union-attr]


def test_hostile_run_id_cannot_inject_sql(tmp_path: Path) -> None:
    path = tmp_path / DEFAULT_STORE_FILENAME
    store = SqliteEvaluationStore(path)
    hostile = "x'; DROP TABLE runs; --"
    store.record_run(_sample_run(run_id=hostile, artifact_path=".agentcheck/runs/x"))
    store.record_run(_sample_run(run_id="safe-run", artifact_path=".agentcheck/runs/safe"))

    assert store.get_run(hostile) is not None
    assert store.get_run("x") is None
    dumped = _dump(path)
    assert "DROP TABLE" in dumped
    connection = sqlite3.connect(os.fspath(path))
    try:
        count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()
    assert count == (2,)
    assert "runs" in tables
    assert "run_fingerprints" in tables


def test_concurrent_record_run_does_not_deadlock(tmp_path: Path) -> None:
    path = tmp_path / DEFAULT_STORE_FILENAME

    def write(index: int) -> None:
        SqliteEvaluationStore(path).record_run(
            _sample_run(
                run_id=f"run-{index:02d}",
                artifact_path=f".agentcheck/runs/run-{index:02d}",
            )
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(8)))

    runs = SqliteEvaluationStore(path).list_runs()
    assert [item.run_id for item in runs] == [f"run-{index:02d}" for index in range(8)]


def test_symlink_store_path_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "target"
    (root / ".agentcheck").mkdir(parents=True)
    outside = tmp_path / "outside.sqlite"
    outside.write_bytes(b"")
    (root / ".agentcheck" / DEFAULT_STORE_FILENAME).symlink_to(outside)
    with pytest.raises(StoreError, match="symlink"):
        resolve_store_path(root, AgentCheckConfig())


def test_future_schema_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / DEFAULT_STORE_FILENAME
    store = SqliteEvaluationStore(path)
    store.record_run(_sample_run())
    connection = sqlite3.connect(os.fspath(path))
    try:
        connection.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (99, "2026-01-01T00:00:00+00:00"),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(StoreSchemaError, match="newer than"):
        store.record_run(_sample_run(run_id="run-2"))
    with pytest.raises(StoreSchemaError, match="newer than"):
        store.get_run("run-1")
    connection = sqlite3.connect(os.fspath(path))
    try:
        ids = [str(row[0]) for row in connection.execute("SELECT run_id FROM runs")]
    finally:
        connection.close()
    assert ids == ["run-1"]


def test_two_consecutive_runs_are_queryable(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    first = application.execute_suite(target, seed=SEED, run_id="store-one")
    second = application.execute_suite(target, seed=SEED, run_id="store-two")

    store = open_evaluation_store(target, first.config)
    runs = store.list_runs()
    assert [item.run_id for item in runs] == ["store-one", "store-two"]
    loaded = store.get_run("store-one")
    assert loaded is not None
    assert loaded.seed == SEED
    assert loaded.spec_id == first.spec.spec_id
    assert loaded.case_count == 12
    assert loaded.passed == 7
    assert loaded.failed == 5
    assert loaded.inconclusive == 0
    assert loaded.infra_error == 0
    assert loaded.finding_count == len(first.findings)
    assert loaded.artifact_path == ".agentcheck/runs/store-one"
    assert loaded.fingerprints == tuple(
        sorted({scenario.fingerprint for scenario in first.scenarios})
    )
    assert second.counts == first.counts
    dumped = _dump(store.path)
    assert INSTRUCTION_PHRASE not in dumped
    assert "OPENAI_API_KEY" not in dumped
    assert "sk-" not in dumped


def test_no_store_fully_bypasses_sqlite(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    execution = application.execute_suite(
        target, seed=SEED, run_id="no-store", persist_store=False
    )
    assert execution.counts == Counter({Verdict.PASS: 7, Verdict.FAIL: 5})
    assert {
        item.scenario_id for item in execution.evaluations if item.verdict == Verdict.FAIL
    } == EXPECTED_FAILURES
    assert not (target / ".agentcheck" / DEFAULT_STORE_FILENAME).exists()
    assert all(item.verdict != Verdict.INFRA_ERROR for item in execution.evaluations)


def test_failing_store_does_not_change_verdicts_or_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = _copy_example(tmp_path)

    def _boom(*_args: object, **_kwargs: object) -> SqliteEvaluationStore:
        raise RuntimeError("disk I/O error")

    monkeypatch.setattr(application, "open_evaluation_store", _boom)
    execution = application.execute_suite(target, seed=SEED, run_id="store-failing")
    captured = capsys.readouterr()
    assert execution.counts == Counter({Verdict.PASS: 7, Verdict.FAIL: 5})
    assert all(item.verdict != Verdict.INFRA_ERROR for item in execution.evaluations)
    assert "AgentCheck warning: evaluation store failed:" in captured.err
    assert "disk I/O error" in captured.err
    assert execution.report_path.is_file()
    assert cli_exit_for(execution) == 1


def test_corrupt_database_degrades_to_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _copy_example(tmp_path)
    store_path = target / ".agentcheck" / DEFAULT_STORE_FILENAME
    store_path.parent.mkdir(mode=0o700, parents=True)
    store_path.write_bytes(b"this is not a sqlite database")
    if os.name == "posix":
        os.chmod(store_path, 0o600)

    execution = application.execute_suite(target, seed=SEED, run_id="store-corrupt")
    captured = capsys.readouterr()
    assert execution.counts == Counter({Verdict.PASS: 7, Verdict.FAIL: 5})
    assert "AgentCheck warning: evaluation store failed:" in captured.err
    assert store_path.read_bytes() == b"this is not a sqlite database"
    assert cli_exit_for(execution) == 1


def test_generate_does_not_write_the_evaluation_store(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    application.generate_suite(target, seed=SEED)
    assert not (target / ".agentcheck" / DEFAULT_STORE_FILENAME).exists()


def test_cli_test_help_documents_no_store(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["test", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--no-store" in help_text
    assert "SQLite" in help_text
    parsed = _parser().parse_args(["test", "--no-store"])
    assert parsed.no_store is True
    assert _parser().parse_args(["test"]).no_store is False


def cli_exit_for(execution: application.SuiteExecution) -> int:
    counts = execution.counts
    if counts[Verdict.INFRA_ERROR]:
        return 2
    if counts[Verdict.FAIL]:
        return 1
    if counts[Verdict.INCONCLUSIVE]:
        return 3
    return 0
