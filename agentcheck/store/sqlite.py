"""Local SQLite index of AgentCheck suite runs.

JSON/JSONL artifacts remain authoritative. This store is a non-load-bearing
index: a storage failure must never change a verdict or an exit code. Rows
record identity, counts, fingerprints, and artifact paths — not payloads,
instructions, handlers, or secrets.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from agentcheck.config import AgentCheckConfig, contained_path
from agentcheck.domain import Verdict, utc_now
from agentcheck.errors import ConfigurationError
from agentcheck.privacy import redact_artifact

from .schema import CURRENT_SCHEMA_VERSION, MIGRATIONS, SCHEMA_VERSION_TABLE

if TYPE_CHECKING:
    from agentcheck.application import SuiteExecution


DEFAULT_STORE_FILENAME = "agentcheck.sqlite"


class StoreError(Exception):
    """The local evaluation index could not be used."""


class StoreSchemaError(StoreError):
    """The on-disk schema version is unsupported."""


class StoreConflictError(StoreError):
    """A run ID is already indexed with incompatible data."""


@dataclass(frozen=True, slots=True)
class StoredRun:
    """One indexed suite run. This is not a domain contract and not a payload."""

    run_id: str
    target: str
    git_revision: str | None
    seed: int
    spec_id: str
    suite_id: str | None
    suite_fingerprint: str | None
    fingerprints: tuple[str, ...]
    passed: int
    failed: int
    inconclusive: int
    infra_error: int
    case_count: int
    finding_count: int
    invalid_scenario_count: int
    artifact_path: str
    recorded_at: str

    def index_key(self) -> tuple[object, ...]:
        """Identity used to detect incompatible overwrites. Timestamps differ."""

        return (
            self.run_id,
            self.target,
            self.git_revision,
            self.seed,
            self.spec_id,
            self.suite_id,
            self.suite_fingerprint,
            self.fingerprints,
            self.passed,
            self.failed,
            self.inconclusive,
            self.infra_error,
            self.case_count,
            self.finding_count,
            self.invalid_scenario_count,
            self.artifact_path,
        )


class EvaluationStore(Protocol):
    """Queryable local index of completed suite runs."""

    def record_run(self, record: StoredRun) -> None:
        """Insert a run, or no-op when an equivalent row already exists."""

    def list_runs(self) -> tuple[StoredRun, ...]:
        """Return indexed runs in recorded order."""

    def get_run(self, run_id: str) -> StoredRun | None:
        """Return one indexed run, or None when it is absent."""


class NullEvaluationStore:
    """No-op default. `--no-store` never opens a database file."""

    def record_run(self, record: StoredRun) -> None:
        return

    def list_runs(self) -> tuple[StoredRun, ...]:
        return ()

    def get_run(self, run_id: str) -> StoredRun | None:
        return None


class SqliteEvaluationStore:
    """File-backed evaluation index with forward-only migrations."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record_run(self, record: StoredRun) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = _load_run(connection, record.run_id)
                if existing is not None:
                    if existing.index_key() == record.index_key():
                        connection.execute("ROLLBACK")
                        return
                    connection.execute("ROLLBACK")
                    raise StoreConflictError(
                        f"run {record.run_id!r} is already indexed with incompatible data"
                    )
                _insert_run(connection, record)
                connection.execute("COMMIT")
            except StoreError:
                _rollback_quietly(connection)
                raise
            except sqlite3.Error as exc:
                _rollback_quietly(connection)
                raise StoreError(f"evaluation store is unavailable: {exc}") from exc
            finally:
                connection.close()

    def list_runs(self) -> tuple[StoredRun, ...]:
        with self._lock:
            connection = self._connect()
            try:
                rows = connection.execute(
                    "SELECT run_id FROM runs ORDER BY recorded_at ASC, run_id ASC"
                ).fetchall()
                runs: list[StoredRun] = []
                for row in rows:
                    loaded = _load_run(connection, str(row["run_id"]))
                    if loaded is not None:
                        runs.append(loaded)
                return tuple(runs)
            except sqlite3.Error as exc:
                raise StoreError(f"evaluation store is unavailable: {exc}") from exc
            finally:
                connection.close()

    def get_run(self, run_id: str) -> StoredRun | None:
        with self._lock:
            connection = self._connect()
            try:
                return _load_run(connection, run_id)
            except sqlite3.Error as exc:
                raise StoreError(f"evaluation store is unavailable: {exc}") from exc
            finally:
                connection.close()

    def _connect(self) -> sqlite3.Connection:
        _ensure_private_database(self.path)
        try:
            connection = sqlite3.connect(
                os.fspath(self.path),
                timeout=30.0,
                isolation_level=None,
                check_same_thread=False,
            )
        except sqlite3.Error as exc:
            raise StoreError(f"evaluation store is unavailable: {exc}") from exc
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 30000")
            try:
                connection.execute("PRAGMA journal_mode = WAL")
            except sqlite3.Error:
                pass
            apply_migrations(connection)
        except (StoreError, sqlite3.Error):
            connection.close()
            raise
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return connection


def apply_migrations(connection: sqlite3.Connection) -> None:
    """Apply unpublished schema versions. Re-running is a no-op."""

    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(SCHEMA_VERSION_TABLE)
        row = connection.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current = int(row[0] or 0) if row is not None else 0
        if current > CURRENT_SCHEMA_VERSION:
            connection.execute("ROLLBACK")
            raise StoreSchemaError(
                f"evaluation store schema version {current} is newer than "
                f"supported version {CURRENT_SCHEMA_VERSION}"
            )
        applied_at = utc_now().isoformat()
        for version, statements in enumerate(MIGRATIONS, start=1):
            if version <= current:
                continue
            for statement in statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                (version, applied_at),
            )
        connection.execute("COMMIT")
    except StoreError:
        _rollback_quietly(connection)
        raise
    except sqlite3.Error as exc:
        _rollback_quietly(connection)
        raise StoreError(f"evaluation store is unavailable: {exc}") from exc


def default_store_relative_path(config: AgentCheckConfig) -> str:
    if config.store_path is not None:
        return config.store_path
    return f"{config.artifacts_directory}/{DEFAULT_STORE_FILENAME}"


def resolve_store_path(root: Path, config: AgentCheckConfig) -> Path:
    relative = default_store_relative_path(config)
    unresolved = root.resolve() / Path(relative)
    if unresolved.is_symlink():
        raise StoreError("evaluation store path must not be a symlink")
    try:
        return contained_path(root, relative)
    except ConfigurationError as exc:
        raise StoreError(str(exc)) from exc


def open_evaluation_store(root: Path, config: AgentCheckConfig) -> SqliteEvaluationStore:
    return SqliteEvaluationStore(resolve_store_path(root, config))


def stored_run_from_execution(execution: SuiteExecution) -> StoredRun:
    """Build an index row from a completed suite execution."""

    counts = execution.counts
    frozen = execution.frozen_suite
    target_root = execution.target_root
    try:
        relative_artifact = execution.artifact_directory.resolve().relative_to(
            target_root.resolve()
        )
    except ValueError as exc:
        raise StoreError("artifact path must remain inside the target directory") from exc
    fingerprints = tuple(sorted({scenario.fingerprint for scenario in execution.scenarios}))
    payload = redact_artifact(
        {
            "run_id": execution.run_id,
            "target": str(target_root),
            "git_revision": execution.git_revision,
            "seed": execution.seed,
            "spec_id": execution.spec.spec_id,
            "suite_id": None if frozen is None else frozen.suite_id,
            "suite_fingerprint": None if frozen is None else frozen.fingerprint,
            "fingerprints": list(fingerprints),
            "passed": int(counts[Verdict.PASS]),
            "failed": int(counts[Verdict.FAIL]),
            "inconclusive": int(counts[Verdict.INCONCLUSIVE]),
            "infra_error": int(counts[Verdict.INFRA_ERROR]),
            "case_count": len(execution.evaluations),
            "finding_count": len(execution.findings),
            "invalid_scenario_count": len(execution.invalid_scenarios),
            "artifact_path": relative_artifact.as_posix(),
            "recorded_at": utc_now().isoformat(),
        }
    )
    if not isinstance(payload, dict):
        raise StoreError("evaluation store redaction produced a non-object")
    return _stored_run_from_mapping(payload)


def _stored_run_from_mapping(payload: dict[str, object]) -> StoredRun:
    fingerprints_raw = payload.get("fingerprints") or ()
    if not isinstance(fingerprints_raw, (list, tuple)):
        raise StoreError("evaluation store fingerprints must be a list")
    fingerprints = tuple(str(item) for item in fingerprints_raw)
    return StoredRun(
        run_id=str(payload["run_id"]),
        target=str(payload["target"]),
        git_revision=_optional_str(payload.get("git_revision")),
        seed=_require_int(payload["seed"], "seed"),
        spec_id=str(payload["spec_id"]),
        suite_id=_optional_str(payload.get("suite_id")),
        suite_fingerprint=_optional_str(payload.get("suite_fingerprint")),
        fingerprints=fingerprints,
        passed=_require_int(payload["passed"], "passed"),
        failed=_require_int(payload["failed"], "failed"),
        inconclusive=_require_int(payload["inconclusive"], "inconclusive"),
        infra_error=_require_int(payload["infra_error"], "infra_error"),
        case_count=_require_int(payload["case_count"], "case_count"),
        finding_count=_require_int(payload["finding_count"], "finding_count"),
        invalid_scenario_count=_require_int(
            payload["invalid_scenario_count"], "invalid_scenario_count"
        ),
        artifact_path=str(payload["artifact_path"]),
        recorded_at=str(payload["recorded_at"]),
    )


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StoreError(f"evaluation store field {field} must be an integer")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _ensure_private_database(path: Path) -> None:
    if path.is_symlink():
        raise StoreError("evaluation store path must not be a symlink")
    parent = path.parent
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(parent, 0o700)
        if not path.exists():
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
        if path.is_symlink() or not path.is_file():
            raise StoreError("evaluation store path must be a regular file")
        os.chmod(path, 0o600)
    except OSError as exc:
        raise StoreError(f"unable to create evaluation store: {exc}") from exc


def _insert_run(connection: sqlite3.Connection, record: StoredRun) -> None:
    connection.execute(
        """
        INSERT INTO runs (
            run_id, target, git_revision, seed, spec_id, suite_id, suite_fingerprint,
            passed, failed, inconclusive, infra_error, case_count, finding_count,
            invalid_scenario_count, artifact_path, recorded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.run_id,
            record.target,
            record.git_revision,
            record.seed,
            record.spec_id,
            record.suite_id,
            record.suite_fingerprint,
            record.passed,
            record.failed,
            record.inconclusive,
            record.infra_error,
            record.case_count,
            record.finding_count,
            record.invalid_scenario_count,
            record.artifact_path,
            record.recorded_at,
        ),
    )
    connection.executemany(
        "INSERT INTO run_fingerprints (run_id, fingerprint) VALUES (?, ?)",
        [(record.run_id, fingerprint) for fingerprint in record.fingerprints],
    )


def _load_run(connection: sqlite3.Connection, run_id: str) -> StoredRun | None:
    row = connection.execute(
        """
        SELECT run_id, target, git_revision, seed, spec_id, suite_id, suite_fingerprint,
               passed, failed, inconclusive, infra_error, case_count, finding_count,
               invalid_scenario_count, artifact_path, recorded_at
        FROM runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    fingerprint_rows = connection.execute(
        """
        SELECT fingerprint FROM run_fingerprints
        WHERE run_id = ?
        ORDER BY fingerprint ASC
        """,
        (run_id,),
    ).fetchall()
    return StoredRun(
        run_id=str(row["run_id"]),
        target=str(row["target"]),
        git_revision=row["git_revision"],
        seed=int(row["seed"]),
        spec_id=str(row["spec_id"]),
        suite_id=row["suite_id"],
        suite_fingerprint=row["suite_fingerprint"],
        fingerprints=tuple(str(item["fingerprint"]) for item in fingerprint_rows),
        passed=int(row["passed"]),
        failed=int(row["failed"]),
        inconclusive=int(row["inconclusive"]),
        infra_error=int(row["infra_error"]),
        case_count=int(row["case_count"]),
        finding_count=int(row["finding_count"]),
        invalid_scenario_count=int(row["invalid_scenario_count"]),
        artifact_path=str(row["artifact_path"]),
        recorded_at=str(row["recorded_at"]),
    )


def _rollback_quietly(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("ROLLBACK")
    except sqlite3.Error:
        pass


__all__ = [
    "DEFAULT_STORE_FILENAME",
    "EvaluationStore",
    "NullEvaluationStore",
    "SqliteEvaluationStore",
    "StoreConflictError",
    "StoreError",
    "StoreSchemaError",
    "StoredRun",
    "apply_migrations",
    "default_store_relative_path",
    "open_evaluation_store",
    "resolve_store_path",
    "stored_run_from_execution",
]
