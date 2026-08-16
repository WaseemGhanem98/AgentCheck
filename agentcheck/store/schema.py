"""Forward-only SQLite migrations for the local AgentCheck evaluation index.

Published migrations are never rewritten. A new schema version is a new tuple
entry; ``CURRENT_SCHEMA_VERSION`` is the length of that sequence.
"""

from __future__ import annotations


SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""

MIGRATION_1_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE runs (
        run_id TEXT NOT NULL PRIMARY KEY,
        target TEXT NOT NULL,
        git_revision TEXT,
        seed INTEGER NOT NULL,
        spec_id TEXT NOT NULL,
        suite_id TEXT,
        suite_fingerprint TEXT,
        passed INTEGER NOT NULL,
        failed INTEGER NOT NULL,
        inconclusive INTEGER NOT NULL,
        infra_error INTEGER NOT NULL,
        case_count INTEGER NOT NULL,
        finding_count INTEGER NOT NULL,
        invalid_scenario_count INTEGER NOT NULL,
        artifact_path TEXT NOT NULL,
        recorded_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE run_fingerprints (
        run_id TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        PRIMARY KEY (run_id, fingerprint),
        FOREIGN KEY (run_id) REFERENCES runs(run_id)
    )
    """,
    """
    CREATE INDEX idx_runs_recorded_at ON runs (recorded_at, run_id)
    """,
)

MIGRATIONS: tuple[tuple[str, ...], ...] = (MIGRATION_1_STATEMENTS,)
CURRENT_SCHEMA_VERSION = len(MIGRATIONS)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "MIGRATIONS",
    "SCHEMA_VERSION_TABLE",
]
