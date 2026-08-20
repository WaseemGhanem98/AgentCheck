"""Local, non-load-bearing persistence for AgentCheck evaluation results."""

from .schema import CURRENT_SCHEMA_VERSION
from .sqlite import (
    DEFAULT_STORE_FILENAME,
    EvaluationStore,
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
    stored_run_from_execution,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
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
    "list_runs_readonly",
    "open_evaluation_store",
    "resolve_store_path",
    "stored_run_from_execution",
]
