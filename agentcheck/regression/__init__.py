"""Semantic run-to-run behavioral regression comparison."""

from .compare import compare_runs, exit_code
from .contract import (
    RUN_COMPARISON_CONTRACT_VERSION,
    Comparability,
    ComparabilityCaveat,
    IncomparableReason,
    RunComparison,
    RunComparisonItem,
)
from .service import (
    CheckedRunComparison,
    compare_stored_runs,
    encode_run_comparison,
    format_run_comparison,
)

__all__ = [
    "RUN_COMPARISON_CONTRACT_VERSION",
    "CheckedRunComparison",
    "Comparability",
    "ComparabilityCaveat",
    "IncomparableReason",
    "RunComparison",
    "RunComparisonItem",
    "compare_runs",
    "compare_stored_runs",
    "encode_run_comparison",
    "exit_code",
    "format_run_comparison",
]
