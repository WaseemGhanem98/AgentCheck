"""Trusted evaluation baselines for local CI regression gating."""

from .compare import compare_baselines, format_comparison, gate_exit_code
from .contract import (
    BASELINE_CONTRACT_VERSION,
    BLOCKING_CATEGORIES,
    CERTIFICATION_FAILURE_CATEGORIES,
    COMPARISON_CONTRACT_VERSION,
    DEFAULT_BASELINE_FILENAME,
    BaselineComparison,
    EvaluationBaseline,
)
from .load import load_baseline

__all__ = [
    "BASELINE_CONTRACT_VERSION",
    "BLOCKING_CATEGORIES",
    "CERTIFICATION_FAILURE_CATEGORIES",
    "COMPARISON_CONTRACT_VERSION",
    "DEFAULT_BASELINE_FILENAME",
    "BaselineComparison",
    "EvaluationBaseline",
    "compare_baselines",
    "format_comparison",
    "gate_exit_code",
    "load_baseline",
]
