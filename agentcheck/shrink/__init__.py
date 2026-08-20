"""Deterministic counterexample shrinking. Consumes replay manifests only."""

from .candidates import REDUCTION_ORDER
from .complexity import ScenarioComplexity, is_strictly_smaller, measure_complexity
from .result import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_ROUNDS,
    MAX_MAX_CANDIDATES,
    MAX_MAX_ROUNDS,
    MAX_SOURCE_SCANS,
    SHRINK_RESULT_CONTRACT_VERSION,
    ShrinkResult,
    encode_shrink_result,
    shrink_result_relative_path,
    write_shrink_result,
)
from .search import ShrinkSearchOutcome, shrink_scenario
from .signature import (
    FAILURE_SIGNATURE_CONTRACT_VERSION,
    FailureSignature,
    extract_failure_signature,
    signatures_match,
    unsupported_failure_reason,
)

__all__ = [
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_ROUNDS",
    "FAILURE_SIGNATURE_CONTRACT_VERSION",
    "MAX_MAX_CANDIDATES",
    "MAX_MAX_ROUNDS",
    "MAX_SOURCE_SCANS",
    "REDUCTION_ORDER",
    "SHRINK_RESULT_CONTRACT_VERSION",
    "FailureSignature",
    "ScenarioComplexity",
    "ShrinkResult",
    "ShrinkSearchOutcome",
    "encode_shrink_result",
    "extract_failure_signature",
    "is_strictly_smaller",
    "measure_complexity",
    "shrink_result_relative_path",
    "shrink_scenario",
    "signatures_match",
    "unsupported_failure_reason",
    "write_shrink_result",
]
