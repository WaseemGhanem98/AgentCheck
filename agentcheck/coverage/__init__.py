"""Declared, observable behavioral coverage for AgentCheck suites."""

from .analyzer import analyze_behavioral_coverage, verify_behavioral_coverage_binding
from .contract import (
    BEHAVIORAL_COVERAGE_CONTRACT_VERSION,
    BehavioralCoverage,
    BehavioralCoverageFamily,
    BehavioralCoverageReferenceScope,
    BehavioralCoverageRequirement,
    BehavioralCoverageStatus,
    BehavioralDimension,
)

__all__ = [
    "BEHAVIORAL_COVERAGE_CONTRACT_VERSION",
    "BehavioralCoverage",
    "BehavioralCoverageFamily",
    "BehavioralCoverageReferenceScope",
    "BehavioralCoverageRequirement",
    "BehavioralCoverageStatus",
    "BehavioralDimension",
    "analyze_behavioral_coverage",
    "verify_behavioral_coverage_binding",
]
