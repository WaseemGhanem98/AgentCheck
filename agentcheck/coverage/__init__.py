"""Declared, observable behavioral coverage for AgentCheck suites."""

from .analyzer import (
    UnmetRiskObligation,
    analyze_behavioral_coverage,
    behavioral_coverage_spec_digest,
    risk_obligations_for_spec,
    unmet_risk_obligations,
    verify_behavioral_coverage_binding,
)
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
    "UnmetRiskObligation",
    "analyze_behavioral_coverage",
    "behavioral_coverage_spec_digest",
    "risk_obligations_for_spec",
    "unmet_risk_obligations",
    "verify_behavioral_coverage_binding",
]
