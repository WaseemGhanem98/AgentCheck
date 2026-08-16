from __future__ import annotations


class AgentCheckError(Exception):
    """Base class for actionable AgentCheck failures."""


class ConfigurationError(AgentCheckError):
    """The target or AgentCheck configuration is invalid."""


class AdapterError(AgentCheckError):
    """A framework adapter cannot safely inspect or execute the target."""


class UnsupportedTargetError(AdapterError):
    """The target uses behavior outside the fail-closed Phase 1 support surface."""


class ScenarioValidationError(AgentCheckError):
    """A scenario is invalid and must not affect the agent's score."""


class InfrastructureError(AgentCheckError):
    """AgentCheck could not execute a valid scenario correctly."""
