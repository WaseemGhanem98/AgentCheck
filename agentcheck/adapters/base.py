"""Framework-neutral adapter contracts.

Adapters are the only AgentCheck layer allowed to depend on a framework SDK.  The
runner and evaluator operate on the domain objects returned through this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentcheck.domain.agent_spec import AgentSpec
    from agentcheck.domain.run import CanonicalRun
    from agentcheck.domain.scenario import ConversationTurn


class AdapterError(RuntimeError):
    """Base class for framework adapter failures."""


class AdapterDependencyError(AdapterError):
    """Raised when an optional framework dependency is unavailable or incompatible."""


class AdapterRuntimeError(AdapterError):
    """Raised when the framework run fails for adapter or platform reasons."""


@dataclass(frozen=True)
class SupportIssue:
    """One actionable reason a target cannot be run safely."""

    code: str
    message: str
    location: str | None = None


def format_support_issues(
    issues: Sequence[SupportIssue],
    *,
    heading: str = "The target is not supported by this adapter:",
) -> str:
    """Render adapter preflight codes for CLI and exception text."""

    if not issues:
        return heading.rstrip(":")
    lines = [heading]
    for issue in issues:
        location = f" ({issue.location})" if issue.location else ""
        lines.append(f"- {issue.code}{location}: {issue.message}")
    return "\n".join(lines)


def encode_preflight_report(report: PreflightReport) -> dict[str, Any]:
    """JSON-safe inspect diagnostic payload; not part of AgentSpec."""

    return {
        "framework": report.framework,
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "location": issue.location,
            }
            for issue in report.issues
        ],
    }


def decode_topology(payload: Any) -> dict[str, Any]:
    """Parse the additive inspect topology object, failing closed on malformation.

    Topology is diagnostic data beside the versioned ``AgentSpec`` result, in the
    same way the ``preflight`` object is; it is not ``agentcheck.agent_spec.v1``.
    """

    if not isinstance(payload, dict):
        raise ValueError("topology must be an object")
    extra = set(payload) - {"framework", "agents"}
    if extra:
        raise ValueError("topology has unknown fields")
    framework = payload.get("framework")
    if not isinstance(framework, str) or not framework.strip():
        raise ValueError("topology requires a framework")
    agents = payload.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("topology requires a non-empty agents list")
    for agent in agents:
        if not isinstance(agent, dict):
            raise ValueError("topology agents must be objects")
        if set(agent) - {
            "name",
            "location",
            "model",
            "instructions_static",
            "tool_names",
            "handoffs",
        }:
            raise ValueError("topology agent has unknown fields")
        if not isinstance(agent.get("name"), str) or not agent["name"]:
            raise ValueError("topology agent requires a name")
        if not isinstance(agent.get("location"), str) or not agent["location"]:
            raise ValueError("topology agent requires a location")
        if not isinstance(agent.get("instructions_static"), bool):
            raise ValueError("topology agent requires instructions_static")
        tool_names = agent.get("tool_names")
        if not isinstance(tool_names, list) or any(
            not isinstance(name, str) for name in tool_names
        ):
            raise ValueError("topology agent tool_names must be strings")
        handoffs = agent.get("handoffs")
        if not isinstance(handoffs, list):
            raise ValueError("topology agent handoffs must be a list")
        for edge in handoffs:
            if not isinstance(edge, dict):
                raise ValueError("topology handoffs must be objects")
            if set(edge) - {"tool_name", "target_agent", "location", "issue_codes"}:
                raise ValueError("topology handoff has unknown fields")
            if not isinstance(edge.get("location"), str) or not edge["location"]:
                raise ValueError("topology handoff requires a location")
            if edge.get("tool_name") is not None and not isinstance(
                edge["tool_name"], str
            ):
                raise ValueError("topology handoff tool_name must be a string or null")
            if edge.get("target_agent") is not None and not isinstance(
                edge["target_agent"], str
            ):
                raise ValueError("topology handoff target_agent must be a string or null")
            issue_codes = edge.get("issue_codes")
            if not isinstance(issue_codes, list) or any(
                not isinstance(code, str) for code in issue_codes
            ):
                raise ValueError("topology handoff issue_codes must be strings")
    return payload


def decode_preflight_report(payload: Any) -> PreflightReport:
    """Parse a worker inspect preflight payload, failing closed on malformation."""

    if not isinstance(payload, dict):
        raise ValueError("preflight report must be an object")
    extra = set(payload) - {"framework", "issues"}
    if extra:
        raise ValueError("preflight report has unknown fields")
    framework = payload.get("framework")
    if not isinstance(framework, str) or not framework.strip():
        raise ValueError("preflight report requires a framework")
    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list):
        raise ValueError("preflight report issues must be a list")
    issues: list[SupportIssue] = []
    for item in raw_issues:
        if not isinstance(item, dict):
            raise ValueError("preflight issue must be an object")
        extra_issue = set(item) - {"code", "message", "location"}
        if extra_issue:
            raise ValueError("preflight issue has unknown fields")
        code = item.get("code")
        message = item.get("message")
        location = item.get("location")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("preflight issue requires a code")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("preflight issue requires a message")
        if location is not None and not isinstance(location, str):
            raise ValueError("preflight issue location must be a string or null")
        issues.append(
            SupportIssue(
                code=code.strip(),
                message=message.strip(),
                location=location.strip() if isinstance(location, str) else None,
            )
        )
    return PreflightReport(framework=framework.strip(), issues=tuple(issues))


class UnsupportedTargetError(AdapterError):
    """Raised before execution when a target cannot be intercepted safely."""

    def __init__(self, issues: list[SupportIssue]):
        self.issues = list(issues)
        super().__init__(
            format_support_issues(self.issues)
            if self.issues
            else "The target is not supported by this adapter"
        )


@dataclass(frozen=True)
class PreflightReport:
    """Static support decision made before any model or tool execution."""

    framework: str
    issues: tuple[SupportIssue, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.issues

    def require_supported(self) -> None:
        if self.issues:
            raise UnsupportedTargetError(list(self.issues))


@dataclass(frozen=True)
class GatewayRequest:
    """Provider-neutral invocation data passed to a controlled tool gateway."""

    tool_name: str
    arguments: dict[str, Any]
    raw_arguments: str
    call_id: str


@runtime_checkable
class ToolGatewayProtocol(Protocol):
    """Minimum gateway interface consumed by framework adapters.

    ``world_state`` is deliberately opaque here.  The runner owns its concrete
    representation; an adapter only forwards it unchanged.
    """

    def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        world_state: Any,
    ) -> Any | Awaitable[Any]: ...


@runtime_checkable
class EventSinkProtocol(Protocol):
    """Optional live event sink used while a framework run is in progress."""

    def emit(self, event: Any) -> Any | Awaitable[Any]: ...


@dataclass
class PreparedTarget:
    """An SDK runtime object whose live tool handlers have been removed."""

    framework: str
    runtime_agent: Any
    spec: "AgentSpec"
    tool_names: tuple[str, ...]
    gateway: ToolGatewayProtocol
    world_state: Any = None
    event_sink: EventSinkProtocol | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FrameworkAdapter(ABC):
    """Contract implemented by every supported agent framework."""

    framework: str

    @abstractmethod
    def inspect(self, target: Any, *, source: str | None = None) -> "AgentSpec":
        """Extract an explicit, versioned specification without running the model."""

    @abstractmethod
    def preflight(self, target: Any) -> PreflightReport:
        """Decide whether every executable surface can be intercepted safely."""

    @abstractmethod
    def prepare(
        self,
        target: Any,
        gateway: ToolGatewayProtocol,
        *,
        world_state: Any = None,
        event_sink: EventSinkProtocol | None = None,
        source: str | None = None,
    ) -> PreparedTarget:
        """Return a sanitized runtime target or fail before model execution."""

    @abstractmethod
    async def run(
        self,
        prepared: PreparedTarget,
        input_text: str | Sequence["ConversationTurn"],
        *,
        run_id: str,
        max_turns: int,
        scenario_id: str | None = None,
        target_id: str | None = None,
    ) -> "CanonicalRun":
        """Execute one prepared target and normalize its observable behavior."""
