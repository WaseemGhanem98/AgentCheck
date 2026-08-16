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


class UnsupportedTargetError(AdapterError):
    """Raised before execution when a target cannot be intercepted safely."""

    def __init__(self, issues: list["SupportIssue"]):
        self.issues = list(issues)
        summary = "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)
        super().__init__(summary or "The target is not supported by this adapter")


class AdapterRuntimeError(AdapterError):
    """Raised when the framework run fails for adapter or platform reasons."""


@dataclass(frozen=True)
class SupportIssue:
    """One actionable reason a target cannot be run safely."""

    code: str
    message: str
    location: str | None = None


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
