"""Framework-neutral adapter contracts.

Adapters are the only AgentCheck layer allowed to depend on a framework SDK.  The
runner and evaluator operate on the domain objects returned through this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any, Awaitable, Protocol, runtime_checkable

from agentcheck.domain import canonical_hash

if TYPE_CHECKING:
    from agentcheck.config import ToolRiskDeclaration
    from agentcheck.domain.agent_spec import AgentSpec
    from agentcheck.domain.run import CanonicalRun
    from agentcheck.domain.scenario import ConversationTurn
    from agentcheck.mcp_manifest import McpManifest


class AdapterError(RuntimeError):
    """Base class for framework adapter failures."""


class AdapterDependencyError(AdapterError):
    """Raised when an optional framework dependency is unavailable or incompatible."""


class AdapterRuntimeError(AdapterError):
    """Raised when the framework run fails for adapter or platform reasons."""


def missing_extra_message(subject: str, extra: str) -> str:
    """Name the install command that actually works for this installation.

    Import and distribution names need not match, so the providing distribution
    is discovered from installed metadata instead of being guessed.
    """

    distribution = "agentcheck"
    try:
        providers = importlib_metadata.packages_distributions().get("agentcheck") or ()
    except Exception:  # pragma: no cover - metadata is best-effort, never fatal
        providers = ()
    if providers and "agentcheck" not in providers:
        distribution = sorted(providers)[0]
    return (
        f"{subject} requires the `{extra}` extra "
        f"(`pip install '{distribution}[{extra}]'`)."
    )


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


def require_known_tool_risk_names(
    spec: "AgentSpec", declared_tool_risk: "Mapping[str, ToolRiskDeclaration] | None"
) -> None:
    """Refuse a run whose ``tool_risk`` (``agentcheck.json``) names an unknown tool.

    ``declared_risk_for`` looks an override up by name; a key that matches no
    declared tool is never read by anything downstream, which makes a
    misspelled tool name in ``tool_risk`` a silently ignored override rather
    than the risk correction the developer asked for. Called from every
    adapter's ``prepare``, once inspection has the full declared tool set, so
    the mismatch is refused before any scenario runs rather than passed
    through as if it had been applied.
    """

    from agentcheck.inspect.risk_authority import unmatched_tool_risk_names

    unmatched = unmatched_tool_risk_names(
        tuple(item.value.name for item in spec.tools.items), declared_tool_risk
    )
    if not unmatched:
        return
    raise UnsupportedTargetError(
        [
            SupportIssue(
                code="unknown_tool_risk_declaration",
                message=(
                    f"agentcheck.json declares tool_risk for {name!r}, which is "
                    "not one of this agent's declared tools. This override "
                    "would never be applied, so AgentCheck refuses the run "
                    "rather than silently ignoring it: fix the tool name, or "
                    "remove the entry."
                ),
                location=f"config.tool_risk.{name}",
            )
            for name in unmatched
        ]
    )


# (module-name prefix, adapter name) -- pure duck-typing, no cross-adapter
# import. See ``guess_other_adapter``.
_FRAMEWORK_MODULE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("agents.", "openai_agents"),
    ("pydantic_ai.", "pydantic_ai"),
)


def guess_other_adapter(target: Any) -> str | None:
    """A best-effort guess at which adapter a wrong-type target actually needs.

    Pure name-based duck-typing on the object already imported at the
    configured entrypoint -- its class's module and name, nothing more. This
    deliberately does not import another framework's SDK to check with
    ``isinstance``: the ``openai-agents`` and ``pydantic-ai`` extras are
    independent on purpose (see ``pyproject.toml``), and evaluating one
    target should never require the other framework's package to be
    installed just to produce a better error message. A wrong or missing
    guess only costs a slightly less helpful message -- the caller still
    refuses the run either way, this never changes what gets accepted.
    """

    if type(target).__name__ != "Agent":
        return None
    module = type(target).__module__
    for prefix, adapter in _FRAMEWORK_MODULE_PREFIXES:
        if module == prefix.rstrip(".") or module.startswith(prefix):
            return adapter
    return None


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
            if set(edge) - {
                "tool_name",
                "target_agent",
                "location",
                "issue_codes",
                "context_assignments",
            }:
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
            assignments = edge.get("context_assignments")
            if assignments is not None:
                if not isinstance(assignments, list):
                    raise ValueError(
                        "topology handoff context_assignments must be a list"
                    )
                for item in assignments:
                    if not isinstance(item, dict) or set(item) != {"field", "value"}:
                        raise ValueError(
                            "topology context_assignments must be field/value objects"
                        )
                    if not isinstance(item.get("field"), str) or not item["field"]:
                        raise ValueError(
                            "topology context assignment requires a field name"
                        )
                    value = item.get("value")
                    if value is not None and type(value) not in (bool, int, str):
                        raise ValueError(
                            "topology context assignment value must be a JSON scalar"
                        )
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


def portable_identity(
    fingerprint: dict[str, Any],
    *,
    location_locator: str | None,
    identity_locator: str | None,
) -> tuple[str, str | None]:
    """Return ``(spec_id, legacy_spec_id)`` for one inspected fingerprint.

    ``fingerprint`` arrives with every semantic input already populated: name,
    framework version, model, instructions digest, declared tools, and any
    handoff graph. Only the locator differs between the two identities, so the
    legacy value is the same semantic surface hashed with the location-bound
    locator restored.

    Returning the legacy identity is what lets an artifact created before this
    contract still be recognized at the location that produced it, without ever
    treating a different location as equivalent. When no portable locator was
    supplied there is no separate legacy identity to record.
    """

    if identity_locator is None:
        return _spec_id(fingerprint, location_locator), None
    portable = _spec_id(fingerprint, identity_locator)
    legacy = _spec_id(fingerprint, location_locator)
    return portable, (None if legacy == portable else legacy)


def _spec_id(fingerprint: dict[str, Any], locator: str | None) -> str:
    # canonical_hash sorts keys, so replacing the locator reproduces the exact
    # bytes the location-bound contract hashed.
    return f"agentspec-{canonical_hash({**fingerprint, 'source': locator}).split(':', 1)[1][:24]}"


class FrameworkAdapter(ABC):
    """Contract implemented by every supported agent framework."""

    framework: str

    @abstractmethod
    def inspect(
        self,
        target: Any,
        *,
        source: str | None = None,
        identity_locator: str | None = None,
        declared_tool_risk: "Mapping[str, ToolRiskDeclaration] | None" = None,
        mcp_manifest: "McpManifest | None" = None,
    ) -> "AgentSpec":
        """Extract an explicit, versioned specification without running the model.

        ``source`` locates the target for evidence and diagnostics and may be an
        absolute path. ``identity_locator`` is the portable entrypoint, relative
        to the target root, and is the only locator allowed to influence
        ``spec_id``. When it is omitted the caller has not established a target
        root, so identity falls back to ``source`` and stays location-bound.

        ``declared_tool_risk`` is the developer's ``tool_risk`` declaration from
        ``agentcheck.json`` (see ``agentcheck.config.ToolRiskDeclaration``),
        keyed by declared tool name. It is authoritative over whatever this
        adapter would otherwise infer for the named tool's ``state_changing``
        and ``destructive`` axes; an axis a declaration does not name still
        falls through to inference. Omitting it keeps prior behaviour exactly:
        every tool resolves through inference or ``UNKNOWN`` alone.

        ``mcp_manifest`` is the developer's frozen snapshot of an external
        (non-function) toolset's tool schemas -- see
        ``agentcheck.mcp_manifest.McpManifest``. An adapter with no concept of
        an external toolset ignores it. Omitting it keeps prior behaviour
        exactly: a target with any such toolset stays unsupported.
        """

    @abstractmethod
    def preflight(
        self, target: Any, *, mcp_manifest: "McpManifest | None" = None
    ) -> PreflightReport:
        """Decide whether every executable surface can be intercepted safely.

        ``mcp_manifest`` has the same meaning as in ``inspect``: when given, it
        is the developer taking explicit responsibility for an external
        toolset's schemas, which is what allows that toolset to stop being an
        unconditional ``unsupported_toolset`` finding for adapters that
        support one.
        """

    @abstractmethod
    def prepare(
        self,
        target: Any,
        gateway: ToolGatewayProtocol,
        *,
        world_state: Any = None,
        event_sink: EventSinkProtocol | None = None,
        source: str | None = None,
        identity_locator: str | None = None,
        controlled_model: bool = False,
        declared_tool_risk: "Mapping[str, ToolRiskDeclaration] | None" = None,
        mcp_manifest: "McpManifest | None" = None,
    ) -> PreparedTarget:
        """Return a sanitized runtime target or fail before model execution.

        ``controlled_model`` substitutes a deterministic offline model for the
        target's provider. The target agent is otherwise unchanged.

        ``declared_tool_risk`` is forwarded to ``inspect`` so the spec attached
        to the prepared target, and the risk markers used to build the actual
        runtime invokers, resolve through the same developer declarations
        rather than disagreeing with each other.

        ``mcp_manifest`` is forwarded to ``inspect`` and ``preflight`` for the
        same reason: one declaration, read consistently everywhere it matters.
        """

    def describe_topology(
        self, target: Any, *, source: str | None = None
    ) -> dict[str, Any] | None:
        """Optional multi-agent topology, as additive inspect diagnostics.

        Not abstract: a framework whose targets are a single agent has no
        topology to describe, and returning ``None`` is the honest answer rather
        than an empty graph.
        """

        del target, source
        return None

    @abstractmethod
    async def run(
        self,
        prepared: PreparedTarget,
        input_text: str | Sequence["ConversationTurn"],
        *,
        run_id: str,
        max_turns: int,
        followup_turns: Sequence["ConversationTurn"] = (),
        scenario_id: str | None = None,
        target_id: str | None = None,
    ) -> "CanonicalRun":
        """Execute one prepared target and normalize its observable behavior.

        ``followup_turns`` are scripted user replies the adapter withholds until
        the agent has finished answering, delivering one per completed execution
        stage. The whole scenario stays one prepared target, one gateway, one
        budget, and one ``CanonicalRun``; stages are not independent runs.
        """
