"""Contracts a custom agent implements to be evaluated by AgentCheck.

AgentCheck supports the OpenAI Agents SDK and PydanticAI by rebuilding the
target from its declared surface, so the original tool handlers are never
reachable. An agent written directly against a model API has no framework
surface to rebuild from, so the same guarantee has to come from somewhere else.

It comes from what AgentCheck is *given*. A custom agent declares its tools as
``ToolDefinition`` -- a name, a schema and risk flags, with no callable
anywhere -- and receives a ``ToolRuntime`` to call them through. AgentCheck
therefore never holds a reference to ``cancel_order``, which is why it cannot
execute it. That is a structural property, not a promise to be careful.

What this does *not* cover is a side effect written directly into the
orchestration loop rather than behind a declared tool. That code is the agent,
so it has to run. Process isolation, an empty environment allowlist and
socket-level network denial catch the common escapes; a local filesystem write
inside reasoning code is not preventable, and the documentation says so rather
than implying a sandbox.

These types remain contracts only. Execution lives in
``agentcheck.adapters.custom.CustomAgentAdapter``; keeping the declarations in
this framework-neutral module lets a target depend on the small public surface
without importing runner internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from agentcheck.domain.agent_spec import ToolDefinition
from agentcheck.domain.run import ToolOutcome


__all__ = ["AsyncToolRuntime", "CustomAgentProtocol", "ToolRuntime", "TurnResult"]


@runtime_checkable
class ToolRuntime(Protocol):
    """The only way a custom agent is allowed to reach a tool.

    Implemented by AgentCheck over the existing ``ToolGateway``, so a call is
    matched against the scenario's fixtures, validated against the declared
    schema, and refused if the tool was never declared. The agent's own
    handlers are not involved at any point.

    Synchronous because ``ToolGateway.invoke`` is synchronous, and because a
    custom loop should not have to become async to be testable. Given to a
    ``start``/``resume`` pair that is itself synchronous; an async pair is
    given ``AsyncToolRuntime`` instead.
    """

    def call(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        """Simulate one tool call and return its canonical outcome.

        The returned ``ToolOutcome`` carries ``status`` and ``error`` as well as
        ``result``, so a loop can react to a simulated failure or an ambiguous
        outcome the way it would react to a real one. Raises rather than
        returning a plausible value when the tool is unknown or the arguments do
        not satisfy the declared schema: a harness that invents tool output
        invents passing runs.
        """
        ...


@runtime_checkable
class AsyncToolRuntime(Protocol):
    """The tool runtime given to an async ``start``/``resume`` pair.

    Exists so a custom loop that awaits a real async provider client (or any
    other real ``await``) does not have to bridge into a synchronous turn
    boundary to do it. It does **not** add support for genuinely concurrent
    tool *dispatch*: ``call`` still runs the same synchronous
    ``ToolGateway`` logic underneath, with no internal ``await`` of its own,
    so one call always runs to completion before the next begins -- even if
    the agent issues them through ``asyncio.gather`` or ``asyncio.create_task``.
    Concurrent-looking syntax therefore still executes, and is fixture- and
    world-state-deterministic, strictly in the order the calls were made, the
    same guarantee the synchronous ``ToolRuntime`` gives.

    What this does not, and cannot, give a custom agent is the same-stage
    "launch group" semantics the OpenAI Agents and PydanticAI adapters support
    for genuinely concurrent dispatch: those adapters derive decision order
    from an observed model response listing several tool calls at once. A
    custom agent's model calls are never observed by AgentCheck at all (see
    ``CustomAgentAdapter``'s docstring), so there is no evidence from which to
    derive a "these were decided together" fact -- inventing one would be
    exactly the kind of guessed verdict this project does not produce. See
    ``docs/concurrent-tool-decisions.md``.
    """

    async def call(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        """Simulate one tool call and return its canonical outcome.

        Same contract as ``ToolRuntime.call``, awaited instead of called
        directly.
        """
        ...


@dataclass
class TurnResult:
    """What one turn of a custom agent produced.

    ``state`` is opaque to AgentCheck and is handed straight back to ``resume``
    on the next turn. It stays in the worker process and is never serialized,
    which keeps arbitrary user objects out of any wire format and keeps pickle
    out of the design entirely.
    """

    output: Any
    state: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CustomAgentProtocol(Protocol):
    """The minimum a custom agent implements to be evaluated.

    AgentCheck drives *turns*; the agent drives its own tool loop within a turn.
    A scenario's opening request becomes ``start``, and each ``followup_turn``
    becomes a ``resume`` -- which is what makes a confirmation flow expressible:
    the agent asks, the scenario answers, and the destructive call happens
    afterwards or not at all.

    ``start``/``resume`` may both be ordinary functions, taking ``ToolRuntime``,
    or both coroutine functions, taking ``AsyncToolRuntime`` instead -- an
    agent whose own loop needs a real ``await`` (a genuine async provider
    client, for example) does not have to fake a synchronous boundary around
    it. The two methods must agree: one sync and one async is a contract
    mismatch AgentCheck refuses at preflight, not a shape it guesses how to
    drive.
    """

    tools: Sequence[ToolDefinition]
    """Declared tools. Names, schemas and risk flags -- never callables."""

    def start(self, message: str, tools: ToolRuntime) -> TurnResult:
        """Handle the scenario's opening user message."""
        ...

    def resume(self, state: Any, message: str, tools: ToolRuntime) -> TurnResult:
        """Continue from a previous turn's state with a follow-up message."""
        ...
