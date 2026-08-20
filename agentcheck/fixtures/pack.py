"""Developer-supplied representative input values, as inert versioned data.

A tool may declare only ``confirmation_number: string`` with a prose
description. Nothing in that contract says a realistic value looks like
``TEST-ABC123``, so generation falls back to an obviously synthetic string and
the action path stays shallow: a model asked to act on
``agentcheck-boundary-value`` may reasonably decline.

This is the developer's answer to that. It supplies what the *user side* of a
scenario provides, which is a different thing from
:class:`~agentcheck.domain.scenario.ToolFixture`: that one says what a tool
*returns* once called. Input values shape the request; outcome fixtures shape
the reply. Keeping them apart keeps "what the user said" from being confused
with "what the tool did".

Values are committed test data, so the loader refuses anything credential
shaped and this format has no way to name a secret. Secrets are deliberately
unsupported here rather than half-supported: see ``loader.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agentcheck.domain import ContractModel, JsonObject, JsonValue


FIXTURE_PACK_CONTRACT_VERSION: Literal["agentcheck.fixtures.v1"] = (
    "agentcheck.fixtures.v1"
)

DEFAULT_FIXTURES_FILENAME = "agentcheck-fixtures.json"


class ToolInputValues(ContractModel):
    """Representative arguments for one tool, keyed by parameter name."""

    arguments: JsonObject = Field(default_factory=dict)
    # The human-facing half of the same problem. Valid argument values are not
    # enough on their own: a request assembled from a schema reads as a data
    # handover ("reason is ..., original_sender is ..."), and a capable model
    # answers it without acting, which leaves the action path unexercised and
    # every trajectory check on it vacuous. Measured on two unrelated targets.
    # A developer knows the situation that genuinely calls for the tool, so
    # this is where they write it. Frozen into the suite like any other input,
    # so the run stays deterministic and replayable; a future authoring layer
    # can emit this field rather than deciding anything at run time.
    user_request: str | None = Field(default=None, min_length=1, max_length=8_000)


class PrerequisiteOutcome(ContractModel):
    """What a gating tool returns when it is called on the way to an action."""

    # Optional: a tool whose result the agent only needs to have *succeeded*
    # gets the same generic acknowledgement every other generated fixture uses.
    # A tool whose returned value the next call depends on -- an identifier
    # looked up before it can be acted on -- needs a real value here, and only
    # the developer knows what that looks like.
    result: JsonValue = None


class FixturePack(ContractModel):
    """Representative input values for a target. Unknown fields are rejected."""

    schema_version: Literal["agentcheck.fixtures.v1"] = FIXTURE_PACK_CONTRACT_VERSION
    tools: dict[str, ToolInputValues] = Field(default_factory=dict)
    # Separate from ``tools`` on purpose, and the separation is the same one
    # this module's docstring draws: ``tools`` shapes the request, this shapes a
    # reply. It is also a different claim about a different tool. An entry here
    # says "this tool may legitimately be called on the way to some *other*
    # tool's action", which is exactly the thing that cannot be inferred:
    # classification is lexical and already over-reaches, so a heuristic
    # excluded the lookup that gated the action on a real target. Naming them is
    # narrow, general, and fails closed -- an undeclared tool still has no
    # fixture, and an in-contract call to it still stops the case.
    prerequisites: dict[str, PrerequisiteOutcome] = Field(default_factory=dict)


__all__ = [
    "DEFAULT_FIXTURES_FILENAME",
    "FIXTURE_PACK_CONTRACT_VERSION",
    "FixturePack",
    "PrerequisiteOutcome",
    "ToolInputValues",
]
