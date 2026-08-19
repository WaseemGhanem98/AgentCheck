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

from agentcheck.domain import ContractModel, JsonObject


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


class FixturePack(ContractModel):
    """Representative input values for a target. Unknown fields are rejected."""

    schema_version: Literal["agentcheck.fixtures.v1"] = FIXTURE_PACK_CONTRACT_VERSION
    tools: dict[str, ToolInputValues] = Field(default_factory=dict)


__all__ = [
    "DEFAULT_FIXTURES_FILENAME",
    "FIXTURE_PACK_CONTRACT_VERSION",
    "FixturePack",
    "ToolInputValues",
]
