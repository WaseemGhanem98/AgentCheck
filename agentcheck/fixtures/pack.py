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
