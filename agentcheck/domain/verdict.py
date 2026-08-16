"""Case and assertion verdicts."""

from __future__ import annotations

from enum import Enum


class Verdict(str, Enum):
    """The only four primary AgentCheck outcomes."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    INFRA_ERROR = "INFRA_ERROR"
