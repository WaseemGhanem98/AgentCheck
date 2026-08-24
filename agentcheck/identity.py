"""Portable target identity and its bounded legacy compatibility path.

``spec_id`` names the agent under test: its declared name, framework version,
model, instructions, tools and tool schemas, plus the entrypoint relative to the
target root. It deliberately excludes the absolute checkout location, so the
same unchanged agent inspects identically on a developer machine and on a CI
runner.

Before that contract existed the absolute entrypoint path was hashed into
``spec_id``. Artifacts created then are bound to the location that produced
them. They stay usable exactly there and nowhere else: recognizing one is proof
that this inspection reproduces the recorded location-bound identity, which is
the same evidence the old contract required. It is never proof that a different
location is equivalent, so nothing is migrated and no equivalence is invented.
"""

from __future__ import annotations

import hmac

from agentcheck.domain import AgentSpec


# A legacy artifact carried in from another directory and a genuine change to
# the behavioral contract are indistinguishable from the recorded identity
# alone, so this states the migration path conditionally instead of diagnosing
# a cause it cannot observe.
IDENTITY_MISMATCH_HINT = (
    ". If this artifact predates portable target identity it is bound to the "
    "directory that created it, and regenerating it from this checkout will "
    "produce a portable identity."
)


def _equal(left: str, right: str) -> bool:
    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    if len(left_bytes) != len(right_bytes):
        hmac.compare_digest(left_bytes, left_bytes)
        return False
    return hmac.compare_digest(left_bytes, right_bytes)


def spec_identity_matches(spec: AgentSpec, recorded_spec_id: str) -> bool:
    """Return whether ``recorded_spec_id`` identifies this inspected target.

    A portable match is the normal case. A legacy match is accepted only when
    this very inspection reproduces the recorded location-bound identity, which
    can happen only at the original path with an unchanged behavioral surface.
    """

    if _equal(spec.spec_id, recorded_spec_id):
        return True
    legacy = spec.legacy_spec_id
    return legacy is not None and _equal(legacy, recorded_spec_id)


def spec_identity_is_legacy(spec: AgentSpec, recorded_spec_id: str) -> bool:
    """Return whether the match came only from the legacy identity."""

    if _equal(spec.spec_id, recorded_spec_id):
        return False
    legacy = spec.legacy_spec_id
    return legacy is not None and _equal(legacy, recorded_spec_id)


def identity_mismatch_hint(spec: AgentSpec, recorded_spec_id: str) -> str:
    """Offer the migration path on a refusal, without diagnosing its cause.

    Nothing is added when this target has no separate location-bound identity,
    because then no artifact of this target could have predated the portable
    contract, or when the recorded identity is that legacy value, because the
    caller has already accepted the match.
    """

    if spec.legacy_spec_id is None or _equal(spec.legacy_spec_id, recorded_spec_id):
        return ""
    return IDENTITY_MISMATCH_HINT


__all__ = [
    "IDENTITY_MISMATCH_HINT",
    "identity_mismatch_hint",
    "spec_identity_is_legacy",
    "spec_identity_matches",
]
