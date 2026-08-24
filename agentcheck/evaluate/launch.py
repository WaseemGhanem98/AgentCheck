"""Derived launch-stage relations for one recorded run.

AgentCheck's question is not "which call ran first" but "what had the agent
actually observed when it decided to act". A model response can carry several
tool calls at once. Those calls are chosen together, before any of their
results exist, so executing them in some order does not mean the agent learned
anything from the earlier one.

Everything here is derived from evidence the adapters already record:

* a ``TOOL_ATTEMPT`` event names the ``MODEL_RESPONSE`` that launched it, so
  attempts sharing that response were decided in the same stage;
* a ``TOOL_RESULT`` event names the attempt it belongs to, so a result has a
  position in the same sequence as the responses.

No stored contract changes, and a run recorded before this analysis existed can
still be analyzed. An adapter that does not observe the target's model
responses -- a custom agent owns its own model calls -- yields no launch
evidence, and every relation stays unknown rather than being assumed serial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from agentcheck.domain import CanonicalEventType, CanonicalRun


@dataclass(frozen=True, slots=True)
class LaunchAnalysis:
    """Which attempts were decided together, and what preceded which decision."""

    launch_event_by_attempt: Mapping[str, str | None]
    launch_sequence_by_attempt: Mapping[str, int | None]
    result_sequence_by_attempt: Mapping[str, int | None]

    def launch_group(self, attempt_id: str) -> str | None:
        """Return the model response that decided this attempt, if observed."""

        return self.launch_event_by_attempt.get(attempt_id)

    def same_launch_group(self, first: str, second: str) -> bool | None:
        """Return whether both attempts were decided in the same stage."""

        left = self.launch_group(first)
        right = self.launch_group(second)
        if left is None or right is None:
            return None
        return left == right

    def observed_before(self, earlier: str, later: str) -> bool | None:
        """Return whether ``earlier``'s result was known when ``later`` was decided.

        ``True`` only when the result is recorded ahead of the model response
        that launched ``later``. Two attempts from one response are ``False``:
        the agent chose both before either had produced anything. ``None`` when
        the run does not record enough to decide, which is not evidence of
        either answer.
        """

        result_sequence = self.result_sequence_by_attempt.get(earlier)
        decision_sequence = self.launch_sequence_by_attempt.get(later)
        if result_sequence is None or decision_sequence is None:
            return None
        return result_sequence < decision_sequence


def analyze_launches(run: CanonicalRun) -> LaunchAnalysis:
    """Derive launch-stage relations from a recorded run."""

    sequence_by_event = {event.event_id: event.sequence for event in run.events}
    response_events = {
        event.event_id
        for event in run.events
        if event.event_type == CanonicalEventType.MODEL_RESPONSE
    }
    event_by_id = {event.event_id: event for event in run.events}

    launch_event: dict[str, str | None] = {}
    launch_sequence: dict[str, int | None] = {}
    for attempt in run.tool_attempts:
        source = event_by_id.get(attempt.event_id)
        origin = None
        if source is not None:
            origin = next(
                (
                    candidate
                    for candidate in source.source_event_ids
                    if candidate in response_events
                ),
                None,
            )
        launch_event[attempt.attempt_id] = origin
        launch_sequence[attempt.attempt_id] = (
            sequence_by_event.get(origin) if origin is not None else None
        )

    attempt_by_event = {
        attempt.event_id: attempt.attempt_id for attempt in run.tool_attempts
    }
    result_sequence: dict[str, int | None] = {
        attempt.attempt_id: None for attempt in run.tool_attempts
    }
    for event in run.events:
        if event.event_type is not CanonicalEventType.TOOL_RESULT:
            continue
        for source_id in event.source_event_ids:
            attempt_id = attempt_by_event.get(source_id)
            if attempt_id is None:
                continue
            recorded = result_sequence.get(attempt_id)
            # Keep the first recorded result: a later one would describe a
            # different observation than the decision under test.
            if recorded is None or event.sequence < recorded:
                result_sequence[attempt_id] = event.sequence

    return LaunchAnalysis(
        launch_event_by_attempt=launch_event,
        launch_sequence_by_attempt=launch_sequence,
        result_sequence_by_attempt=result_sequence,
    )


__all__ = ["LaunchAnalysis", "analyze_launches"]
