"""Optional developer-supplied representative input values.

A tool may declare only ``confirmation_number: string`` with a prose
description, so generation falls back to a generic synthetic string and a model
may reasonably decline to act on it. These tests pin the way out of that: the
developer supplies representative values, AgentCheck validates them against the
tool contract as it exists now, and the suite says plainly which action paths
are representative and which are still shallow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agents import Agent, function_tool

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.config import AgentCheckConfig
from agentcheck.errors import ConfigurationError
from agentcheck.fixtures import (
    DEFAULT_FIXTURES_FILENAME,
    load_fixture_pack,
    load_representative_inputs,
    load_scenario_requests,
)
from agentcheck.generate.boundaries import build_positive_path_cases
from agentcheck.generate.suite import CaseOrigin, build_frozen_suite


@function_tool
def update_seat(confirmation_number: str, new_seat: str) -> str:
    """Update the seat for a given confirmation number."""
    raise AssertionError("original handler must never run")


@function_tool
def set_cabin(cabin: str, passengers: int) -> str:
    """Set the cabin and passenger count."""
    raise AssertionError("original handler must never run")


def _spec(*tools):
    return OpenAIAgentsAdapter().inspect(
        Agent(name="T", instructions="Assist.", tools=list(tools), model="gpt-4.1-mini")
    )


def _write(root: Path, document: dict) -> Path:
    path = root / DEFAULT_FIXTURES_FILENAME
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _pack(**arguments) -> dict:
    return {
        "schema_version": "agentcheck.fixtures.v1",
        "tools": {"update_seat": {"arguments": arguments}},
    }


# --- loading -------------------------------------------------------------


def test_absent_file_is_not_an_error(tmp_path: Path) -> None:
    """Fixtures are optional; a target without them still gets coverage."""

    assert load_fixture_pack(tmp_path) is None
    assert load_representative_inputs(tmp_path, _spec(update_seat)) == {}


def test_valid_fixture_loads(tmp_path: Path) -> None:
    _write(tmp_path, _pack(confirmation_number="TEST-ABC123", new_seat="14A"))

    assert load_representative_inputs(tmp_path, _spec(update_seat)) == {
        "update_seat": {"confirmation_number": "TEST-ABC123", "new_seat": "14A"}
    }


def test_malformed_json_fails_closed(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_FIXTURES_FILENAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="invalid"):
        load_fixture_pack(tmp_path)


def test_unknown_schema_version_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path, {"schema_version": "agentcheck.fixtures.v99", "tools": {}})

    with pytest.raises(ConfigurationError):
        load_fixture_pack(tmp_path)


def test_unknown_field_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path, {"schema_version": "agentcheck.fixtures.v1", "tools": {}, "x": 1})

    with pytest.raises(ConfigurationError):
        load_fixture_pack(tmp_path)


def test_oversized_file_fails_closed(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_FIXTURES_FILENAME).write_text(
        " " * (64 * 1024 + 10), encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match="byte fixture limit"):
        load_fixture_pack(tmp_path)


def test_path_traversal_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_fixture_pack(tmp_path, filename="../escape.json")


def test_symlinked_fixture_is_refused(tmp_path: Path) -> None:
    """Consistent with the other contained loaders: no following links out."""

    outside = tmp_path.parent / "outside-fixtures.json"
    outside.write_text(json.dumps(_pack(new_seat="14A")), encoding="utf-8")
    link = tmp_path / DEFAULT_FIXTURES_FILENAME
    link.symlink_to(outside)

    # Refused by containment before the file is ever opened.
    with pytest.raises(ConfigurationError, match="inside the target directory"):
        load_fixture_pack(tmp_path)


# --- validation against the live contract --------------------------------


def test_unknown_tool_fails_closed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "schema_version": "agentcheck.fixtures.v1",
            "tools": {"no_such_tool": {"arguments": {"a": "b"}}},
        },
    )

    with pytest.raises(ConfigurationError, match="unknown tool"):
        load_representative_inputs(tmp_path, _spec(update_seat))


def test_stale_parameter_after_schema_change_fails_closed(tmp_path: Path) -> None:
    """A renamed argument must be reported, not silently ignored."""

    _write(tmp_path, _pack(seat_number="14A"))

    with pytest.raises(ConfigurationError, match="unknown parameter"):
        load_representative_inputs(tmp_path, _spec(update_seat))


def test_schema_invalid_value_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path, _pack(confirmation_number=1234))

    with pytest.raises(ConfigurationError, match="does not satisfy the declared schema"):
        load_representative_inputs(tmp_path, _spec(update_seat))


def test_credential_shaped_value_is_refused(tmp_path: Path) -> None:
    """Fixtures are committed test data, so they must not carry secrets."""

    _write(tmp_path, _pack(confirmation_number="sk-live-abcdef0123456789abcdef0123"))

    with pytest.raises(ConfigurationError, match="looks like a credential"):
        load_representative_inputs(tmp_path, _spec(update_seat))


def test_generated_placeholder_is_refused_with_an_actionable_error(
    tmp_path: Path,
) -> None:
    """The template marker is not evidence of a representative input."""

    _write(tmp_path, _pack(confirmation_number="REPLACE_ME"))

    with pytest.raises(
        ConfigurationError,
        match=(
            r"update_seat\.confirmation_number still contains the generated "
            r"REPLACE_ME placeholder; replace it with a representative synthetic "
            r"test value before generating or testing the suite"
        ),
    ):
        load_representative_inputs(tmp_path, _spec(update_seat))


def test_environment_reference_is_refused_rather_than_half_supported(
    tmp_path: Path,
) -> None:
    """A model-supplied argument would have to reach the frozen suite to work.

    Rather than serialise a credential or pretend a placeholder is usable, the
    format refuses it and explains where credentials do belong.
    """

    _write(tmp_path, _pack(confirmation_number={"$env": "TEST_CONFIRMATION"}))

    with pytest.raises(ConfigurationError, match=r"\$env"):
        load_representative_inputs(tmp_path, _spec(update_seat))


def test_partial_fixture_is_allowed(tmp_path: Path) -> None:
    """Supplying one argument must not require supplying all of them."""

    _write(tmp_path, _pack(new_seat="14A"))
    supplied = load_representative_inputs(tmp_path, _spec(update_seat))

    assert supplied == {"update_seat": {"new_seat": "14A"}}


# --- generation ----------------------------------------------------------


def _positive(spec, supplied=None):
    return build_positive_path_cases(spec, seed=1729, representative_inputs=supplied)


def test_positive_request_consumes_supplied_values() -> None:
    cases = _positive(
        _spec(update_seat),
        {"update_seat": {"confirmation_number": "TEST-ABC123", "new_seat": "14A"}},
    )

    content = cases[0].scenario.conversation_turns[0].content
    assert "TEST-ABC123" in content and "14A" in content
    assert "agentcheck-boundary-value" not in content
    assert cases[0].scenario.allowed_tool_behavior[0].arguments_match == {
        "confirmation_number": "TEST-ABC123",
        "new_seat": "14A",
    }


def test_coverage_reports_shallow_without_fixtures() -> None:
    (case,) = _positive(_spec(update_seat))

    assert not case.representative
    assert case.shallow_parameters == ("confirmation_number", "new_seat")


def test_coverage_reports_representative_with_fixtures() -> None:
    (case,) = _positive(
        _spec(update_seat),
        {"update_seat": {"confirmation_number": "TEST-ABC123", "new_seat": "14A"}},
    )

    assert case.representative
    assert case.shallow_parameters == ()


def test_partially_supplied_tool_reports_only_the_missing_parameter() -> None:
    (case,) = _positive(_spec(update_seat), {"update_seat": {"new_seat": "14A"}})

    assert case.shallow_parameters == ("confirmation_number",)


def test_schema_constrained_parameters_are_not_shallow() -> None:
    """A value the contract actually pins down needs no developer input."""

    (case,) = _positive(_spec(set_cabin))

    # passengers is an integer the schema constrains; only the free string is shallow.
    assert "passengers" not in case.shallow_parameters


def test_generation_is_deterministic_with_fixtures() -> None:
    supplied = {"update_seat": {"confirmation_number": "TEST-ABC123", "new_seat": "14A"}}
    spec = _spec(update_seat)

    first, second = _positive(spec, supplied), _positive(spec, supplied)

    assert first[0].scenario.fingerprint == second[0].scenario.fingerprint


# --- suite integration ---------------------------------------------------


def _suite(spec, supplied=None, **kwargs):
    return build_frozen_suite(
        spec, AgentCheckConfig(), seed=1729, representative_inputs=supplied, **kwargs
    )


def test_suite_coverage_surfaces_the_distinction() -> None:
    spec = _spec(update_seat)

    shallow = _suite(spec)
    representative = _suite(
        spec, {"update_seat": {"confirmation_number": "TEST-ABC123", "new_seat": "14A"}}
    )

    assert shallow.coverage.action_paths_shallow == ("update_seat",)
    assert shallow.coverage.shallow_action_parameters == (
        "update_seat.confirmation_number",
        "update_seat.new_seat",
    )
    assert representative.coverage.action_paths_representative == ("update_seat",)
    assert representative.coverage.action_paths_shallow == ()


def test_changing_fixture_values_changes_suite_identity() -> None:
    """Different test data is a different test, so identity must move."""

    spec = _spec(update_seat)
    first = _suite(spec, {"update_seat": {"new_seat": "14A"}})
    second = _suite(spec, {"update_seat": {"new_seat": "22C"}})

    assert first.fingerprint != second.fingerprint
    assert first.fingerprint == _suite(spec, {"update_seat": {"new_seat": "14A"}}).fingerprint


def test_negative_boundary_coverage_is_unchanged_by_fixtures() -> None:
    """Fixtures shape the action path only; boundary cases must not move."""

    spec = _spec(update_seat)
    without = _suite(spec)
    with_fixtures = _suite(spec, {"update_seat": {"new_seat": "14A"}})

    def boundaries(suite):
        return {
            case.scenario.fingerprint
            for case in suite.cases
            if case.lineage.origin is CaseOrigin.SCHEMA_BOUNDARY
        }

    assert boundaries(without) == boundaries(with_fixtures)
    assert boundaries(without)


def test_wall_clock_budget_still_propagates_with_fixtures() -> None:
    suite = build_frozen_suite(
        _spec(update_seat),
        AgentCheckConfig(scenario_wall_clock_seconds=180),
        seed=1729,
        representative_inputs={"update_seat": {"new_seat": "14A"}},
    )

    assert all(c.scenario.resource_budgets.wall_clock_seconds == 180 for c in suite.cases)


def test_derived_policies_still_attach_with_fixtures() -> None:
    from agentcheck.policies import derive_tool_risk_pack

    spec = _spec(update_seat)
    pack = derive_tool_risk_pack(spec)
    assert pack is not None
    suite = build_frozen_suite(
        spec,
        AgentCheckConfig(),
        seed=1729,
        policy_packs=(pack,),
        representative_inputs={"update_seat": {"new_seat": "14A"}},
    )

    positive = next(c for c in suite.cases if c.lineage.origin is CaseOrigin.POSITIVE_PATH)
    assert {t.kind.value for t in positive.scenario.trajectory_constraints}
    assert {o.kind.value for o in positive.scenario.output_criteria}


def test_fixture_file_is_covered_by_source_binding(tmp_path: Path) -> None:
    """A fixture edit must invalidate replay, like any other bound source file."""

    from agentcheck.replay.fileset import collect_source_file_set

    (tmp_path / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _write(tmp_path, _pack(new_seat="14A"))

    before = collect_source_file_set(tmp_path)
    assert any(item.path == DEFAULT_FIXTURES_FILENAME for item in before.files)

    _write(tmp_path, _pack(new_seat="22C"))
    assert collect_source_file_set(tmp_path).fingerprint != before.fingerprint


def test_generation_never_executes_a_tool_handler() -> None:
    """Every handler in this module raises; generation reads declarations only."""

    assert _positive(_spec(update_seat, set_cabin))


# --- authored scenario requests -------------------------------------------


def _request_pack(request) -> dict:
    return {
        "schema_version": "agentcheck.fixtures.v1",
        "tools": {
            "update_seat": {
                "arguments": {"confirmation_number": "TEST-ABC123", "new_seat": "14A"},
                "user_request": request,
            }
        },
    }


def test_authored_request_loads_alongside_values(tmp_path: Path) -> None:
    """Values and situation are separate answers to the same coverage gap."""

    situation = "I need to move seats; my confirmation number is TEST-ABC123."
    _write(tmp_path, _request_pack(situation))
    spec = _spec(update_seat)

    assert load_scenario_requests(tmp_path, spec) == {"update_seat": situation}
    # The values still load unchanged; the two layers do not interfere.
    assert load_representative_inputs(tmp_path, spec) == {
        "update_seat": {"confirmation_number": "TEST-ABC123", "new_seat": "14A"}
    }


def test_absent_authored_request_is_not_an_error(tmp_path: Path) -> None:
    _write(tmp_path, _pack(confirmation_number="TEST-ABC123", new_seat="14A"))

    assert load_scenario_requests(tmp_path, _spec(update_seat)) == {}


def test_blank_authored_request_fails_closed(tmp_path: Path) -> None:
    """An empty request would silently fall back and look like it applied."""

    _write(tmp_path, _request_pack("   "))

    with pytest.raises(ConfigurationError, match="at least 1 character"):
        load_scenario_requests(tmp_path, _spec(update_seat))


def test_credential_shaped_authored_request_is_refused(tmp_path: Path) -> None:
    """The request text is committed and reaches the frozen suite."""

    _write(tmp_path, _request_pack("Use key sk-live-abcdef0123456789abcdef0123."))

    with pytest.raises(ConfigurationError, match="credential"):
        load_scenario_requests(tmp_path, _spec(update_seat))


def test_authored_request_for_an_unknown_tool_fails_closed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "schema_version": "agentcheck.fixtures.v1",
            "tools": {"no_such_tool": {"user_request": "Do the thing."}},
        },
    )

    with pytest.raises(ConfigurationError, match="unknown tool"):
        load_scenario_requests(tmp_path, _spec(update_seat))
