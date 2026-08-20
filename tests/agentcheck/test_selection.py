from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import agentcheck.application as application
from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.cli import main
from agentcheck.config import AgentCheckConfig, load_config
from agentcheck.domain import Verdict
from agentcheck.errors import ConfigurationError
from agentcheck.generate import (
    MAX_CASES,
    CaseOrigin,
    build_frozen_suite,
    encode_frozen_suite,
)
from agentcheck.generate.selection import (
    SELECTION_ALGORITHM,
    CoverageUnit,
    scenario_is_mandatory,
    select_units,
)
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.inspect import load_target
from agentcheck.policies import CONFIRM_BEFORE_DESTRUCTIVE_V1, apply_policy_packs


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
SEED = 1729
EXPECTED_FAILURES = {
    "delete_without_confirmation",
    "ambiguous_account_selection",
    "destructive_ambiguous_timeout",
    "fabricated_email_success",
    "duplicate_email_update",
}


def _unit(
    scenario_id: str,
    dimensions: tuple[str, ...],
    *,
    mandatory: bool = False,
    fingerprint: str | None = None,
) -> CoverageUnit:
    return CoverageUnit(
        scenario_id=scenario_id,
        fingerprint=fingerprint or f"fp:{scenario_id}",
        dimensions=dimensions,
        mandatory=mandatory,
    )


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "account_agent"
    shutil.copytree(
        EXAMPLE,
        target,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    return target


def _example_spec() -> Any:
    target, source = load_target(EXAMPLE)
    return OpenAIAgentsAdapter().inspect(target, source=source)


def test_greedy_selection_is_deterministic_at_a_fixed_pool() -> None:
    units = (
        _unit("alpha", ("tool:lookup", "schema:wrong_type")),
        _unit("bravo", ("tool:delete", "mutation:withhold_confirmation")),
        _unit("charlie", ("tool:lookup", "policy:confirm_before_destructive_v1")),
        _unit("delta", ("tool:lookup", "schema:wrong_type")),
    )
    first = select_units(units, max_cases=2)
    second = select_units(units, max_cases=2)
    assert first.plan == second.plan
    assert first.selected_ids == second.selected_ids
    assert first.plan.algorithm == SELECTION_ALGORITHM
    assert set(first.selected_ids) <= {unit.scenario_id for unit in units}


def test_budget_at_or_above_pool_selects_everything() -> None:
    units = (
        _unit("one", ("tool:a",)),
        _unit("two", ("tool:a",)),
        _unit("three", ("tool:b",)),
    )
    result = select_units(units, max_cases=3)
    assert result.selected_ids == ("one", "two", "three")
    assert result.plan.excluded_ids == ()
    assert all(item.reason == "pool_within_budget" for item in result.plan.decisions)
    oversized = select_units(units, max_cases=MAX_CASES)
    assert oversized.selected_ids == result.selected_ids


def test_already_small_suite_is_unchanged() -> None:
    units = (_unit("only", ("tool:lookup", "schema:missing_required_property")),)
    result = select_units(units, max_cases=8)
    assert result.selected_ids == ("only",)
    assert result.plan.excluded_ids == ()


def test_tie_break_prefers_lexicographic_scenario_id() -> None:
    units = (
        _unit("zeta", ("tool:shared", "extra:z")),
        _unit("alpha", ("tool:shared", "extra:a")),
        _unit("mu", ("other:dim",)),
    )
    result = select_units(units, max_cases=1)
    assert result.selected_ids == ("alpha",)
    reasons = {item.scenario_id: item.reason for item in result.plan.decisions}
    assert reasons["alpha"] == "covers_new_dimensions"


def test_redundant_cases_are_dropped_when_budget_is_a_cap() -> None:
    units = (
        _unit("cover-a", ("tool:a", "schema:wrong_type")),
        _unit("cover-b", ("tool:b", "mutation:duplicate_request")),
        _unit("redundant-a", ("tool:a",)),
        _unit("unique-late", ("tool:c",)),
    )
    result = select_units(units, max_cases=2)
    assert "unique-late" not in result.selected_ids
    reasons = {item.scenario_id: item.reason for item in result.plan.decisions}
    assert reasons["redundant-a"] == "redundant_coverage"
    assert reasons["unique-late"] == "budget_exhausted"
    assert "tool:c" in result.plan.coverage.uncovered


def test_covering_set_without_numeric_budget_drops_only_redundant_cases() -> None:
    units = (
        _unit("keep-a", ("tool:a",)),
        _unit("keep-b", ("tool:b",)),
        _unit("redundant", ("tool:a",)),
    )
    result = select_units(units, max_cases=None)
    assert result.selected_ids == ("keep-a", "keep-b")
    assert result.plan.excluded_ids == ("redundant",)
    assert result.plan.max_cases is None


def test_mandatory_policy_cases_are_never_dropped() -> None:
    units = (
        _unit("policy-case", ("policy:confirm_before_destructive_v1",), mandatory=True),
        _unit("rich", ("tool:a", "tool:b", "schema:wrong_type", "mutation:reorder_turns")),
        _unit("other", ("tool:c",)),
    )
    result = select_units(units, max_cases=2)
    assert "policy-case" in result.selected_ids
    reasons = {item.scenario_id: item.reason for item in result.plan.decisions}
    assert reasons["policy-case"] == "mandatory_policy"


def test_mandatory_cases_exceeding_budget_fail_closed() -> None:
    units = (
        _unit("one", ("policy:a",), mandatory=True),
        _unit("two", ("policy:b",), mandatory=True),
    )
    with pytest.raises(ConfigurationError, match="mandatory policy cases exceed"):
        select_units(units, max_cases=1)


def test_no_candidate_behavior_fails_closed() -> None:
    with pytest.raises(ConfigurationError, match="no candidates remain"):
        select_units((), max_cases=4)
    with pytest.raises(ConfigurationError, match="between 1 and"):
        select_units((_unit("one", ("tool:a",)),), max_cases=0)
    with pytest.raises(ConfigurationError, match="between 1 and"):
        select_units((_unit("one", ("tool:a",)),), max_cases=MAX_CASES + 1)


def test_unknown_and_unsupported_are_distinct_from_uncovered() -> None:
    units = (_unit("keep", ("tool:lookup", "schema:wrong_type")),)
    result = select_units(
        units,
        max_cases=1,
        extra_goals=("tool:lookup", "tool:mystery"),
        unsupported=("unsupported:pattern is not inverted",),
        unknown=("unknown:capability:unnamed",),
    )
    coverage = result.plan.coverage
    assert "tool:lookup" in coverage.covered
    assert "tool:mystery" in coverage.uncovered
    assert "unsupported:pattern is not inverted" in coverage.unsupported
    assert "unknown:capability:unnamed" in coverage.unknown
    assert "unsupported:pattern is not inverted" not in coverage.uncovered
    assert "unknown:capability:unnamed" not in coverage.uncovered
    assert "unknown:capability:unnamed" not in coverage.covered


def test_declared_policy_pack_marks_tool_specific_cases_mandatory() -> None:
    parent = next(
        scenario
        for scenario in build_account_support_suite(seed=SEED)
        if scenario.scenario_id == "delete_without_confirmation"
    )
    applied = apply_policy_packs(
        parent, (CONFIRM_BEFORE_DESTRUCTIVE_V1,), declared=True
    )
    assert scenario_is_mandatory(applied) is True
    assert scenario_is_mandatory(parent) is False


def test_default_frozen_suite_identity_omits_selection() -> None:
    spec = _example_spec()
    config = AgentCheckConfig()
    first = build_frozen_suite(spec, config, seed=SEED)
    second = build_frozen_suite(spec, config, seed=SEED)
    payload = json.loads(encode_frozen_suite(first))
    assert first == second
    assert first.selection is None
    assert "selection" not in payload
    bounded = build_frozen_suite(spec, config, seed=SEED, max_cases=MAX_CASES)
    assert bounded.selection is not None
    assert {case.scenario.scenario_id for case in bounded.cases} == {
        case.scenario.scenario_id for case in first.cases
    }
    assert bounded.fingerprint != first.fingerprint
    assert bounded.suite_id != first.suite_id


def test_generate_max_cases_records_lineage_and_same_seed_identity() -> None:
    spec = _example_spec()
    config = AgentCheckConfig()
    left = build_frozen_suite(spec, config, seed=SEED, max_cases=8)
    right = build_frozen_suite(spec, config, seed=SEED, max_cases=8)
    assert left == right
    assert left.selection is not None
    assert len(left.cases) <= 8
    assert len(left.selection.selected_ids) == len(left.cases)
    assert left.selection.excluded_ids
    origins = {case.lineage.origin for case in left.cases}
    assert CaseOrigin.BUILT_IN in origins or CaseOrigin.SCHEMA_BOUNDARY in origins
    coverage = left.selection.coverage
    assert coverage.covered
    assert any(tag.startswith("tool:") for tag in coverage.covered)
    assert any(
        tag.startswith("schema:") or tag.startswith("source:schema_boundary")
        for tag in (*coverage.covered, *coverage.uncovered)
    )


def test_config_max_cases_is_optional_and_keeps_v1(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("agent = object()\n", encoding="utf-8")
    (tmp_path / "agentcheck.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.config.v1",
                "entrypoint": "agent.py:agent",
            }
        ),
        encoding="utf-8",
    )
    _, config = load_config(tmp_path)
    assert config.max_cases is None
    assert AgentCheckConfig().max_cases is None
    loaded = AgentCheckConfig(max_cases=12)
    assert loaded.max_cases == 12
    with pytest.raises(ValidationError):
        AgentCheckConfig(max_cases=0)


def test_cli_generate_max_cases_and_test_select_coverage(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    assert main(["generate", str(target), "--max-cases", "8", "--force"]) == 0
    suite_path = target / "agentcheck-suite.json"
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    assert payload["selection"]["algorithm"] == SELECTION_ALGORITHM
    assert len(payload["cases"]) <= 8
    assert payload["selection"]["excluded_ids"]

    full = _copy_example(tmp_path / "full")
    generation = application.generate_suite(full, seed=SEED, force=True)
    execution = application.execute_suite(
        full, run_id="coverage-select", persist_store=False, select="coverage"
    )
    summary = json.loads(
        (execution.artifact_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["invalid_scenarios"] == 0
    assert summary["excluded_by_selection"] > 0
    assert "selection" in summary
    assert "coverage" in summary
    excluded = set(summary["selection"]["excluded_ids"])
    evaluated = {item.scenario_id for item in execution.evaluations}
    assert excluded.isdisjoint(evaluated)
    assert execution.selection is not None
    report = (execution.artifact_directory / "report.html").read_text(encoding="utf-8")
    assert "Excluded by selection" in report
    assert "Uncovered dimensions" in report
    assert "<script>" not in report
    assert "default-src 'none'" in report
    assert len(execution.evaluations) < len(generation.suite.cases)


def test_selected_out_case_is_never_reported_as_passing(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    application.generate_suite(target, seed=SEED, force=True)
    execution = application.execute_suite(
        target, run_id="selection-honesty", persist_store=False, select="coverage"
    )
    summary = json.loads(
        (execution.artifact_directory / "summary.json").read_text(encoding="utf-8")
    )
    excluded = set(summary["selection"]["excluded_ids"])
    assert excluded
    passing = {
        item.scenario_id
        for item in execution.evaluations
        if item.verdict is Verdict.PASS
    }
    failed = {
        item.scenario_id
        for item in execution.evaluations
        if item.verdict is Verdict.FAIL
    }
    assert excluded.isdisjoint(passing)
    assert excluded.isdisjoint(failed)
    lint_excluded = summary["invalid_scenarios"]
    assert lint_excluded == 0
    assert summary["excluded_by_selection"] == len(excluded)


def test_select_coverage_is_not_applied_implicitly(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    generation = application.generate_suite(target, seed=SEED, force=True)
    execution = application.execute_suite(
        target, run_id="full-suite", persist_store=False
    )
    assert len(execution.evaluations) == len(generation.suite.cases)
    summary = json.loads(
        (execution.artifact_directory / "summary.json").read_text(encoding="utf-8")
    )
    assert "excluded_by_selection" not in summary
    assert "selection" not in summary


def test_end_to_end_frozen_selection_then_test(tmp_path: Path) -> None:
    target = _copy_example(tmp_path)
    generation = application.generate_suite(target, seed=SEED, max_cases=10, force=True)
    execution = application.execute_suite(
        target, run_id="bounded-frozen", persist_store=True
    )
    assert generation.suite.selection is not None
    assert len(execution.evaluations) == len(generation.suite.cases) <= 10
    evaluated = {item.scenario_id for item in execution.evaluations}
    assert evaluated == set(generation.suite.selection.selected_ids)
    assert evaluated.isdisjoint(generation.suite.selection.excluded_ids)
    html = execution.report_path.read_text(encoding="utf-8")
    assert "greedy_set_cover.v1" in html
    remaining_failures = EXPECTED_FAILURES.intersection(evaluated)
    observed_failures = {
        item.scenario_id
        for item in execution.evaluations
        if item.verdict is Verdict.FAIL
    }
    assert observed_failures == remaining_failures
