"""One CI status, and the three ways it must refuse to say "pass".

`test` says what the agent did. `baseline check` says whether any of it is new.
A release gate needs both plus the question neither answers alone: whether the
run is certifiable at all. A suite that could not execute is not passing and is
not a regression either, and reporting it as either one is how a broken harness
becomes a green build.

These tests drive the decision logic with fakes, so nothing here executes a
suite or contacts a provider.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from agentcheck import gate as gate_module
from agentcheck.baseline.contract import DEFAULT_BASELINE_FILENAME
from agentcheck.coverage import (
    BehavioralCoverage,
    BehavioralCoverageFamily,
    BehavioralCoverageRequirement,
    BehavioralCoverageStatus,
    BehavioralDimension,
)
from agentcheck.domain import RiskAuthority, Verdict
from agentcheck.gate import (
    EXIT_BEHAVIORAL_FAILURE,
    EXIT_INCONCLUSIVE,
    EXIT_NOT_CERTIFIABLE,
    EXIT_PASS,
    GateDecision,
    find_unmet_risk_obligations,
    gate_error_result,
    run_gate,
)


_ALL_RISK_DIMENSIONS = (
    BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
    BehavioralDimension.DUPLICATE_ACTION,
    BehavioralDimension.AMBIGUOUS_OUTCOME,
    BehavioralDimension.RETRY_CONTROL,
)


def _spec(*declared: str) -> Any:
    """A real inspected spec whose named tools carry a developer declaration.

    Authority is read from the spec now, so these tests cannot fake it with a
    stub -- the declaration has to survive the adapter's own risk resolution.
    """

    from agents import Agent, function_tool

    from agentcheck.adapters import OpenAIAgentsAdapter
    from agentcheck.config import ToolRiskDeclaration

    tools = []
    for name in declared:

        def _make(tool_name: str) -> Any:
            @function_tool(
                name_override=tool_name,
                description_override="Look up a record by id.",
            )
            def _tool(record_id: str) -> str:
                raise AssertionError("never runs")

            return _tool

        tools.append(_make(name))

    return OpenAIAgentsAdapter().inspect(
        Agent(name="T", instructions="Assist.", tools=tools, model="gpt-4.1-mini"),
        declared_tool_risk={
            name: ToolRiskDeclaration(state_changing=True, destructive=True)
            for name in declared
        },
    )


def _spec_with_risk(**declarations: Any) -> Any:
    """A spec whose named tools carry exactly the given declaration, or none.

    `None` means the tool is present but undeclared, so its risk can only be
    inferred -- the case that proves authority is actually being checked rather
    than every tool being handed an obligation.
    """

    from agents import Agent, function_tool

    from agentcheck.adapters import OpenAIAgentsAdapter

    tools = []
    for name in declarations:

        def _make(tool_name: str) -> Any:
            @function_tool(
                name_override=tool_name,
                # "purge" is in the destructive verb set, so an undeclared tool
                # here is still *inferred* risky -- which is the point.
                description_override="Look up a record by id.",
            )
            def _tool(record_id: str) -> str:
                raise AssertionError("never runs")

            return _tool

        tools.append(_make(name))

    return OpenAIAgentsAdapter().inspect(
        Agent(name="T", instructions="Assist.", tools=tools, model="gpt-4.1-mini"),
        declared_tool_risk={
            name: declaration
            for name, declaration in declarations.items()
            if declaration is not None
        },
    )


def _coverage(
    *requirements: tuple[BehavioralDimension, str, BehavioralCoverageStatus, str],
) -> BehavioralCoverage:
    """A coverage report carrying exactly the requirements a test cares about."""

    by_dimension: dict[BehavioralDimension, list[BehavioralCoverageRequirement]] = {}
    for dimension, subject, status, reason in requirements:
        by_dimension.setdefault(dimension, []).append(
            BehavioralCoverageRequirement(
                subject=subject, status=status, reason_code=reason
            )
        )

    # A real report emits a family for every dimension that has a seed, and an
    # obligation always implies a seed. Fill the risk families the test did not
    # speak about with COVERED rows for the same subjects, so a fixture that
    # only cares about one dimension does not accidentally look like a report
    # with whole families missing.
    subjects = {subject for _, subject, _, _ in requirements}
    for dimension in _ALL_RISK_DIMENSIONS:
        present = {item.subject for item in by_dimension.get(dimension, ())}
        for subject in sorted(subjects - present):
            by_dimension.setdefault(dimension, []).append(
                BehavioralCoverageRequirement(
                    subject=subject,
                    status=BehavioralCoverageStatus.COVERED,
                    reason_code="covered",
                )
            )
    return BehavioralCoverage(
        spec_id="spec",
        spec_digest="sha256:spec",
        scenario_count=1,
        scenario_digest="sha256:scenarios",
        reference_scenario_count=1,
        reference_scenario_digest="sha256:scenarios",
        families=tuple(
            BehavioralCoverageFamily(
                dimension=dimension,
                requirements=tuple(items),
                **{
                    status.value: sum(1 for item in items if item.status is status)
                    for status in BehavioralCoverageStatus
                },
            )
            for dimension, items in by_dimension.items()
        ),
    )


# One declared-destructive tool with no evidence at all: the frozen AC-P0-6
# reproducer, reduced to the coverage shape the gate actually reads.
_UNMET_DECLARED_RISK = _coverage(
    (
        BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
        "tool:purge_records",
        BehavioralCoverageStatus.MISSING,
        "fabricated_success_case_missing",
    ),
    (
        BehavioralDimension.RETRY_CONTROL,
        "tool:purge_records",
        BehavioralCoverageStatus.MISSING,
        "retry_control_case_missing",
    ),
)


class _FakeExecution:
    def __init__(
        self,
        root: Path,
        counts: dict[Verdict, int],
        coverage: BehavioralCoverage | None = None,
        spec: Any = None,
    ) -> None:
        self.target_root = root
        self.run_id = "run-1"
        self.frozen_suite = type("S", (), {"fingerprint": "sha256:abc"})()
        self.report_path = root / "report.html"
        self._counts = Counter(counts)
        # A real SuiteExecution always carries both, so the fake does too; an
        # empty report means "no obligations", never "cannot tell".
        self.behavioral_coverage = coverage if coverage is not None else _coverage()
        self.spec = spec if spec is not None else _spec()

    @property
    def counts(self) -> Counter:
        return self._counts


class _FakeChecked:
    def __init__(self, exit_code: int, summary: str = "summary\n") -> None:
        self.exit_code = exit_code
        self.summary = summary


@pytest.fixture
def target(tmp_path: Path) -> Path:
    return tmp_path


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    counts: dict[Verdict, int],
    root: Path,
    checked: _FakeChecked | None = None,
    coverage: BehavioralCoverage | None = None,
    spec: Any = None,
) -> dict[str, int]:
    calls = {"check_baseline": 0}

    def _execute(target: Any, **_: Any) -> Any:
        return _FakeExecution(root, counts, coverage, spec)

    def _check(*_: Any, **__: Any) -> Any:
        calls["check_baseline"] += 1
        assert checked is not None, "baseline was compared when none was expected"
        return checked

    monkeypatch.setattr(gate_module, "execute_suite", _execute)
    monkeypatch.setattr(gate_module, "check_baseline", _check)
    return calls


def _baseline(root: Path) -> None:
    (root / DEFAULT_BASELINE_FILENAME).write_text("{}", encoding="utf-8")


# --- the certifiability question comes first -------------------------------


def test_an_infra_error_is_not_a_regression_and_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _baseline(target)
    calls = _patch(
        monkeypatch,
        counts={Verdict.PASS: 3, Verdict.INFRA_ERROR: 1},
        root=target,
    )

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_NOT_CERTIFIABLE
    assert "not certifiable" in result.reason
    assert any("not recorded as a behavioural regression" in d for d in result.detail)
    # The baseline is not even consulted: there is nothing trustworthy to compare.
    assert calls["check_baseline"] == 0


def test_an_infra_error_outranks_a_failure_in_the_same_run(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """A run that half executed cannot certify the half that did."""

    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.FAIL: 2, Verdict.INFRA_ERROR: 1},
        root=target,
    )

    result = run_gate(target)

    assert result.exit_code == EXIT_NOT_CERTIFIABLE


# --- with a trusted baseline -----------------------------------------------


def test_no_new_failure_against_the_baseline_passes(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 40, Verdict.FAIL: 5},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.PASS
    assert result.exit_code == EXIT_PASS
    assert result.baseline_compared is True


def test_inconclusive_run_with_a_baseline_is_never_reported_as_pass(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """A regression baseline cannot certify evidence the current run lacks."""

    _baseline(target)
    calls = _patch(
        monkeypatch,
        counts={Verdict.PASS: 39, Verdict.INCONCLUSIVE: 1},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_INCONCLUSIVE
    assert result.baseline_compared is True
    assert "not a pass" in result.reason
    assert any("cannot supply evidence" in item for item in result.detail)
    assert result.summary_block == "summary\n"
    assert calls["check_baseline"] == 1


def test_a_new_failure_against_the_baseline_blocks(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 40, Verdict.FAIL: 6},
        root=target,
        checked=_FakeChecked(EXIT_BEHAVIORAL_FAILURE),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_BEHAVIORAL_FAILURE
    assert "new" in result.reason


def test_an_untrustworthy_comparison_is_not_a_regression(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """Suite, source or baseline mismatch is a certification problem."""

    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 40},
        root=target,
        checked=_FakeChecked(EXIT_NOT_CERTIFIABLE),
    )

    result = run_gate(target)

    assert result.exit_code == EXIT_NOT_CERTIFIABLE
    assert "trusted" in result.reason


# --- without a baseline ----------------------------------------------------


def test_a_clean_run_without_a_baseline_passes_and_says_so(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _patch(monkeypatch, counts={Verdict.PASS: 12}, root=target)

    result = run_gate(target)

    assert result.decision == GateDecision.PASS
    assert result.exit_code == EXIT_PASS
    assert result.baseline_compared is False
    assert any("no trusted baseline" in d for d in result.detail)
    assert any("baseline create" in d for d in result.detail)


def test_a_failure_without_a_baseline_still_blocks(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _patch(monkeypatch, counts={Verdict.PASS: 10, Verdict.FAIL: 1}, root=target)

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_BEHAVIORAL_FAILURE


def test_inconclusive_is_never_silently_a_pass(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 10, Verdict.INCONCLUSIVE: 2},
        root=target,
    )

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_INCONCLUSIVE
    assert "not a pass" in result.reason


# --- machine readable ------------------------------------------------------


def test_the_decision_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 40, Verdict.FAIL: 6},
        root=target,
        checked=_FakeChecked(EXIT_BEHAVIORAL_FAILURE),
    )

    payload = json.loads(run_gate(target).to_json())

    assert payload["decision"] == "block"
    assert payload["exit_code"] == EXIT_BEHAVIORAL_FAILURE
    assert payload["counts"]["fail"] == 6
    assert payload["run_id"] == "run-1"
    assert payload["suite_fingerprint"] == "sha256:abc"
    assert payload["baseline_compared"] is True


def test_the_rendering_leads_with_the_decision(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _patch(monkeypatch, counts={Verdict.PASS: 1}, root=target)

    rendered = run_gate(target).render()

    assert rendered.startswith("PASS: ")
    assert "verdicts" in rendered
    assert "report" in rendered


def test_a_baseline_outside_the_target_is_not_followed(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """A gate must not be pointed at a baseline from somewhere else."""

    _patch(monkeypatch, counts={Verdict.PASS: 5}, root=target)

    result = run_gate(target, baseline="../elsewhere.json")

    assert result.baseline_compared is False


# --- the exit contract is the one the CLI already documents ----------------


def test_the_exit_codes_match_the_published_contract() -> None:
    assert (EXIT_PASS, EXIT_BEHAVIORAL_FAILURE, EXIT_NOT_CERTIFIABLE, EXIT_INCONCLUSIVE) == (
        0,
        1,
        2,
        3,
    )


# --- a run that never started is still an answer --------------------------


def test_a_suite_that_could_not_load_is_reported_as_not_certifiable() -> None:
    """A suite that no longer matches its target raises before anything runs.

    That is not a pass and not a regression. It is the not-certifiable case,
    and it has to survive as a decision rather than as an exception.
    """
    result = gate_error_result("frozen suite was generated for a different target")

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_NOT_CERTIFIABLE
    assert "different target" in result.reason
    assert any("no behavioural claim" in item for item in result.detail)
    assert any("not recorded as a behavioural regression" in item for item in result.detail)


def test_a_run_that_never_started_reports_no_counts_rather_than_zeroes() -> None:
    """Zeroed counts would read as "nothing failed".

    No scenario executed, so the run is in no position to make that claim.
    Absent evidence has to stay absent instead of being rendered as a clean
    result a caller could believe.
    """
    result = gate_error_result("boom")

    assert result.counts == {}
    assert json.loads(result.to_json())["counts"] == {}
    assert result.run_id == ""
    assert result.suite_fingerprint is None
    assert result.report_path == ""


def test_the_error_path_still_produces_a_parseable_document() -> None:
    """`--json` promises a document on every path.

    The not-certifiable path is the one a CI script most needs to parse, since
    it is the case where a broken harness would otherwise look green. Emitting
    nothing there makes the consumer fail on an empty document instead of
    reading the decision.
    """
    payload = json.loads(gate_error_result("suite mismatch").to_json())

    assert payload["decision"] == "block"
    assert payload["exit_code"] == EXIT_NOT_CERTIFIABLE
    assert payload["baseline_compared"] is False
    for field in ("decision", "exit_code", "reason", "detail", "counts", "report"):
        assert field in payload, f"{field} missing; consumers parse a stable shape"


def test_the_cli_emits_the_document_instead_of_an_empty_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The regression this guards: exit 2 with zero bytes on stdout."""
    from agentcheck import cli as cli_module
    from agentcheck.errors import AgentCheckError

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AgentCheckError("frozen suite does not match this target")

    monkeypatch.setattr(cli_module, "run_gate", _boom)

    exit_code = cli_module._gate_command(
        str(tmp_path),
        baseline=None,
        seed=None,
        run_id=None,
        persist_store=False,
        as_json=True,
        python_executable=None,
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_NOT_CERTIFIABLE
    payload = json.loads(captured.out)
    assert payload["exit_code"] == EXIT_NOT_CERTIFIABLE
    assert "does not match this target" in payload["reason"]
    # The human line stays on stderr so a log reads the same either way.
    assert "AgentCheck error:" in captured.err


def test_the_error_path_without_json_keeps_the_existing_human_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `--json` the top-level handler owns the message. Do not swallow it."""
    from agentcheck import cli as cli_module
    from agentcheck.errors import AgentCheckError

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AgentCheckError("frozen suite does not match this target")

    monkeypatch.setattr(cli_module, "run_gate", _boom)

    with pytest.raises(AgentCheckError):
        cli_module._gate_command(
            str(tmp_path),
            baseline=None,
            seed=None,
            run_id=None,
            persist_store=False,
            as_json=False,
            python_executable=None,
        )


# --- the authoritative-risk evidence floor ---------------------------------
#
# A declared risk creates a required behavioural evidence obligation. The gate
# used to return PASS / exit 0 for a target whose declared-destructive tool had
# zero cases, because it decided only from executed-case verdicts and never
# consulted coverage. Missing required evidence can never be upgraded to PASS.


def test_missing_declared_risk_evidence_is_never_a_pass(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """The frozen AC-P0-6 reproducer, as a decision-level regression: every
    case passes and the baseline is clean, yet the gate must refuse."""

    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=_UNMET_DECLARED_RISK,
        spec=_spec("purge_records"),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_INCONCLUSIVE
    assert result.exit_code != EXIT_PASS
    assert len(result.unmet_risk_obligations) == 2


def test_missing_declared_risk_evidence_blocks_without_a_baseline(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """The floor is a property of the evidence, not of having a baseline."""

    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        coverage=_UNMET_DECLARED_RISK,
        spec=_spec("purge_records"),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_INCONCLUSIVE


def test_the_block_names_the_tool_the_requirement_and_the_next_step(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """Changing the exit code alone would tell a developer their build stopped
    without telling them what is missing or how to fix it."""

    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=_UNMET_DECLARED_RISK,
        spec=_spec("purge_records"),
    )

    rendered = run_gate(target).render()

    assert "purge_records" in rendered
    assert "fabricated_success_after_failure" in rendered
    assert "retry_control" in rendered
    # why it blocked, and that the obligation came from a declaration
    assert "declared, not inferred" in rendered
    # what to do next
    assert "agentcheck.json" in rendered


def test_the_unmet_obligations_are_machine_readable(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=_UNMET_DECLARED_RISK,
        spec=_spec("purge_records"),
    )

    document = json.loads(run_gate(target).to_json())

    assert document["decision"] == "block"
    assert document["exit_code"] == EXIT_INCONCLUSIVE
    reported = document["unmet_risk_obligations"]
    assert {item["tool"] for item in reported} == {"purge_records"}
    assert {item["dimension"] for item in reported} == {
        "fabricated_success_after_failure",
        "retry_control",
    }
    assert all(item["reason_code"] for item in reported)


def test_an_infra_error_still_outranks_the_evidence_floor(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """INFRA_ERROR and evidence insufficiency stay distinct concepts. A run
    that could not execute says nothing, including nothing about coverage."""

    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 3, Verdict.INFRA_ERROR: 1},
        root=target,
        coverage=_UNMET_DECLARED_RISK,
        spec=_spec("purge_records"),
    )

    result = run_gate(target)

    # Precedence is what matters: a run that could not execute says nothing,
    # and its own answer wins. The obligations are still recorded rather than
    # dropped, so `--json` stays useful for triage.
    assert result.exit_code == EXIT_NOT_CERTIFIABLE
    assert result.decision == GateDecision.BLOCK
    assert result.unmet_risk_obligations != ()


def test_a_behavioural_failure_still_outranks_the_evidence_floor(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """A real regression keeps its own, more specific answer."""

    _patch(
        monkeypatch,
        counts={Verdict.PASS: 2, Verdict.FAIL: 1},
        root=target,
        coverage=_UNMET_DECLARED_RISK,
        spec=_spec("purge_records"),
    )

    result = run_gate(target)

    assert result.exit_code == EXIT_BEHAVIORAL_FAILURE


def test_unknown_applicability_never_becomes_an_obligation(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """Non-authoritative risk stays UNKNOWN and must not manufacture a failure.
    Inference is not authority, even when it is right."""

    inferred_only = _coverage(
        (
            BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
            "tool:guessed",
            BehavioralCoverageStatus.UNKNOWN,
            "risk_metadata_not_authoritative",
        ),
        (
            BehavioralDimension.RETRY_CONTROL,
            "tool:guessed",
            BehavioralCoverageStatus.UNKNOWN,
            "risk_metadata_not_authoritative",
        ),
    )
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=inferred_only,
        spec=_spec(),  # nothing declared: no obligation can exist
    )

    result = run_gate(target)

    assert result.decision == GateDecision.PASS
    assert result.exit_code == EXIT_PASS


def test_an_uncovered_tool_is_not_by_itself_a_gate_failure(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """The accepted scope is narrow. success_path, failure_handling and
    timeout_handling are seeded for every declared tool regardless of risk, so
    a gap there is not a risk obligation and must not block."""

    non_risk_gaps = _coverage(
        (
            BehavioralDimension.SUCCESS_PATH,
            "tool:ordinary",
            BehavioralCoverageStatus.MISSING,
            "success_path_missing",
        ),
        (
            BehavioralDimension.FAILURE_HANDLING,
            "tool:ordinary",
            BehavioralCoverageStatus.MISSING,
            "failure_path_missing",
        ),
        (
            BehavioralDimension.TIMEOUT_HANDLING,
            "tool:ordinary",
            BehavioralCoverageStatus.MISSING,
            "timeout_path_missing",
        ),
    )
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=non_risk_gaps,
        spec=_spec("ordinary"),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.PASS
    assert find_unmet_risk_obligations(_spec("ordinary"), non_risk_gaps) == ()


def test_partial_risk_evidence_does_not_block(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """PARTIAL is the ordinary state of a healthy generated suite. Blocking on
    it would be the strictest rejected option, not the accepted one."""

    partial = _coverage(
        (
            BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
            "tool:declared",
            BehavioralCoverageStatus.PARTIAL,
            "fabrication_terms_unconfigured",
        ),
        (
            BehavioralDimension.RETRY_CONTROL,
            "tool:declared",
            BehavioralCoverageStatus.COVERED,
            "covered",
        ),
    )
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=partial,
        spec=_spec("declared"),
    )

    assert run_gate(target).decision == GateDecision.PASS
    assert find_unmet_risk_obligations(_spec("declared"), partial) == ()


def test_a_clean_target_with_no_risk_declarations_is_unaffected(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """Backward compatibility: nothing changes for a target that declares no
    authoritative risk."""

    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.PASS
    assert result.exit_code == EXIT_PASS
    assert json.loads(result.to_json())["unmet_risk_obligations"] == []


# --- the two ways this floor was wrong before independent review ------------


def test_inferred_risk_never_creates_an_obligation_even_when_coverage_says_missing(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """Authority comes from the spec, never from a coverage status.

    `_seed_from_scenarios` can raise a risk dimension to APPLICABLE from
    scenario constraints alone, so a tool whose risk is only *inferred* can
    reach MISSING. Reading MISSING as "declared" would make inference
    authoritative and would tell the developer to fix a `tool_risk` entry they
    never wrote.
    """

    missing_but_inferred = _coverage(
        (
            BehavioralDimension.AMBIGUOUS_OUTCOME,
            "tool:act",
            BehavioralCoverageStatus.MISSING,
            "ambiguous_outcome_case_missing",
        ),
    )
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=missing_but_inferred,
        spec=_spec(),  # `act` exists in coverage but is declared by nobody
    )

    result = run_gate(target)

    assert result.decision == GateDecision.PASS
    assert result.unmet_risk_obligations == ()
    assert find_unmet_risk_obligations(_spec(), missing_but_inferred) == ()


def _truncated_family(
    dimension: BehavioralDimension, *, missing: int, visible: tuple[str, ...]
) -> BehavioralCoverageFamily:
    """A family whose detail rows are all COVERED while its counts say some
    requirement is MISSING -- exactly what `_family()` emits once the number of
    subjects exceeds MAX_COVERAGE_DETAILS."""

    rows = tuple(
        BehavioralCoverageRequirement(
            subject=subject,
            status=BehavioralCoverageStatus.COVERED,
            reason_code="covered",
        )
        for subject in visible
    )
    return BehavioralCoverageFamily(
        dimension=dimension,
        covered=len(visible),
        missing=missing,
        requirements=rows,
        omitted=missing,
    )


def _families(*families: BehavioralCoverageFamily) -> BehavioralCoverage:
    return BehavioralCoverage(
        spec_id="spec",
        spec_digest="sha256:spec",
        scenario_count=1,
        scenario_digest="sha256:scenarios",
        reference_scenario_count=1,
        reference_scenario_digest="sha256:scenarios",
        families=families,
    )


def test_an_unreadable_obligation_is_not_assumed_met_when_something_is_missing(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """`family.requirements` is a display view, capped and stripped of
    colliding subjects, while its counters are complete. When the counters say
    a requirement is MISSING and no visible row accounts for it, a declared
    tool whose row is absent cannot be assumed satisfied."""

    coverage = _families(
        *(
            _truncated_family(dimension, missing=1, visible=("tool:someone_else",))
            for dimension in _ALL_RISK_DIMENSIONS
        )
    )
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=coverage,
        spec=_spec("purge_records"),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_INCONCLUSIVE
    assert any(
        item.reason_code == gate_module.TRUNCATED_DETAIL_REASON
        for item in result.unmet_risk_obligations
    )


def test_a_truncated_but_complete_report_does_not_block(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """The inverse, and the reason the rule above is guarded by the counters.

    A target with more declared tools than the detail cap has rows it cannot
    show. When every counter says `missing == 0`, the counts themselves prove
    nothing is missing, so blocking would be provably false -- and unfixable,
    because no amount of added evidence can shrink the family below the cap.
    """

    coverage = _families(
        *(
            _truncated_family(dimension, missing=0, visible=("tool:someone_else",))
            for dimension in _ALL_RISK_DIMENSIONS
        )
    )
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=coverage,
        spec=_spec("purge_records"),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.PASS
    assert result.unmet_risk_obligations == ()


def test_an_unbindable_declared_tool_is_not_a_pass(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """A declared-risky tool whose coverage binding is lossy -- a schema too
    large or deep to project stably -- has every seed forced to UNKNOWN. The
    declaration is still authoritative, so an unreadable status is unknown, not
    satisfied. Passing here would leave the original false green intact for any
    tool with a big schema."""

    unbindable = _coverage(
        *(
            (
                dimension,
                "tool:purge_records",
                BehavioralCoverageStatus.UNKNOWN,
                "coverage_binding_redacted_or_truncated",
            )
            for dimension in _ALL_RISK_DIMENSIONS
        )
    )
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=unbindable,
        spec=_spec("purge_records"),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_INCONCLUSIVE
    assert len(result.unmet_risk_obligations) == 4


@pytest.mark.parametrize(
    ("dimension", "reason"),
    [
        (
            BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
            "fabricated_success_case_missing",
        ),
        (BehavioralDimension.DUPLICATE_ACTION, "duplicate_action_case_missing"),
        (BehavioralDimension.AMBIGUOUS_OUTCOME, "ambiguous_outcome_case_missing"),
        (BehavioralDimension.RETRY_CONTROL, "retry_control_case_missing"),
    ],
)
def test_each_declared_risk_dimension_is_individually_load_bearing(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    dimension: BehavioralDimension,
    reason: str,
) -> None:
    """All four dimensions the decision names must block on their own, so none
    can be dropped from the set without a test failing."""

    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=_coverage(
            (dimension, "tool:purge_records", BehavioralCoverageStatus.MISSING, reason)
        ),
        spec=_spec("purge_records"),
    )

    result = run_gate(target)

    assert result.exit_code == EXIT_INCONCLUSIVE
    assert [item.dimension for item in result.unmet_risk_obligations] == [
        dimension.value
    ]


def test_unsupported_risk_evidence_does_not_block(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """UNSUPPORTED is a statement about what AgentCheck can express, not
    missing evidence, and must not be turned into an obligation."""

    unsupported = _coverage(
        (
            BehavioralDimension.RETRY_CONTROL,
            "tool:purge_records",
            BehavioralCoverageStatus.UNSUPPORTED,
            "trajectory_ordering_not_supported",
        ),
        (
            BehavioralDimension.AMBIGUOUS_OUTCOME,
            "tool:purge_records",
            BehavioralCoverageStatus.COVERED,
            "covered",
        ),
        (
            BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
            "tool:purge_records",
            BehavioralCoverageStatus.COVERED,
            "covered",
        ),
        (
            BehavioralDimension.DUPLICATE_ACTION,
            "tool:purge_records",
            BehavioralCoverageStatus.COVERED,
            "covered",
        ),
    )
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        coverage=unsupported,
        spec=_spec("purge_records"),
    )

    assert run_gate(target).decision == GateDecision.PASS


def test_an_undeclared_tool_present_in_the_spec_gets_no_obligation() -> None:
    """The authority check must be observable, not vacuous.

    A previous version of this file proved "inference creates no obligation"
    against a spec containing no tools at all, so the assertion held for want of
    anything to over-report. Here `purge_records` is genuinely present and
    genuinely inferred-risky, so removing the authority check makes this fail.
    """

    from agentcheck.coverage import risk_obligations_for_spec

    spec = _spec_with_risk(purge_records=None)
    tool = next(item for item in spec.tools.items if item.value.name == "purge_records")
    assertion = next(
        item for item in spec.tool_risk.items if item.tool_name == "purge_records"
    )
    # Inferred risk, and the heuristic really did fire, so the only thing
    # keeping this tool out of the obligation set is the authority check.
    assert assertion.destructive.authority is not RiskAuthority.DEVELOPER_DECLARED
    assert tool.value.destructive is True

    assert risk_obligations_for_spec(spec) == {}


def test_declaring_only_state_changing_obligates_only_the_action_dimensions() -> None:
    """The action-gated and destructive-gated groups must not be swapped.

    Every other test declares both axes, which makes the mapping unobservable.
    A one-axis declaration is where it matters: swapping the two sets would
    demand ambiguous_outcome and retry_control rows that `_seed_from_spec` never
    seeds for this tool, turning into an immediate unreadable-status block.
    """

    from agentcheck.config import ToolRiskDeclaration
    from agentcheck.coverage import risk_obligations_for_spec

    spec = _spec_with_risk(
        purge_records=ToolRiskDeclaration(state_changing=True, destructive=False)
    )

    assert risk_obligations_for_spec(spec) == {
        "purge_records": frozenset(
            {
                BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
                BehavioralDimension.DUPLICATE_ACTION,
            }
        )
    }
