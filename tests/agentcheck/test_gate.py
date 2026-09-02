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
from agentcheck.coverage import BehavioralDimension, unmet_risk_obligations
from agentcheck.domain import Verdict
from agentcheck.gate import (
    EXIT_BEHAVIORAL_FAILURE,
    EXIT_INCONCLUSIVE,
    EXIT_NOT_CERTIFIABLE,
    EXIT_PASS,
    GateDecision,
    gate_error_result,
    run_gate,
)


SEED = 1729

_RISK_DIMENSIONS = (
    BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
    BehavioralDimension.DUPLICATE_ACTION,
    BehavioralDimension.AMBIGUOUS_OUTCOME,
    BehavioralDimension.RETRY_CONTROL,
)


def _agent(*names: str) -> Any:
    from agents import Agent, function_tool

    tools = []
    for name in names:

        def _make(tool_name: str) -> Any:
            # "purge" is in the destructive verb set, so an *undeclared* tool
            # here is still inferred risky -- which is what makes the negative
            # tests below non-vacuous.
            @function_tool(
                name_override=tool_name,
                description_override="Look up a record by id.",
            )
            def _tool(record_id: str) -> str:
                raise AssertionError("never runs")

            return _tool

        tools.append(_make(name))
    return Agent(name="T", instructions="Assist.", tools=tools, model="gpt-4.1-mini")


def _spec(*declared: str) -> Any:
    """An inspected spec whose named tools carry a developer declaration."""

    from agentcheck.adapters import OpenAIAgentsAdapter
    from agentcheck.config import ToolRiskDeclaration

    return OpenAIAgentsAdapter().inspect(
        _agent(*declared),
        declared_tool_risk={
            name: ToolRiskDeclaration(state_changing=True, destructive=True)
            for name in declared
        },
    )


def _spec_with_risk(**declarations: Any) -> Any:
    """A spec whose named tools carry exactly the given declaration, or none.

    `None` means the tool is present but undeclared, so its risk can only be
    inferred -- the case that proves authority is actually checked rather than
    every tool being handed an obligation.
    """

    from agentcheck.adapters import OpenAIAgentsAdapter

    return OpenAIAgentsAdapter().inspect(
        _agent(*declarations),
        declared_tool_risk={
            name: declaration
            for name, declaration in declarations.items()
            if declaration is not None
        },
    )


class _FakeExecution:
    def __init__(
        self,
        root: Path,
        counts: dict[Verdict, int],
        spec: Any = None,
        scenarios: tuple[Any, ...] = (),
    ) -> None:
        self.target_root = root
        self.run_id = "run-1"
        self.frozen_suite = type("S", (), {"fingerprint": "sha256:abc"})()
        self.report_path = root / "report.html"
        self._counts = Counter(counts)
        # The gate derives obligations from these two, exactly as it does from
        # a real SuiteExecution -- no coverage report is fabricated anywhere.
        self.spec = spec if spec is not None else _spec()
        self.scenarios = scenarios

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
    spec: Any = None,
    scenarios: tuple[Any, ...] = (),
) -> dict[str, int]:
    calls = {"check_baseline": 0}

    def _execute(target: Any, **_: Any) -> Any:
        return _FakeExecution(root, counts, spec, scenarios)

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
#
# Obligations are evaluated from the spec and the scenarios directly, never
# reconstructed from a rendered coverage report: that report is capped, drops
# redaction-collided subjects, and redacts subject names, and three separate
# attempts to reconstruct per-tool truth from it each produced a false PASS or
# a false BLOCK. The tests below therefore drive real specs and real scenarios.


def _generated(spec: Any) -> tuple[Any, ...]:
    """The suite AgentCheck really generates for this spec."""

    from agentcheck.config import AgentCheckConfig
    from agentcheck.generate import build_frozen_suite

    return tuple(build_frozen_suite(spec, AgentCheckConfig(), seed=SEED).scenarios)


def test_a_declared_tool_with_no_evidence_is_never_a_pass(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """The frozen AC-P0-6 reproducer as a decision-level regression: every case
    passes and the baseline is clean, yet the gate must refuse."""

    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        spec=_spec("purge_records"),
        scenarios=(),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_INCONCLUSIVE
    assert {item.dimension for item in result.unmet_risk_obligations} == set(
        _RISK_DIMENSIONS
    )


def test_a_declared_tool_with_its_generated_suite_passes(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """Backward compatibility: the ordinary case, where AgentCheck's own
    generated suite supplies the evidence, is unaffected."""

    spec = _spec("purge_records")
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        spec=spec,
        scenarios=_generated(spec),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.PASS
    assert result.unmet_risk_obligations == ()


def test_an_undeclared_tool_with_no_evidence_still_passes(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """Inference must never create an obligation. `purge_records` is inferred
    risky here, and has no evidence at all, and still must not block."""

    spec = _spec_with_risk(purge_records=None)
    assert spec.tools.items[0].value.destructive is True  # inference did fire
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        spec=spec,
        scenarios=(),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.PASS
    assert unmet_risk_obligations(spec, ()) == ()


def test_a_redaction_shaped_tool_name_still_blocks(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """A declared tool whose *name* looks like a secret must still block.

    Coverage subjects are passed through `redact_log_text`, so a name matching
    a product-token or provider-key pattern renders as `tool:[REDACTED]` and no
    longer matches the spec's name. A gate that looked the tool up in the
    rendered report silently dropped the obligation and returned PASS -- the
    original false green, reachable with a single tool and no truncation.
    """

    spec = _spec("al_purge_records")
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        spec=spec,
        scenarios=(),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert result.exit_code == EXIT_INCONCLUSIVE
    assert {item.tool_name for item in result.unmet_risk_obligations} == {
        "al_purge_records"
    }


def test_many_declared_tools_do_not_overwhelm_the_bounded_report(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """More declared tools than the coverage report's detail cap.

    An earlier implementation reconstructed status from that capped view and
    either missed tools past the cap (false PASS) or charged every unreadable
    row as unmet (a false BLOCK no evidence could clear). Evaluating the
    obligations directly makes the cap irrelevant: all 30 tools are accounted
    for, individually.
    """

    names = tuple(f"purge_record_{index:02d}" for index in range(30))
    spec = _spec(*names)
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        spec=spec,
        scenarios=(),
    )

    result = run_gate(target)

    assert result.decision == GateDecision.BLOCK
    assert {item.tool_name for item in result.unmet_risk_obligations} == set(names)


def test_an_infra_error_still_outranks_the_evidence_floor(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    """INFRA_ERROR and evidence insufficiency stay distinct. A run that could
    not execute says nothing, including nothing about coverage."""

    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 3, Verdict.INFRA_ERROR: 1},
        root=target,
        spec=_spec("purge_records"),
        scenarios=(),
    )

    result = run_gate(target)

    assert result.exit_code == EXIT_NOT_CERTIFIABLE
    assert result.decision == GateDecision.BLOCK
    # Recorded, not dropped, so --json stays useful for triage.
    assert result.unmet_risk_obligations != ()


def test_a_behavioural_failure_still_outranks_the_evidence_floor(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 2, Verdict.FAIL: 1},
        root=target,
        spec=_spec("purge_records"),
        scenarios=(),
    )

    assert run_gate(target).exit_code == EXIT_BEHAVIORAL_FAILURE


def test_the_floor_applies_without_a_baseline_too(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        spec=_spec("purge_records"),
        scenarios=(),
    )

    assert run_gate(target).exit_code == EXIT_INCONCLUSIVE


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
        spec=_spec("purge_records"),
        scenarios=(),
    )

    rendered = run_gate(target).render()

    assert "purge_records" in rendered
    assert "retry_control" in rendered
    assert "declared, not inferred" in rendered
    # both places a declaration can live, so the advice fits either adapter
    assert "agentcheck.json" in rendered
    assert "ToolDefinition" in rendered


def test_the_unmet_obligations_are_machine_readable(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> None:
    _baseline(target)
    _patch(
        monkeypatch,
        counts={Verdict.PASS: 12},
        root=target,
        checked=_FakeChecked(EXIT_PASS),
        spec=_spec("purge_records"),
        scenarios=(),
    )

    document = json.loads(run_gate(target).to_json())

    assert document["exit_code"] == EXIT_INCONCLUSIVE
    reported = document["unmet_risk_obligations"]
    assert {item["tool"] for item in reported} == {"purge_records"}
    assert {item["dimension"] for item in reported} == {
        dimension.value for dimension in _RISK_DIMENSIONS
    }
    assert all(item["reason_code"] for item in reported)


def test_declaring_only_state_changing_obligates_only_the_action_dimensions() -> None:
    """The action-gated and destructive-gated groups must not be swapped.

    Every other test declares both axes, which makes the mapping unobservable.
    A one-axis declaration is where it matters.
    """

    from agentcheck.config import ToolRiskDeclaration

    spec = _spec_with_risk(
        purge_records=ToolRiskDeclaration(state_changing=True, destructive=False)
    )

    assert {item.dimension for item in unmet_risk_obligations(spec, ())} == {
        BehavioralDimension.FABRICATED_SUCCESS_AFTER_FAILURE,
        BehavioralDimension.DUPLICATE_ACTION,
    }


def test_a_tool_whose_coverage_binding_is_lossy_still_blocks() -> None:
    """A declared tool with a schema too large to project stably.

    In a rendered coverage report every seed for such a tool is forced to
    UNKNOWN, because its subject cannot be bound reliably for display. A gate
    reading that report saw UNKNOWN and passed, leaving the original false
    green intact for any tool with a big schema. Evaluating the obligation
    directly sidesteps the display binding entirely: the tool is identified by
    name, so its real status is computed and the absence of evidence blocks.
    """

    from agentcheck.adapters import CustomAgentAdapter
    from agentcheck.domain import ToolDefinition

    schema = {
        "type": "object",
        "properties": {f"p{index}": {"type": "string"} for index in range(101)},
        "additionalProperties": False,
    }
    tool = ToolDefinition(
        name="purge_big",
        description="Purge records.",
        input_schema=schema,
        state_changing=True,
        destructive=True,
        replaceable=True,
    )

    class _Agent:
        name = "T"
        instructions = "Assist."
        tools = (tool,)

        def start(self, message, tools):  # pragma: no cover - never reached
            raise AssertionError("no turn runs")

        def resume(self, state, message, tools):  # pragma: no cover
            raise AssertionError("no turn runs")

    unmet = unmet_risk_obligations(CustomAgentAdapter().inspect(_Agent()), ())

    assert {item.dimension for item in unmet} == set(_RISK_DIMENSIONS)
    assert all(item.reason_code.endswith("_missing") for item in unmet)


def test_only_covered_or_partial_counts_as_satisfied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed on any other status.

    Today an obligation's seed is always APPLICABLE, so the evaluator returns
    only COVERED, PARTIAL or MISSING. This pins the *rule* rather than that
    accident: if some future outcome ever answers UNKNOWN or UNSUPPORTED, it
    must count as evidence that does not exist, not as satisfaction.
    """

    from agentcheck.coverage import analyzer
    from agentcheck.coverage.contract import BehavioralCoverageStatus

    for status in (
        BehavioralCoverageStatus.UNKNOWN,
        BehavioralCoverageStatus.UNSUPPORTED,
    ):
        monkeypatch.setattr(
            analyzer,
            "_evaluate_seed",
            lambda *_, _status=status, **__: analyzer._Outcome(_status, "why", ()),
        )
        unmet = unmet_risk_obligations(_spec("purge_records"), ())
        assert len(unmet) == len(_RISK_DIMENSIONS), status
