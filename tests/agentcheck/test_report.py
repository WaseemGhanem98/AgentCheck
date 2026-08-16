from __future__ import annotations

from typing import TypeVar

from agentcheck.domain import (
    AgentProperty,
    AgentSpec,
    CapabilitiesSpec,
    CanonicalRun,
    IdentitySpec,
    InspectionProvenance,
    InstructionsSpec,
    InterfaceSpec,
    ObservabilitySpec,
    RunTermination,
    RuntimeSpec,
    SourceKind,
    SourceReference,
    SpecEvidence,
    ToolsSpec,
    UsageMetrics,
    utc_now,
)
from agentcheck.report import render_report
from agentcheck.generate import build_account_support_suite


T = TypeVar("T")


def _property(value: T) -> AgentProperty[T]:
    return AgentProperty(value=value, source=SourceReference(kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"), confidence=1, evidence=(SpecEvidence(evidence_id="e", summary="test"),))


def _spec(hostile: str, *, system_prompt: str = "TOP SECRET SYSTEM PROMPT") -> AgentSpec:
    return AgentSpec(
        spec_id="spec",
        identity=IdentitySpec(name=_property(hostile), framework=_property("OpenAI Agents SDK"), framework_version=_property("0.20.0"), provider=_property(None), model=_property(None)),
        interface=InterfaceSpec(entrypoint=_property("agent.py:agent"), input_modalities=_property(("text",)), output_modalities=_property(("text",)), input_schema=_property(None), output_schema=_property(None), interactive=_property(True)),
        instructions=InstructionsSpec(system=_property(system_prompt), developer=_property(None)),
        capabilities=CapabilitiesSpec(),
        tools=ToolsSpec(),
        runtime=RuntimeSpec(max_model_turns=_property(None), max_tool_calls=_property(None), timeout_seconds=_property(None), token_budget=_property(None), cost_budget_usd=_property(None)),
        observability=ObservabilitySpec(supported_event_types=_property(("final_output",)), usage_metrics=_property(()), provider_request_ids=_property(False), source_event_links=_property(True)),
        provenance=InspectionProvenance(inspector="test", inspector_version="1", inspected_at=utc_now(), target="test", sources=(SourceReference(kind=SourceKind.RUNTIME_INTROSPECTION, locator="test"),)),
    )


def test_report_escapes_xss_and_hides_system_prompt() -> None:
    hostile = '</pre><script>alert("x")</script><img src=x onerror=alert(1)>'

    report = render_report(run_id="run", target=hostile, git_revision=None, spec=_spec(hostile), scenarios=(), runs=(), evaluations=(), findings=())

    assert hostile not in report
    assert "&lt;script&gt;" in report
    assert "TOP SECRET SYSTEM PROMPT" not in report
    assert "<script" not in report.casefold()
    assert "https://" not in report
    assert "Content-Security-Policy" in report


def test_report_redacts_secrets_in_instructions_and_structured_state() -> None:
    scenario = build_account_support_suite()[0].model_copy(
        update={
            "initial_world_state": {
                "password": "world-secret-value",
                "message": "Bearer report-secret-token",
            }
        }
    )
    report = render_report(
        run_id="run",
        target="api_key=sk-targetsecret12345",
        git_revision=None,
        spec=_spec(
            "safe",
            system_prompt="api_key=sk-systemsecret12345",
        ),
        scenarios=(scenario,),
        runs=(),
        evaluations=(),
        findings=(),
        include_instructions=True,
    )

    assert "targetsecret" not in report
    assert "systemsecret" not in report
    assert "world-secret-value" not in report
    assert "report-secret-token" not in report
    assert "[REDACTED]" in report


def test_report_does_not_present_partial_usage_as_a_complete_total() -> None:
    now = utc_now()
    known = CanonicalRun(
        run_id="known",
        scenario_id="case-1",
        target_id="spec",
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
        usage=UsageMetrics(total_tokens=12, cost_usd=0.25),
    )
    unknown = CanonicalRun(
        run_id="unknown",
        scenario_id="case-2",
        target_id="spec",
        started_at=now,
        ended_at=now,
        termination=RunTermination.COMPLETED,
    )

    report = render_report(
        run_id="run",
        target="target",
        git_revision=None,
        spec=_spec("agent"),
        scenarios=(),
        runs=(known, unknown),
        evaluations=(),
        findings=(),
    )

    assert report.count("Unknown (1/2 runs reported)") == 2
    assert ">12<" not in report
    assert ">0.25 USD<" not in report
