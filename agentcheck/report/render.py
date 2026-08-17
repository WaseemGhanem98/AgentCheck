from __future__ import annotations

import html
import json
import statistics
from collections import Counter
from typing import Any, Iterable

from agentcheck.domain import AgentSpec, CanonicalRun, CaseEvaluation, Finding, Scenario, Verdict
from agentcheck.generate.selection import SelectionPlan
from agentcheck.generate.suite import FrozenSuite
from agentcheck.privacy import redact_artifact
from agentcheck.review.contract import HumanReview, bound_reviews_for_finding


def _escape(value: Any) -> str:
    return html.escape(str(redact_artifact(value)), quote=True)


def _review_html(finding: Finding, reviews: tuple[HumanReview, ...]) -> str:
    related = tuple(item for item in reviews if item.finding_id == finding.finding_id)
    bound = bound_reviews_for_finding(finding, related)
    bound_ids = {item.review_id for item in bound}
    latest = bound[-1] if bound else None
    latest_decision = latest.decision if latest is not None else "none"
    latest_note = latest.note if latest is not None and latest.note else "None"
    mismatch = len(related) - len(bound)
    mismatch_html = (
        (
            f"<p class=\"muted\">{mismatch} human review record(s) no longer bind to "
            "this finding identity. The automated result is unchanged.</p>"
        )
        if mismatch
        else ""
    )
    history = "".join(
        f"<li><span class=\"mono\">{_escape(item.recorded_at.isoformat())}</span> — "
        f"{_escape(item.decision)}"
        + (f" · {_escape(item.note)}" if item.note else "")
        + (
            " · bound"
            if item.review_id in bound_ids
            else " · finding identity mismatch"
        )
        + "</li>"
        for item in related
    ) or "<li class=\"muted\">No human reviews.</li>"
    return (
        f"<p><b>Human review:</b> {_escape(latest_decision)} · "
        f"{len(bound)} bound of {_escape(len(related))} recorded</p>"
        f"<p><b>Latest note:</b> {_escape(latest_note)}</p>"
        f"{mismatch_html}"
        f"<details><summary>Review history ({len(related)})</summary>"
        f"<ul>{history}</ul></details>"
    )


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=False)
    value = redact_artifact(value)
    return html.escape(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True),
        quote=True,
    )


def _metric(value: Any, suffix: str = "") -> str:
    return "Unknown" if value is None else f"{_escape(value)}{suffix}"


def _percent(numerator: int, denominator: int) -> str:
    return "N/A" if denominator == 0 else f"{(numerator / denominator) * 100:.1f}%"


def _reported_total(values: list[int] | list[float], run_count: int, suffix: str = "") -> str:
    if run_count == 0 or len(values) != run_count:
        return f"Unknown ({len(values)}/{run_count} runs reported)"
    return _metric(round(sum(values), 6), suffix)


def _cards(items: Iterable[tuple[str, str]]) -> str:
    return "".join(
        f'<div class="metric"><span>{_escape(label)}</span><strong>{value}</strong></div>'
        for label, value in items
    )


def render_report(
    *,
    run_id: str,
    target: str,
    git_revision: str | None,
    spec: AgentSpec,
    scenarios: tuple[Scenario, ...],
    runs: tuple[CanonicalRun, ...],
    evaluations: tuple[CaseEvaluation, ...],
    findings: tuple[Finding, ...],
    include_instructions: bool = False,
    seed: int | None = None,
    frozen_suite: FrozenSuite | None = None,
    selection_plan: SelectionPlan | None = None,
    reviews: tuple[HumanReview, ...] = (),
) -> str:
    """Render a single escaped HTML file with no external resources or JavaScript."""

    counts = Counter(item.verdict for item in evaluations)
    total = len(evaluations)
    passed = counts[Verdict.PASS]
    latencies = [item.latency_ms for item in runs if item.latency_ms is not None]
    total_tokens_values = [item.usage.total_tokens for item in runs if item.usage.total_tokens is not None]
    cost_values = [item.usage.cost_usd for item in runs if item.usage.cost_usd is not None]
    mean_latency = round(statistics.fmean(latencies), 2) if latencies else None
    dimensions = sorted({tag for scenario in scenarios for tag in scenario.dimension_tags})
    tools = [item.value for item in spec.tools.items]
    capabilities = [item.value for item in spec.capabilities.items]
    evaluation_by_scenario = {item.scenario_id: item for item in evaluations}
    run_by_scenario = {item.scenario_id: item for item in runs}
    findings_by_scenario: dict[str, list[Finding]] = {}
    for finding in findings:
        for scenario_id in finding.affected_scenario_ids:
            findings_by_scenario.setdefault(scenario_id, []).append(finding)
    lineage_by_id = (
        {case.scenario.scenario_id: case.lineage for case in frozen_suite.cases}
        if frozen_suite is not None
        else {}
    )
    realization_by_id = (
        {
            case.scenario.scenario_id: case.realization
            for case in frozen_suite.cases
            if case.realization is not None
        }
        if frozen_suite is not None
        else {}
    )
    plan = selection_plan
    if plan is None and frozen_suite is not None:
        plan = frozen_suite.selection
    origin_counts = Counter(
        lineage.origin.value for lineage in lineage_by_id.values()
    )
    policy_ids = [item.value.policy_id for item in spec.policies.items]
    if frozen_suite is not None:
        for pack in frozen_suite.provenance.policy_packs:
            if pack not in policy_ids:
                policy_ids.append(pack)

    instruction_html = (
        f"<pre>{_escape(spec.instructions.system.value or '')}</pre>"
        if include_instructions
        else '<p class="muted">Raw system instructions are hidden by default.</p>'
    )
    finding_html = "".join(
        f"""
        <article class="finding severity-{_escape(finding.severity.value)}">
          <h3>{_escape(finding.title)}</h3>
          <p><b>Automated verdict:</b> FAIL</p>
          <p><b>Severity:</b> {_escape(finding.severity.value.upper())} · <b>Confidence:</b> {finding.confidence:.0%}</p>
          <p>{_escape(finding.description)}</p>
          <p><b>Affected:</b> {_escape(', '.join(finding.affected_scenario_ids))}</p>
          <p><b>Nearest passing:</b> {_escape(', '.join(finding.nearest_passing_scenario_ids) or 'None identified')}</p>
          <p><b>Likely layer:</b> {_escape(finding.root_cause_layer.value)}</p>
          <p><b>Likely cause:</b> {_escape(finding.likely_cause or 'Unknown')}</p>
          {''.join(f'<p><b>Suggested fix (human review required):</b> {_escape(fix.summary)}</p>' for fix in finding.suggested_fixes)}
          <p class="mono"><b>Evidence:</b> {_escape(', '.join(finding.evidence_ids))}</p>
          {_review_html(finding, reviews)}
        </article>
        """
        for finding in findings
    ) or '<p class="muted">No high-confidence deterministic findings.</p>'

    cases: list[str] = []
    for scenario in scenarios:
        evaluation = evaluation_by_scenario.get(scenario.scenario_id)
        case_run = run_by_scenario.get(scenario.scenario_id)
        verdict = evaluation.verdict.value if evaluation is not None else "INVALID"
        assertions = ""
        evidence = ""
        if evaluation is not None:
            assertions = "".join(
                f"""
                <li class="assertion result-{_escape(assertion.result.value.lower())}">
                  <b>{_escape(assertion.result.value)}</b> — {_escape(assertion.criterion)}
                  <p>{_escape(assertion.rationale)}</p>
                  <small>Evidence: {_escape(', '.join(assertion.supporting_evidence_ids) or 'none')} · Confidence: {assertion.confidence:.0%}</small>
                </li>
                """
                for assertion in evaluation.assertions
            )
            evidence = "".join(
                f"<details><summary>{_escape(item.evidence_id)} — {_escape(item.summary)}</summary><pre>{_json(item.data)}</pre></details>"
                for item in evaluation.evidence
            )
        trace = _json([item.model_dump(mode="json") for item in case_run.events]) if case_run else "[]"
        initial = _json(case_run.initial_world_state) if case_run else _json(scenario.initial_world_state)
        final = _json(case_run.final_world_state) if case_run else "Unavailable"
        related = findings_by_scenario.get(scenario.scenario_id, [])
        cause = "; ".join(item.likely_cause or "Unknown" for item in related) or "No failure cause identified."
        fix = "; ".join(
            suggested.summary for item in related for suggested in item.suggested_fixes
        ) or "No fix suggested."
        conversation = _json([turn.model_dump(mode="json") for turn in scenario.conversation_turns])
        lineage = lineage_by_id.get(scenario.scenario_id)
        lineage_html = ""
        if lineage is not None:
            extra = ""
            if lineage.mutation_kind:
                extra = (
                    f" · Parent <span class=\"mono\">{_escape(lineage.parent_scenario_id)}</span>"
                    f" · Mutation {_escape(lineage.mutation_kind)}"
                )
            lineage_html = (
                f"<section><h4>Lineage</h4><p>Origin {_escape(lineage.origin.value)}"
                f"{extra}</p></section>"
            )
        realization = realization_by_id.get(scenario.scenario_id)
        realization_html = ""
        if realization is not None:
            overlay = _json(list(realization.turns)) if realization.turns else "None recorded"
            realization_html = (
                "<section><h4>Realized wording (non-authoritative)</h4>"
                f"<p>Source {_escape(realization.source_kind)} · inferred · not authoritative · "
                f"provider {_escape(realization.provider)}</p>"
                f"<p>{_escape(realization.title)}</p>"
                f"<pre>{overlay}</pre></section>"
            )
        cases.append(
            f"""
            <details class="case" {'open' if verdict != 'PASS' else ''}>
              <summary><span class="badge verdict-{_escape(verdict.lower())}">{_escape(verdict)}</span> {_escape(scenario.title)} <span class="mono">{_escape(scenario.scenario_id)}</span></summary>
              <div class="case-grid">
                <section><h4>Conversation</h4><pre>{conversation}</pre></section>
                <section><h4>Assertions</h4><ul>{assertions}</ul></section>
                <section><h4>Evidence</h4>{evidence or '<p class="muted">No evidence packet.</p>'}</section>
                <section><h4>Initial state</h4><pre>{initial}</pre></section>
                <section><h4>Final state</h4><pre>{final}</pre></section>
                <section><h4>Observable trace</h4><pre>{trace}</pre></section>
                <section><h4>Likely cause</h4><p>{_escape(cause)}</p><h4>Suggested fix</h4><p>{_escape(fix)}</p></section>
                <section><h4>Reproducibility</h4><p>Seed {_escape(scenario.generation_seed)} · Fingerprint <span class="mono">{_escape(scenario.fingerprint)}</span></p></section>
                {lineage_html}
                {realization_html}
              </div>
            </details>
            """
        )

    suite_id = frozen_suite.suite_id if frozen_suite is not None else None
    identity_bits = [
        f"Spec <span class=\"mono\">{_escape(spec.spec_id)}</span>",
        f"Seed {_escape(seed if seed is not None else 'Unknown')}",
        f"Suite <span class=\"mono\">{_escape(suite_id or 'not recorded in this run')}</span>",
    ]
    origin_html = ""
    if origin_counts:
        origin_parts = [
            f"{origin_counts.get('built_in', 0)} built-in",
            f"{origin_counts.get('schema_boundary', 0)} schema-boundary",
            f"{origin_counts.get('workflow_mutation', 0)} workflow mutation",
        ]
        zero_input = origin_counts.get("zero_input_invocation", 0)
        if zero_input:
            origin_parts.append(f"{zero_input} zero-input invocation")
        origin_html = "<p>Case origins: " + " · ".join(origin_parts) + "</p>"
    policy_html = (
        f"<p>Declared policy packs: {_escape(', '.join(policy_ids))}</p>"
        if policy_ids
        else ""
    )
    coverage_extra = ""
    if frozen_suite is not None:
        coverage = frozen_suite.coverage
        coverage_extra = (
            f"<p>Covered tools: {_escape(', '.join(coverage.tools) or 'None recorded')}</p>"
            f"<p>Boundary kinds: {_escape(', '.join(coverage.boundary_kinds) or 'None recorded')}</p>"
            "<p>Unsupported schema features: "
            f"{_escape(', '.join(coverage.unsupported_schema_features) or 'None recorded')}</p>"
        )
    selection_html = ""
    if plan is not None:
        excluded_lines = "".join(
            f"<li><span class=\"mono\">{_escape(item.scenario_id)}</span> — "
            f"{_escape(item.reason)}</li>"
            for item in plan.decisions
            if not item.selected
        ) or "<li class=\"muted\">None</li>"
        selection_html = (
            f"<p>Selection algorithm {_escape(plan.algorithm)} · "
            f"budget {_escape(plan.max_cases if plan.max_cases is not None else 'covering set')} · "
            f"{len(plan.selected_ids)} selected · {len(plan.excluded_ids)} excluded</p>"
            f"<p>Covered dimensions: {_escape(', '.join(plan.coverage.covered) or 'None')}</p>"
            f"<p>Uncovered dimensions: {_escape(', '.join(plan.coverage.uncovered) or 'None')}</p>"
            f"<p>Unsupported: {_escape(', '.join(plan.coverage.unsupported) or 'None')}</p>"
            f"<p>Unknown: {_escape(', '.join(plan.coverage.unknown) or 'None')}</p>"
            f"<p>Excluded by selection (not scored):</p><ul>{excluded_lines}</ul>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AgentCheck report {_escape(run_id)}</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#0c111b; --card:#141c2a; --text:#e9eef7; --muted:#9aa9bd; --line:#2a3a50; --pass:#43c47a; --fail:#ff6577; --inc:#f5bd55; --infra:#a98df5; }}
    * {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,sans-serif }}
    main {{ max-width:1180px; margin:auto; padding:32px 20px 64px }} h1,h2,h3,h4 {{ line-height:1.2 }} .muted,small {{ color:var(--muted) }}
    .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:20px 0 }}
    .metric,.panel,.finding,.case {{ background:var(--card); border:1px solid var(--line); border-radius:10px }}
    .metric {{ padding:14px }} .metric span {{ display:block; color:var(--muted) }} .metric strong {{ font-size:1.45rem }}
    .panel,.finding {{ padding:18px; margin:14px 0 }} .finding {{ border-left:5px solid var(--inc) }} .severity-high {{ border-left-color:var(--fail) }}
    .case {{ margin:12px 0; overflow:hidden }} .case>summary {{ cursor:pointer; padding:16px; font-weight:650 }} .case-grid {{ padding:0 16px 16px; display:grid; gap:12px }}
    pre {{ white-space:pre-wrap; word-break:break-word; max-height:440px; overflow:auto; padding:12px; background:#090d14; border-radius:6px; color:#dce8f8 }}
    ul {{ padding-left:22px }} .assertion {{ margin:10px 0 }} .badge {{ display:inline-block; min-width:105px; text-align:center; margin-right:8px; padding:2px 8px; border-radius:999px; background:var(--line) }}
    .verdict-pass,.result-pass b {{ color:var(--pass) }} .verdict-fail,.result-fail b {{ color:var(--fail) }} .verdict-inconclusive,.result-inconclusive b {{ color:var(--inc) }} .verdict-infra_error,.result-infra_error b {{ color:var(--infra) }}
    .mono {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.88em }} details details {{ margin:8px 0 }} a {{ color:#82b8ff }}
  </style>
</head>
<body><main>
  <header><p class="muted">AgentCheck deterministic evaluation report</p><h1>{_escape(spec.identity.name.value)}</h1><p>Target {_escape(target)} · Run <span class="mono">{_escape(run_id)}</span> · Git {_escape(git_revision or 'Unknown')}</p><p>{' · '.join(identity_bits)}</p></header>
  <section class="metrics">{_cards((
      ('Observed suite pass rate', _percent(passed, total)),
      ('Passed', str(passed)),
      ('Failed', str(counts[Verdict.FAIL])),
      ('Inconclusive', str(counts[Verdict.INCONCLUSIVE])),
      ('Infra errors', str(counts[Verdict.INFRA_ERROR])),
      ('Mean latency', _metric(mean_latency, ' ms')),
      ('Token usage', _reported_total(total_tokens_values, len(runs))),
      ('Known cost', _reported_total(cost_values, len(runs), ' USD')),
  ))}</section>
  <section class="panel"><h2>Target and AgentSpec</h2><p>Framework: {_escape(spec.identity.framework.value)} {_escape(spec.identity.framework_version.value or '')} · Model: {_escape(spec.identity.model.value or 'Unknown')}</p><p>Tools ({len(tools)}): {_escape(', '.join(tool.name for tool in tools))}</p><p>Capabilities ({len(capabilities)}): {_escape(', '.join(item.name for item in capabilities) or 'None derived')}</p><p>Unknown properties: {len(spec.unknowns)}</p>{policy_html}<h3>Instructions</h3>{instruction_html}</section>
  <section class="panel"><h2>Coverage and reproducibility</h2><p>{len(scenarios)} valid scenarios · {len(dimensions)} distinct dimension tags</p><p>{_escape(', '.join(dimensions))}</p>{origin_html}{coverage_extra}{selection_html}<p>Every case records its generation seed and structural fingerprint. Invalid scenarios are excluded from these counts. Cases excluded by coverage selection are listed above and are not scored as passing.</p></section>
  <section><h2>Findings</h2>{finding_html}</section>
  <section><h2>Scenarios</h2>{''.join(cases)}</section>
</main></body></html>"""
