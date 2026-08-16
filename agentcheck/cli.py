"""Command-line interface for the deterministic AgentCheck Phase 1 workflow."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from typing import Sequence

from agentcheck import __version__
from agentcheck.application import execute_suite, inspect_target
from agentcheck.domain import AgentSpec, Severity, Verdict
from agentcheck.errors import ScenarioValidationError
from agentcheck.privacy import redact_artifact, redact_log_text


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$")


def _run_id(value: str) -> str:
    if _RUN_ID_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "run ID must contain only letters, digits, underscores, or hyphens"
        )
    return value


def _seed(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if parsed < 0 or parsed > 2**63 - 1:
        raise argparse.ArgumentTypeError("seed must be between 0 and 2^63 - 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentcheck",
        description="Safely evaluate a trusted local AI agent with deterministic scenarios.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect",
        help="import a trusted local agent and inspect it without running a turn",
        description=(
            "Import the configured trusted local agent in a child process "
            "(executing its module-level code), then inspect its metadata without "
            "running an agent turn."
        ),
    )
    inspect_parser.add_argument("target", nargs="?", default=".")
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="emit the versioned AgentSpec as JSON",
    )

    test_parser = commands.add_parser(
        "test",
        help="run the deterministic Phase 1 suite in isolated child processes",
    )
    test_parser.add_argument("target", nargs="?", default=".")
    test_parser.add_argument("--seed", type=_seed, help="override the configured suite seed")
    test_parser.add_argument(
        "--run-id",
        type=_run_id,
        help="set a safe artifact run ID (normally generated automatically)",
    )
    return parser


def _print_inspection(spec: AgentSpec) -> None:
    tools = [item.value for item in spec.tools.items]
    capabilities = [item.value for item in spec.capabilities.items]
    state_changing = sum(item.state_changing for item in tools)
    destructive = sum(item.destructive for item in tools)
    print(f"Agent: {spec.identity.name.value}")
    print(
        "Framework: "
        f"{spec.identity.framework.value} "
        f"{spec.identity.framework_version.value or '(unknown version)'}"
    )
    print(f"Model: {spec.identity.model.value or '(unknown)'}")
    print()
    print("Detected:")
    print(f"- {len(tools)} tools")
    print(f"- {len(capabilities)} capabilities")
    print(f"- {state_changing} state-changing actions")
    print(f"- {destructive} destructive actions")
    if tools:
        print()
        print("Tools:")
        for tool in tools:
            markers = []
            if tool.state_changing:
                markers.append("state-changing")
            if tool.destructive:
                markers.append("destructive")
            suffix = f" ({', '.join(markers)})" if markers else ""
            print(f"✓ {tool.name}{suffix}")
    if spec.unknowns:
        print()
        print(f"Unknown properties: {len(spec.unknowns)}")


def _json_spec(spec: AgentSpec) -> str:
    payload = redact_artifact(spec.model_dump(mode="json", exclude_none=False))
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)


def _inspect_command(target: str, *, as_json: bool) -> int:
    if not as_json:
        print("Inspecting agent...")
        print()
    _, _, result = inspect_target(target)
    spec = result.require_value()
    if as_json:
        print(_json_spec(spec))
    else:
        _print_inspection(spec)
    return 0


def _test_command(target: str, *, seed: int | None, run_id: str | None) -> int:
    print("Inspecting agent...")
    print()
    execution = execute_suite(target, seed=seed, run_id=run_id)
    if not execution.scenarios:
        raise ScenarioValidationError(
            "No valid scenarios remain after linting; no agent verdict was produced."
        )
    _print_inspection(execution.spec)
    print()
    print("Building deterministic test suite...")
    print(f"Generated: {len(execution.scenarios)} valid scenarios")
    if execution.invalid_scenarios:
        print(
            f"Excluded: {len(execution.invalid_scenarios)} invalid scenario(s); "
            "these do not affect results"
        )
    print()
    print(f"Running {len(execution.scenarios)} scenarios in isolated child processes...")
    print()
    evaluation_by_id = {item.scenario_id: item for item in execution.evaluations}
    for scenario in execution.scenarios:
        evaluation = evaluation_by_id[scenario.scenario_id]
        print(f"{evaluation.verdict.value:<12} {scenario.title}")

    counts: Counter[Verdict] = execution.counts
    pass_rate = execution.observed_pass_rate
    print()
    print(
        "Observed suite pass rate: "
        f"{pass_rate * 100:.1f}%" if pass_rate is not None else "Observed suite pass rate: N/A"
    )
    print()
    print(f"Passed:        {counts[Verdict.PASS]}")
    print(f"Failed:        {counts[Verdict.FAIL]}")
    print(f"Inconclusive:  {counts[Verdict.INCONCLUSIVE]}")
    print(f"Infra errors:  {counts[Verdict.INFRA_ERROR]}")

    severity_counts = Counter(item.severity for item in execution.findings)
    print()
    print(f"High-confidence failures discovered: {counts[Verdict.FAIL]}")
    print(f"High-severity findings: {severity_counts[Severity.HIGH]}")
    print(f"Medium findings:        {severity_counts[Severity.MEDIUM]}")
    print()
    print("Report:")
    print(execution.report_path)

    if counts[Verdict.INFRA_ERROR]:
        return 2
    if counts[Verdict.FAIL]:
        return 1
    if counts[Verdict.INCONCLUSIVE]:
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect_command(args.target, as_json=args.json)
        if args.command == "test":
            return _test_command(args.target, seed=args.seed, run_id=args.run_id)
    except KeyboardInterrupt:
        print("AgentCheck interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        message = redact_log_text(str(exc)) or type(exc).__name__
        print(f"AgentCheck error: {message}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":  # pragma: no cover - console script is the primary path
    raise SystemExit(main())
