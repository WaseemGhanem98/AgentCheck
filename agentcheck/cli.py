"""Command-line interface for the deterministic AgentCheck workflow."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from typing import Sequence

from agentcheck import __version__
from agentcheck.application import execute_suite, generate_suite, inspect_target, replay_suite
from agentcheck.config import DEFAULT_ENTRYPOINT, entrypoint_location
from agentcheck.domain import AgentSpec, Severity, Verdict
from agentcheck.errors import ConfigurationError, ScenarioValidationError
from agentcheck.generate.mutations import DEFAULT_MAX_MUTATIONS, MAX_MUTATIONS_PER_SUITE
from agentcheck.generate.selection import MAX_CASES
from agentcheck.initialize import DEFAULT_ADAPTER, SUPPORTED_ADAPTERS, write_initial_config
from agentcheck.inspect.capabilities import ExtractedCapability, extract_capabilities
from agentcheck.privacy import redact_artifact, redact_log_text
from agentcheck.report import render_stored_run


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$")


def _run_id(value: str) -> str:
    if _RUN_ID_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "run ID must contain only letters, digits, underscores, or hyphens"
        )
    return value


def _max_cases(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max-cases must be an integer") from exc
    if parsed < 1 or parsed > MAX_CASES:
        raise argparse.ArgumentTypeError(
            f"max-cases must be between 1 and {MAX_CASES}"
        )
    return parsed


def _max_mutations(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("max-mutations must be an integer") from exc
    if parsed < 1 or parsed > MAX_MUTATIONS_PER_SUITE:
        raise argparse.ArgumentTypeError(
            f"max-mutations must be between 1 and {MAX_MUTATIONS_PER_SUITE}"
        )
    return parsed


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

    init_parser = commands.add_parser(
        "init",
        help="write an explicit agentcheck.json for a target directory",
        description=(
            "Write an explicit local AgentCheck configuration. This command never "
            "imports or executes the target, so it is safe to run against code that "
            "has not been reviewed yet."
        ),
    )
    init_parser.add_argument("target", nargs="?", default=".")
    init_parser.add_argument(
        "--entrypoint",
        default=DEFAULT_ENTRYPOINT,
        help="agent source and attribute inside the target, as 'relative/path.py:attribute'",
    )
    init_parser.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        choices=SUPPORTED_ADAPTERS,
        help="framework adapter used to inspect and run the target",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing agentcheck.json instead of refusing",
    )

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

    generate_parser = commands.add_parser(
        "generate",
        help="derive, lint, and freeze a deterministic suite for a trusted local agent",
        description=(
            "Import the configured trusted local agent in a child process "
            "(executing its module-level code), derive the supported built-in and "
            "schema-boundary cases, lint them, and write a frozen suite. Workflow "
            "mutations are off by default. Coverage selection is off unless "
            "--max-cases is passed. The written file is inert data, not a "
            "replay manifest, and is not a security sandbox."
        ),
    )
    generate_parser.add_argument("target", nargs="?", default=".")
    generate_parser.add_argument(
        "--seed",
        type=_seed,
        help="override the configured suite seed recorded into the frozen suite",
    )
    generate_parser.add_argument(
        "--out",
        help=(
            "relative path inside the target for the frozen suite "
            "(defaults to suite_path or agentcheck-suite.json)"
        ),
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing frozen suite instead of refusing",
    )
    generate_parser.add_argument(
        "--mutations",
        action="store_true",
        help="include bounded workflow mutations of lint-clean built-in cases",
    )
    generate_parser.add_argument(
        "--max-mutations",
        type=_max_mutations,
        help=(
            "maximum mutation-derived cases to freeze (requires --mutations; "
            f"default {DEFAULT_MAX_MUTATIONS})"
        ),
    )
    generate_parser.add_argument(
        "--policy-pack",
        action="append",
        dest="policy_packs",
        metavar="NAME",
        help=(
            "declare a versioned policy pack by built-in name or in-target path "
            "(repeatable)"
        ),
    )
    generate_parser.add_argument(
        "--realize",
        action="store_true",
        help=(
            "opt in to consent-gated LLM rewriting of display text only; "
            "requires llm_realization.enabled and an allowlisted provider credential"
        ),
    )
    generate_parser.add_argument(
        "--max-cases",
        type=_max_cases,
        help=(
            "maximum lint-clean cases to freeze after deterministic coverage "
            f"selection (default: keep every valid case; maximum {MAX_CASES})"
        ),
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
    test_parser.add_argument(
        "--no-store",
        action="store_true",
        help="skip writing the local SQLite evaluation index",
    )
    test_parser.add_argument(
        "--select",
        choices=("coverage",),
        help=(
            "run a deterministic coverage-maximizing subset instead of every "
            "valid case; excluded cases are recorded and never scored as passing"
        ),
    )

    report_parser = commands.add_parser(
        "report",
        help="render a stored run's HTML report without rerunning cases",
        description=(
            "Read stored evaluation artifacts and regenerate the offline HTML "
            "report. This command never imports the target, never spawns a worker, "
            "and never makes a network call."
        ),
    )
    report_parser.add_argument("target", nargs="?", default=".")
    report_parser.add_argument(
        "--run-id",
        type=_run_id,
        help="render this stored run ID",
    )
    report_parser.add_argument(
        "--latest",
        action="store_true",
        help="render the most recently indexed run (default when --run-id is omitted)",
    )
    report_parser.add_argument(
        "--out",
        help="relative path inside the target for the HTML report",
    )

    replay_parser = commands.add_parser(
        "replay",
        help="re-execute a stored replay manifest without using disclosure artifacts",
        description=(
            "Load an agentcheck.replay_manifest.v1 document as untrusted input, "
            "verify spec and source bindings, and re-run its scenarios through "
            "the existing isolated ToolGateway. Frozen suites, suite.json, SQLite, "
            "and HTML reports are not replay manifests. This command executes "
            "trusted local agent code and is not a sandbox for hostile repositories."
        ),
    )
    replay_parser.add_argument("target", nargs="?", default=".")
    replay_parser.add_argument(
        "--manifest",
        required=True,
        help="relative path inside the target to a replay manifest",
    )
    replay_parser.add_argument(
        "--run-id",
        type=_run_id,
        help="set a safe artifact run ID for the new replay run",
    )
    replay_parser.add_argument(
        "--no-store",
        action="store_true",
        help="skip writing the local SQLite evaluation index",
    )
    return parser


def _capability_line(extracted: ExtractedCapability) -> str:
    capability = extracted.capability
    markers = [capability.action_kind.value]
    markers.append("state-changing" if capability.state_changing else "read-only")
    if capability.destructive:
        markers.append("destructive")
    arguments = extracted.arguments
    if arguments.schema_known:
        surface = (
            f"{len(arguments.required_parameters)} required, "
            f"{len(arguments.optional_parameters)} optional argument(s)"
        )
        unknown_types = sum(
            1 for parameter in arguments.parameters if not parameter.types_known
        )
        if unknown_types:
            surface += f", {unknown_types} of unknown type"
    else:
        surface = "argument surface unknown"
    return (
        f"{extracted.tool_name}: {', '.join(markers)} "
        f"(confidence {extracted.confidence:.2f}); {surface}"
    )


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
        print()
        print("Capabilities (action kind and risk are inferred, not authoritative):")
        for extracted in extract_capabilities(tools):
            print(f"- {_capability_line(extracted)}")
    if spec.policies.items:
        print()
        print("Declared policy packs:")
        for item in spec.policies.items:
            version = item.value.version or "unknown"
            print(f"- {item.value.policy_id} (v{version})")
    if spec.unknowns:
        print()
        print(f"Unknown properties: {len(spec.unknowns)}")


def _json_spec(spec: AgentSpec) -> str:
    payload = redact_artifact(spec.model_dump(mode="json", exclude_none=False))
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)


def _init_command(target: str, *, entrypoint: str, adapter: str, force: bool) -> int:
    config_path = write_initial_config(
        target,
        adapter=adapter,
        entrypoint=entrypoint,
        force=force,
    )
    root = config_path.parent
    print("AgentCheck configuration written.")
    print()
    print(f"Config:     {config_path}")
    print(f"Adapter:    {adapter}")
    print(f"Entrypoint: {entrypoint.strip()}")
    source, _ = entrypoint_location(root, entrypoint.strip())
    if not source.is_file():
        print()
        print(f"Note: the entrypoint source does not exist yet: {source}")
        print("Create it, or re-run init with --entrypoint and --force.")
    print()
    print("Next steps:")
    print(f"- agentcheck inspect {root}")
    print(f"- agentcheck generate {root}")
    print(f"- agentcheck test {root}")
    return 0


def _generate_command(
    target: str,
    *,
    seed: int | None,
    out: str | None,
    force: bool,
    include_mutations: bool,
    max_mutations: int | None,
    policy_packs: list[str] | None,
    max_cases: int | None,
    realize: bool,
) -> int:
    if max_mutations is not None and not include_mutations:
        raise ConfigurationError("--max-mutations requires --mutations")
    print("Inspecting agent...")
    print()
    generation = generate_suite(
        target,
        seed=seed,
        out=out,
        force=force,
        include_mutations=include_mutations,
        max_mutations=max_mutations,
        policy_packs=policy_packs,
        max_cases=max_cases,
        realize=realize,
    )
    _print_inspection(generation.spec)
    print()
    print("Frozen suite written.")
    print()
    print(f"Suite:        {generation.suite_path}")
    print(f"Suite ID:     {generation.suite.suite_id}")
    print(f"Spec ID:      {generation.suite.spec_id}")
    print(f"Seed:         {generation.suite.seed}")
    print(f"Cases:        {len(generation.suite.cases)}")
    mutation_cases = sum(
        1
        for case in generation.suite.cases
        if case.lineage.origin.value == "workflow_mutation"
    )
    if include_mutations:
        print(f"Mutations:    {mutation_cases}")
    print(f"Rejected:     {len(generation.suite.rejected)}")
    if generation.suite.selection is not None:
        print(f"Selected:     {len(generation.suite.selection.selected_ids)}")
        print(f"Unselected:   {len(generation.suite.selection.excluded_ids)}")
    realized = sum(1 for case in generation.suite.cases if case.realization is not None)
    if realized:
        print(f"Realized:     {realized} display overlay(s)")
    print(f"Fingerprint:  {generation.suite.fingerprint}")
    print()
    print("Next steps:")
    print(f"- agentcheck test {generation.target_root}")
    return 0


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


def _test_command(
    target: str,
    *,
    seed: int | None,
    run_id: str | None,
    persist_store: bool,
    select: str | None,
) -> int:
    print("Inspecting agent...")
    print()
    execution = execute_suite(
        target,
        seed=seed,
        run_id=run_id,
        persist_store=persist_store,
        select=select,
    )
    if not execution.scenarios:
        raise ScenarioValidationError(
            "No valid scenarios remain after linting; no agent verdict was produced."
        )
    _print_inspection(execution.spec)
    print()
    if execution.frozen_suite is not None:
        print("Using frozen suite...")
        print(f"Suite ID: {execution.frozen_suite.suite_id}")
        print(f"Seed:     {execution.frozen_suite.seed}")
    else:
        print("Building deterministic test suite...")
    print(f"Generated: {len(execution.scenarios)} valid scenarios")
    if execution.invalid_scenarios:
        print(
            f"Excluded: {len(execution.invalid_scenarios)} invalid scenario(s); "
            "these do not affect results"
        )
    if execution.selection is not None and execution.selection.excluded_ids:
        print(
            f"Excluded by selection: {len(execution.selection.excluded_ids)} "
            "valid scenario(s); these are not scored"
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
    if execution.replay_manifest_path is not None:
        print("Replay manifest:")
        print(execution.replay_manifest_path)

    if counts[Verdict.INFRA_ERROR]:
        return 2
    if counts[Verdict.FAIL]:
        return 1
    if counts[Verdict.INCONCLUSIVE]:
        return 3
    return 0


def _report_command(
    target: str,
    *,
    run_id: str | None,
    latest: bool,
    out: str | None,
) -> int:
    generated = render_stored_run(target, run_id=run_id, latest=latest, out=out)
    print("Regenerating report from stored artifacts...")
    print()
    print(f"Run ID: {generated.run_id}")
    print("Report:")
    print(generated.report_path)
    return 0


def _replay_command(
    target: str,
    *,
    manifest: str,
    run_id: str | None,
    persist_store: bool,
) -> int:
    print("Loading replay manifest...")
    print()
    execution = replay_suite(
        target,
        manifest,
        run_id=run_id,
        persist_store=persist_store,
    )
    _print_inspection(execution.spec)
    print()
    print("Replaying stored scenarios...")
    print(f"Manifest seed: {execution.seed}")
    print(f"Generated: {len(execution.scenarios)} valid scenarios")
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
    print()
    print("Report:")
    print(execution.report_path)
    if execution.replay_manifest_path is not None:
        print("Replay manifest:")
        print(execution.replay_manifest_path)

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
        if args.command == "init":
            return _init_command(
                args.target,
                entrypoint=args.entrypoint,
                adapter=args.adapter,
                force=args.force,
            )
        if args.command == "inspect":
            return _inspect_command(args.target, as_json=args.json)
        if args.command == "generate":
            return _generate_command(
                args.target,
                seed=args.seed,
                out=args.out,
                force=args.force,
                include_mutations=args.mutations,
                max_mutations=args.max_mutations,
                policy_packs=args.policy_packs,
                max_cases=args.max_cases,
                realize=args.realize,
            )
        if args.command == "test":
            return _test_command(
                args.target,
                seed=args.seed,
                run_id=args.run_id,
                persist_store=not args.no_store,
                select=args.select,
            )
        if args.command == "report":
            return _report_command(
                args.target,
                run_id=args.run_id,
                latest=args.latest,
                out=args.out,
            )
        if args.command == "replay":
            return _replay_command(
                args.target,
                manifest=args.manifest,
                run_id=args.run_id,
                persist_store=not args.no_store,
            )
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
