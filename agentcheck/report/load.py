"""Load stored AgentCheck run artifacts as untrusted input.

JSON/JSONL files remain authoritative. The SQLite index is used only to resolve
``--latest``. This module never imports a target, never spawns a worker, and
never makes a network call.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentcheck.artifacts import replace_private_file
from agentcheck.config import (
    CONFIG_FILENAME,
    AgentCheckConfig,
    contained_path,
    normalize_target,
)
from agentcheck.coverage import (
    BehavioralCoverage,
    BehavioralCoverageReferenceScope,
    analyze_behavioral_coverage,
    verify_behavioral_coverage_binding,
)
from agentcheck.domain import (
    AGENT_SPEC_CONTRACT_VERSION,
    CANONICAL_RUN_CONTRACT_VERSION,
    CASE_EVALUATION_CONTRACT_VERSION,
    FINDING_CONTRACT_VERSION,
    AgentSpec,
    CanonicalRun,
    CaseEvaluation,
    Finding,
    Scenario,
)
from agentcheck.errors import ConfigurationError
from agentcheck.generate.suite import FrozenSuite, configured_frozen_suite
from agentcheck.generate.selection import SelectionPlan
from agentcheck.store import (
    StoreError,
    default_store_relative_path,
    list_runs_readonly,
    resolve_store_path,
)
from agentcheck.review.store import load_reviews_for_run

from .render import render_report


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$")
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_JSONL_BYTES = 16 * 1024 * 1024
_MAX_CONFIG_BYTES = 64 * 1024
_REQUIRED_ARTIFACTS = (
    "agent-spec.json",
    "suite.json",
    "evaluations.jsonl",
    "runs.jsonl",
    "findings.json",
    "summary.json",
)
_SUITE_CONTRACT = "agentcheck.suite.v1"
_SUMMARY_CONTRACT = "agentcheck.summary.v1"
_INVALID_SCENARIOS_CONTRACT = "agentcheck.invalid_scenarios.v1"
_MISSING_BEHAVIORAL_COVERAGE = object()


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """Validated artifacts for one stored suite run."""

    root: Path
    config: AgentCheckConfig
    run_id: str
    run_directory: Path
    spec: AgentSpec
    scenarios: tuple[Scenario, ...]
    runs: tuple[CanonicalRun, ...]
    evaluations: tuple[CaseEvaluation, ...]
    findings: tuple[Finding, ...]
    seed: int
    git_revision: str | None
    frozen_suite: FrozenSuite | None
    selection: SelectionPlan | None
    behavioral_coverage: BehavioralCoverage


@dataclass(frozen=True, slots=True)
class StoredReport:
    run_id: str
    report_path: Path
    loaded: LoadedRun


def load_report_config(target: str | os.PathLike[str]) -> tuple[Path, AgentCheckConfig]:
    """Load AgentCheck config without importing or requiring the entrypoint."""

    root = normalize_target(target)
    config_path = root / CONFIG_FILENAME
    if not config_path.exists():
        return root, AgentCheckConfig()
    document = _read_json_object(config_path, max_bytes=_MAX_CONFIG_BYTES)
    try:
        return root, AgentCheckConfig.model_validate(document)
    except ValueError as exc:
        raise ConfigurationError(f"invalid {CONFIG_FILENAME}: {exc}") from exc


def resolve_stored_run_id(
    root: Path,
    config: AgentCheckConfig,
    *,
    run_id: str | None,
    latest: bool,
) -> tuple[str, str]:
    """Return ``(run_id, source)`` where source is ``cli``, ``store``, or ``filesystem``."""

    if run_id and latest:
        raise ConfigurationError("pass only one of --run-id and --latest")
    if run_id:
        if _SAFE_RUN_ID.fullmatch(run_id) is None:
            raise ConfigurationError(
                "run ID must contain only letters, digits, underscores, or hyphens"
            )
        return run_id, "cli"

    indexed = _latest_from_store(root, config)
    if indexed is not None:
        return indexed, "store"
    return _latest_from_filesystem(root, config), "filesystem"


def load_stored_run(
    target: str | os.PathLike[str],
    *,
    run_id: str | None = None,
    latest: bool = False,
) -> LoadedRun:
    root, config = load_report_config(target)
    resolved_id, source = resolve_stored_run_id(
        root, config, run_id=run_id, latest=latest
    )
    directory = _run_directory(root, config, resolved_id, source=source)
    spec = _load_spec(directory / "agent-spec.json")
    seed, scenarios = _load_suite(directory / "suite.json", resolved_id)
    invalid_scenario_ids = _load_invalid_scenario_ids(
        directory / "invalid-scenarios.json"
    )
    git_revision, selection, behavioral_coverage = _load_summary(
        directory / "summary.json",
        resolved_id,
        seed,
        spec=spec,
        scenarios=scenarios,
        invalid_scenario_ids=invalid_scenario_ids,
    )
    evaluations = _load_jsonl(
        directory / "evaluations.jsonl",
        CaseEvaluation,
        version_field="contract_version",
        expected=CASE_EVALUATION_CONTRACT_VERSION,
        filename="evaluations.jsonl",
    )
    runs = _load_jsonl(
        directory / "runs.jsonl",
        CanonicalRun,
        version_field="contract_version",
        expected=CANONICAL_RUN_CONTRACT_VERSION,
        filename="runs.jsonl",
    )
    findings = _load_findings(directory / "findings.json")
    frozen = _optional_frozen_suite(root, config, spec)
    return LoadedRun(
        root=root,
        config=config,
        run_id=resolved_id,
        run_directory=directory,
        spec=spec,
        scenarios=scenarios,
        runs=runs,
        evaluations=evaluations,
        findings=findings,
        seed=seed,
        git_revision=git_revision,
        frozen_suite=frozen,
        selection=selection,
        behavioral_coverage=behavioral_coverage,
    )


def render_stored_run(
    target: str | os.PathLike[str],
    *,
    run_id: str | None = None,
    latest: bool = False,
    out: str | None = None,
) -> StoredReport:
    loaded = load_stored_run(target, run_id=run_id, latest=latest)
    reviews = load_reviews_for_run(loaded.root, loaded.config, loaded.run_id)
    # This configured frozen suite may postdate the stored run. It remains
    # useful display metadata, but must not become a new source binding after
    # the loader has verified coverage against the persisted artifacts.
    html = render_report(
        run_id=loaded.run_id,
        target=str(loaded.root),
        git_revision=loaded.git_revision,
        spec=loaded.spec,
        scenarios=loaded.scenarios,
        runs=loaded.runs,
        evaluations=loaded.evaluations,
        findings=loaded.findings,
        include_instructions=loaded.config.include_instructions_in_report,
        seed=loaded.seed,
        frozen_suite=loaded.frozen_suite,
        selection_plan=loaded.selection,
        behavioral_coverage=loaded.behavioral_coverage,
        reviews=reviews,
        _verify_frozen_coverage_binding=False,
    )
    destination = _report_destination(loaded, out)
    try:
        parent = destination.parent
        if parent != loaded.root.resolve() and parent != loaded.root:
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                os.chmod(parent, 0o700)
            except OSError:
                pass
        replace_private_file(destination, html.encode("utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"unable to write report: {exc}") from exc
    return StoredReport(run_id=loaded.run_id, report_path=destination, loaded=loaded)


def _latest_from_store(root: Path, config: AgentCheckConfig) -> str | None:
    relative = default_store_relative_path(config)
    unresolved = root.resolve() / Path(relative)
    if not unresolved.exists():
        return None
    try:
        path = resolve_store_path(root, config)
        runs = list_runs_readonly(path)
    except StoreError as exc:
        raise ConfigurationError(f"evaluation store is unreadable: {exc}") from exc
    if not runs:
        return None
    return runs[-1].run_id


def _latest_from_filesystem(root: Path, config: AgentCheckConfig) -> str:
    relative = f"{config.artifacts_directory}/runs"
    unresolved = root.resolve() / Path(relative)
    if not unresolved.exists():
        raise ConfigurationError("no stored AgentCheck runs were found")
    try:
        runs_root = contained_path(root, relative)
    except ConfigurationError as exc:
        raise ConfigurationError("no stored AgentCheck runs were found") from exc
    if runs_root.is_symlink() or not runs_root.is_dir():
        raise ConfigurationError("no stored AgentCheck runs were found")
    candidates: list[str] = []
    try:
        children = list(runs_root.iterdir())
    except OSError as exc:
        raise ConfigurationError(f"unable to list stored runs: {exc}") from exc
    for child in children:
        if child.is_symlink():
            continue
        if _SAFE_RUN_ID.fullmatch(child.name) is None:
            continue
        if child.is_dir() and _has_required_artifacts(child):
            candidates.append(child.name)
    if not candidates:
        raise ConfigurationError("no stored AgentCheck runs were found")
    return max(candidates)


def _run_directory(
    root: Path, config: AgentCheckConfig, run_id: str, *, source: str
) -> Path:
    relative = f"{config.artifacts_directory}/runs/{run_id}"
    unresolved = root.resolve() / Path(relative)
    if unresolved.is_symlink():
        raise ConfigurationError("run artifact path must not be a symlink")
    try:
        directory = contained_path(root, relative)
    except ConfigurationError as exc:
        raise ConfigurationError(
            f"run artifacts were not found for {run_id!r}"
        ) from exc
    if not directory.is_dir():
        if source == "store":
            raise ConfigurationError(
                f"indexed run {run_id!r} has no artifacts; the evaluation store is stale"
            )
        raise ConfigurationError(f"run artifacts were not found for {run_id!r}")
    missing = [name for name in _REQUIRED_ARTIFACTS if not (directory / name).is_file()]
    if missing:
        raise ConfigurationError(
            f"run {run_id!r} is missing required artifact(s): {', '.join(missing)}"
        )
    return directory


def _has_required_artifacts(directory: Path) -> bool:
    return all((directory / name).is_file() and not (directory / name).is_symlink()
               for name in _REQUIRED_ARTIFACTS)


def _report_destination(loaded: LoadedRun, out: str | None) -> Path:
    if out is None:
        return loaded.run_directory / "report.html"
    try:
        destination = contained_path(loaded.root, out)
    except ConfigurationError as exc:
        raise ConfigurationError("report output path must remain inside the target") from exc
    unresolved = loaded.root.resolve() / Path(out)
    if unresolved.is_symlink():
        raise ConfigurationError("report output path must not be a symlink")
    if destination.exists() and destination.is_dir():
        raise ConfigurationError("report output path must be a file")
    return destination


def _optional_frozen_suite(
    root: Path, config: AgentCheckConfig, spec: AgentSpec
) -> FrozenSuite | None:
    try:
        configured = configured_frozen_suite(root, config)
    except ConfigurationError:
        return None
    if configured is None:
        return None
    _, frozen = configured
    if frozen.spec_id != spec.spec_id:
        return None
    return frozen


def _load_spec(path: Path) -> AgentSpec:
    raw = _read_bytes(path, max_bytes=_MAX_JSON_BYTES)
    document = _parse_json_object(raw, filename=path.name)
    _require_contract(
        document,
        field="contract_version",
        expected=AGENT_SPEC_CONTRACT_VERSION,
        filename=path.name,
    )
    try:
        return AgentSpec.model_validate_json(raw)
    except ValueError as exc:
        raise ConfigurationError(f"invalid {path.name}: {exc}") from exc


def _load_suite(path: Path, run_id: str) -> tuple[int, tuple[Scenario, ...]]:
    raw = _read_bytes(path, max_bytes=_MAX_JSON_BYTES)
    document = _parse_json_object(raw, filename=path.name)
    _require_contract(
        document, field="schema_version", expected=_SUITE_CONTRACT, filename=path.name
    )
    if document.get("run_id") != run_id:
        raise ConfigurationError(
            f"{path.name} run_id {document.get('run_id')!r} does not match {run_id!r}"
        )
    try:
        seed = int(document["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid {path.name}: seed is required") from exc
    raw_scenarios = document.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ConfigurationError(f"invalid {path.name}: scenarios must be a list")
    scenarios: list[Scenario] = []
    for index, item in enumerate(raw_scenarios):
        try:
            scenarios.append(Scenario.model_validate_json(json.dumps(item)))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"invalid {path.name} scenario at index {index}: {exc}"
            ) from exc
    return seed, tuple(scenarios)


def _load_invalid_scenario_ids(path: Path) -> tuple[str, ...] | None:
    if not path.exists():
        return None
    document = _read_json_object(path, max_bytes=_MAX_JSON_BYTES)
    _require_contract(
        document,
        field="schema_version",
        expected=_INVALID_SCENARIOS_CONTRACT,
        filename=path.name,
    )
    raw_items = document.get("items")
    if not isinstance(raw_items, list):
        raise ConfigurationError(f"invalid {path.name}: items must be a list")
    scenario_ids: list[str] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ConfigurationError(f"invalid {path.name} item {index}")
        try:
            scenario = Scenario.model_validate_json(json.dumps(item.get("scenario")))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"invalid {path.name} scenario at index {index}: {exc}"
            ) from exc
        issues = item.get("issues")
        if not isinstance(issues, list) or not issues:
            raise ConfigurationError(
                f"invalid {path.name} issues at index {index}: expected a non-empty list"
            )
        if any(
            not isinstance(issue, dict)
            or any(
                not isinstance(issue.get(field), str)
                for field in ("code", "message", "severity")
            )
            for issue in issues
        ):
            raise ConfigurationError(f"invalid {path.name} issues at index {index}")
        scenario_ids.append(scenario.scenario_id)
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ConfigurationError(f"invalid {path.name}: scenario IDs must be unique")
    return tuple(scenario_ids)


def _load_summary(
    path: Path,
    run_id: str,
    seed: int,
    *,
    spec: AgentSpec,
    invalid_scenario_ids: tuple[str, ...] | None,
    scenarios: tuple[Scenario, ...],
) -> tuple[str | None, SelectionPlan | None, BehavioralCoverage]:
    document = _read_json_object(path, max_bytes=_MAX_JSON_BYTES)
    _require_contract(
        document, field="schema_version", expected=_SUMMARY_CONTRACT, filename=path.name
    )
    if document.get("run_id") != run_id:
        raise ConfigurationError(
            f"{path.name} run_id {document.get('run_id')!r} does not match {run_id!r}"
        )
    if document.get("seed") != seed:
        raise ConfigurationError(f"{path.name} seed does not match suite.json")
    raw_invalid_count = document.get("invalid_scenarios", 0)
    if (
        isinstance(raw_invalid_count, bool)
        or not isinstance(raw_invalid_count, int)
        or raw_invalid_count < 0
    ):
        raise ConfigurationError(
            f"invalid {path.name}: invalid_scenarios must be a non-negative integer"
        )
    if invalid_scenario_ids is None:
        if raw_invalid_count:
            raise ConfigurationError(
                f"invalid {path.name}: invalid-scenarios.json is required when "
                "invalid scenarios were recorded"
            )
        persisted_invalid_ids: tuple[str, ...] = ()
    else:
        persisted_invalid_ids = invalid_scenario_ids
        if raw_invalid_count != len(persisted_invalid_ids):
            raise ConfigurationError(
                f"invalid {path.name}: invalid scenario count does not match "
                "invalid-scenarios.json"
            )
    valid_ids = {scenario.scenario_id for scenario in scenarios}
    if valid_ids.intersection(persisted_invalid_ids):
        raise ConfigurationError(
            f"invalid {path.name}: valid and invalid scenario IDs overlap"
        )
    revision = document.get("git_revision")
    selection = _optional_selection(document.get("selection"), filename=path.name)
    behavioral_coverage = _optional_behavioral_coverage(
        document.get("behavioral_coverage", _MISSING_BEHAVIORAL_COVERAGE),
        filename=path.name,
        spec=spec,
        scenarios=scenarios,
        reference_scope=(
            BehavioralCoverageReferenceScope.AVAILABLE_SCENARIOS_ONLY
            if selection is not None and selection.excluded_ids
            else BehavioralCoverageReferenceScope.COMPLETE
        ),
    )
    _validate_selection_coverage_binding(
        selection,
        behavioral_coverage,
        scenarios,
        invalid_scenario_ids=persisted_invalid_ids,
        filename=path.name,
    )
    if revision is None:
        return None, selection, behavioral_coverage
    return str(revision), selection, behavioral_coverage


def _optional_selection(value: object, *, filename: str) -> SelectionPlan | None:
    if value is None:
        return None
    try:
        return SelectionPlan.model_validate_json(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid {filename} selection plan: {exc}") from exc


def _optional_behavioral_coverage(
    value: object,
    *,
    filename: str,
    spec: AgentSpec,
    scenarios: tuple[Scenario, ...],
    reference_scope: BehavioralCoverageReferenceScope,
) -> BehavioralCoverage:
    if value is _MISSING_BEHAVIORAL_COVERAGE:
        # Behavioral coverage was added without changing the summary contract,
        # so reports created before the field existed remain readable. Derive
        # only from this run's artifacts; a currently configured frozen suite
        # may have changed since the run. When selection discarded cases before
        # persistence, explicitly retain that the available denominator is
        # incomplete instead of presenting the selected subset as the universe.
        return analyze_behavioral_coverage(
            spec, scenarios, reference_scope=reference_scope
        )
    try:
        coverage = BehavioralCoverage.model_validate_json(json.dumps(value))
        verify_behavioral_coverage_binding(coverage, spec, scenarios)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"invalid {filename} behavioral coverage: {exc}"
        ) from exc
    return coverage


def _validate_selection_coverage_binding(
    selection: SelectionPlan | None,
    coverage: BehavioralCoverage,
    scenarios: tuple[Scenario, ...],
    *,
    invalid_scenario_ids: tuple[str, ...],
    filename: str,
) -> None:
    if selection is None:
        return
    scenario_ids = tuple(scenario.scenario_id for scenario in scenarios)
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ConfigurationError(
            f"invalid {filename} selection binding: suite.json scenario IDs "
            "must be unique"
        )
    valid_ids = set(scenario_ids)
    invalid_ids = set(invalid_scenario_ids)
    selected_ids = set(selection.selected_ids)
    decision_ids = {item.scenario_id for item in selection.decisions}
    invalid_in_plan = invalid_ids.intersection(decision_ids)
    if invalid_in_plan and invalid_in_plan != invalid_ids:
        raise ConfigurationError(
            f"invalid {filename} selection binding: invalid scenario IDs are "
            "only partially represented by the selection plan"
        )
    if invalid_in_plan.intersection(selection.excluded_ids):
        raise ConfigurationError(
            f"invalid {filename} selection binding: a persisted invalid scenario "
            "cannot be excluded by the selection plan"
        )
    if selected_ids != valid_ids.union(invalid_in_plan):
        raise ConfigurationError(
            f"invalid {filename} selection binding: selected scenario IDs do not "
            "match persisted valid and invalid scenarios"
        )
    if coverage.scenario_count != len(scenario_ids):
        raise ConfigurationError(
            f"invalid {filename} selection binding: behavioral coverage scenario "
            "count does not match lint-valid scenarios"
        )
    expected_reference_count = (
        len(scenario_ids)
        if coverage.reference_scope
        is BehavioralCoverageReferenceScope.AVAILABLE_SCENARIOS_ONLY
        else len(selection.decisions) - len(invalid_in_plan)
    )
    if coverage.reference_scenario_count != expected_reference_count:
        raise ConfigurationError(
            f"invalid {filename} selection binding: behavioral coverage reference "
            "count does not match the lint-valid selection universe"
        )


def _load_findings(path: Path) -> tuple[Finding, ...]:
    raw = _read_bytes(path, max_bytes=_MAX_JSON_BYTES)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid {path.name}: {exc}") from exc
    if not isinstance(document, list):
        raise ConfigurationError(f"invalid {path.name}: findings must be a JSON array")
    findings: list[Finding] = []
    for index, item in enumerate(document):
        if not isinstance(item, dict):
            raise ConfigurationError(f"invalid {path.name} item {index}")
        _require_contract(
            item,
            field="contract_version",
            expected=FINDING_CONTRACT_VERSION,
            filename=path.name,
        )
        try:
            findings.append(Finding.model_validate_json(json.dumps(item)))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid {path.name} item {index}: {exc}") from exc
    return tuple(findings)


def _load_jsonl(
    path: Path,
    model: type[Any],
    *,
    version_field: str,
    expected: str,
    filename: str,
) -> tuple[Any, ...]:
    raw = _read_bytes(path, max_bytes=_MAX_JSONL_BYTES)
    if not raw.strip():
        return ()
    items: list[Any] = []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"invalid {filename}: {exc}") from exc
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            document = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"invalid {filename} line {index + 1}: {exc}") from exc
        if not isinstance(document, dict):
            raise ConfigurationError(f"invalid {filename} line {index + 1}")
        _require_contract(
            document, field=version_field, expected=expected, filename=filename
        )
        try:
            items.append(model.model_validate_json(line))
        except ValueError as exc:
            raise ConfigurationError(
                f"invalid {filename} line {index + 1}: {exc}"
            ) from exc
    return tuple(items)


def _require_contract(
    document: dict[str, Any], *, field: str, expected: str, filename: str
) -> None:
    declared = document.get(field)
    if declared != expected:
        raise ConfigurationError(
            f"unsupported {filename} contract {declared!r}; this build reads {expected}"
        )


def _read_json_object(path: Path, *, max_bytes: int) -> dict[str, Any]:
    return _parse_json_object(_read_bytes(path, max_bytes=max_bytes), filename=path.name)


def _parse_json_object(raw: bytes, *, filename: str) -> dict[str, Any]:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid {filename}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError(f"invalid {filename}: the document must be a JSON object")
    return document


def _read_bytes(path: Path, *, max_bytes: int) -> bytes:
    if path.is_symlink():
        raise ConfigurationError(f"{path.name} must not be a symlink")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfigurationError(f"unable to read {path.name}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ConfigurationError(f"unable to read {path.name}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > max_bytes:
        raise ConfigurationError(f"{path.name} exceeds the {max_bytes} byte artifact limit")
    return raw


__all__ = [
    "LoadedRun",
    "StoredReport",
    "load_report_config",
    "load_stored_run",
    "render_stored_run",
    "resolve_stored_run_id",
]
