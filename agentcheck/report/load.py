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
from agentcheck.store import (
    StoreError,
    default_store_relative_path,
    list_runs_readonly,
    resolve_store_path,
)

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
    git_revision = _load_summary(directory / "summary.json", resolved_id, seed)
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
    )


def render_stored_run(
    target: str | os.PathLike[str],
    *,
    run_id: str | None = None,
    latest: bool = False,
    out: str | None = None,
) -> StoredReport:
    loaded = load_stored_run(target, run_id=run_id, latest=latest)
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


def _load_summary(path: Path, run_id: str, seed: int) -> str | None:
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
    revision = document.get("git_revision")
    if revision is None:
        return None
    return str(revision)


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
