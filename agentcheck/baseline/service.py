"""Create and check evaluation baselines without executing the target."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentcheck.artifacts import create_private_file, replace_private_file
from agentcheck.config import contained_path
from agentcheck.errors import ConfigurationError
from agentcheck.identity import identity_mismatch_hint, spec_identity_matches
from agentcheck.privacy import redact_log_text
from agentcheck.report.load import load_report_config, load_stored_run

from .build import baseline_from_loaded
from .compare import compare_baselines, format_comparison, gate_exit_code
from .contract import (
    COMPARISON_CONTRACT_VERSION,
    DEFAULT_BASELINE_FILENAME,
    BaselineComparison,
    EvaluationBaseline,
)
from .load import load_baseline


@dataclass(frozen=True, slots=True)
class CreatedBaseline:
    baseline: EvaluationBaseline
    path: Path


@dataclass(frozen=True, slots=True)
class CheckedBaseline:
    comparison: BaselineComparison
    baseline: EvaluationBaseline
    current: EvaluationBaseline
    exit_code: int
    summary: str


def create_baseline(
    target: str | Path,
    *,
    run_id: str | None = None,
    latest: bool = False,
    out: str | None = None,
    force: bool = False,
) -> CreatedBaseline:
    loaded = load_stored_run(target, run_id=run_id, latest=latest)
    baseline = baseline_from_loaded(loaded)
    destination = _destination(loaded.root, out)
    _assert_no_secret_shaped_values(baseline.model_dump(mode="json"))
    payload = _encode(baseline)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if force:
            replace_private_file(destination, payload)
        else:
            create_private_file(destination, payload)
    except FileExistsError as exc:
        raise ConfigurationError(
            f"{destination} already exists; pass --force to replace it"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(f"unable to write baseline {destination}: {exc}") from exc
    return CreatedBaseline(baseline=baseline, path=destination)


def check_baseline(
    target: str | Path,
    *,
    baseline_path: str,
    run_id: str | None = None,
    latest: bool = False,
) -> CheckedBaseline:
    root, _config = load_report_config(target)
    baseline = load_baseline(root, baseline_path)
    loaded = load_stored_run(target, run_id=run_id, latest=latest)
    current = baseline_from_loaded(loaded)
    # `current.spec_id` comes from this run's stored agent-spec.json, so a
    # baseline and a run recorded under the same identity contract compare
    # directly. A baseline written before portable identity is recognized only
    # when this run reproduces that location-bound identity.
    if not spec_identity_matches(loaded.spec, baseline.spec_id):
        raise ConfigurationError(
            "baseline spec_id does not match the current run; refusing comparison"
            + identity_mismatch_hint(loaded.spec, baseline.spec_id)
        )
    comparison = compare_baselines(baseline, current)
    return CheckedBaseline(
        comparison=comparison,
        baseline=baseline,
        current=current,
        exit_code=gate_exit_code(comparison),
        summary=format_comparison(comparison),
    )


def encode_comparison(comparison: BaselineComparison) -> str:
    return json.dumps(
        comparison.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )


def _destination(root: Path, out: str | None) -> Path:
    relative = out or DEFAULT_BASELINE_FILENAME
    unresolved = root.resolve() / Path(relative)
    if unresolved.is_symlink():
        raise ConfigurationError("baseline output path must not be a symlink")
    try:
        destination = contained_path(root, relative)
    except ConfigurationError as exc:
        raise ConfigurationError(
            "baseline path must remain inside the target directory"
        ) from exc
    if destination.exists() and destination.is_dir():
        raise ConfigurationError("baseline output path must be a file")
    return destination


def _encode(baseline: EvaluationBaseline) -> bytes:
    return (
        json.dumps(
            baseline.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _assert_no_secret_shaped_values(value: object, *, path: str = "$") -> None:
    if isinstance(value, str):
        if redact_log_text(value) != value:
            raise ConfigurationError(f"credential-shaped content at {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_secret_shaped_values(item, path=f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_no_secret_shaped_values(item, path=f"{path}[{index}]")


__all__ = [
    "COMPARISON_CONTRACT_VERSION",
    "CheckedBaseline",
    "CreatedBaseline",
    "check_baseline",
    "create_baseline",
    "encode_comparison",
]
