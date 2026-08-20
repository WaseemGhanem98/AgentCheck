"""Append-only filesystem storage for human review records."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from agentcheck.artifacts import create_private_file
from agentcheck.config import AgentCheckConfig, contained_path
from agentcheck.errors import ConfigurationError
from agentcheck.privacy import redact_artifact, redact_log_text

from .contract import (
    HUMAN_REVIEW_CONTRACT_VERSION,
    MAX_NOTE_CHARS,
    MAX_REVIEW_BYTES,
    REVIEW_SUBDIRECTORY,
    HumanReview,
)


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$")
_SAFE_REVIEW_FILE = re.compile(r"^review-[0-9a-f]{24}\.json$")
_SQLITE_MAGIC = b"SQLite format 3"


def findings_relative_path(config: AgentCheckConfig, run_id: str) -> str:
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ConfigurationError(
            "run ID must contain only letters, digits, underscores, or hyphens"
        )
    return f"{config.artifacts_directory}/runs/{run_id}/findings.json"


def reviews_directory_relative(config: AgentCheckConfig, run_id: str) -> str:
    if _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise ConfigurationError(
            "run ID must contain only letters, digits, underscores, or hyphens"
        )
    return f"{config.artifacts_directory}/{REVIEW_SUBDIRECTORY}/{run_id}"


def hash_findings_file(path: Path) -> str:
    payload = _read_untrusted_bytes(path, 8 * 1024 * 1024)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def screen_review_note(note: str) -> str:
    """Refuse credential-shaped notes rather than storing a redacted substitute."""

    if len(note) > MAX_NOTE_CHARS:
        raise ConfigurationError(
            f"review note exceeds the {MAX_NOTE_CHARS} character bound"
        )
    if redact_log_text(note) != note or redact_artifact(note) != note:
        raise ConfigurationError("review note contains credential-shaped content")
    return note


def screen_reviewer_label(reviewer: str | None) -> str | None:
    if reviewer is None:
        return None
    label = reviewer.strip()
    if not label:
        raise ConfigurationError("reviewer must not be blank when supplied")
    if len(label) > 80:
        raise ConfigurationError("reviewer label exceeds the 80 character bound")
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{0,79}", label) is None:
        raise ConfigurationError(
            "reviewer must be an explicit label of letters, digits, dots, "
            "underscores, or hyphens"
        )
    return label


def write_human_review(
    root: Path,
    config: AgentCheckConfig,
    review: HumanReview,
) -> Path:
    relative_dir = reviews_directory_relative(config, review.run_id)
    directory = contained_path(root, relative_dir)
    filename = f"{review.review_id}.json"
    if _SAFE_REVIEW_FILE.fullmatch(filename) is None:
        raise ConfigurationError("review ID is not a safe file name")
    destination = contained_path(root, f"{relative_dir}/{filename}")
    payload = _encode_review(review)
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        create_private_file(destination, payload)
    except FileExistsError as exc:
        raise ConfigurationError(
            f"refusing to overwrite existing review {review.review_id}"
        ) from exc
    except OSError as exc:
        raise ConfigurationError(f"unable to write human review: {exc}") from exc
    return destination


def load_reviews_for_run(
    root: Path,
    config: AgentCheckConfig,
    run_id: str,
) -> tuple[HumanReview, ...]:
    relative = reviews_directory_relative(config, run_id)
    unresolved = root.resolve() / Path(relative)
    if not unresolved.exists():
        return ()
    directory = contained_path(root, relative)
    if directory.is_symlink():
        raise ConfigurationError(f"refusing to follow symlink at {directory}")
    if not directory.is_dir():
        raise ConfigurationError(f"{relative} is not a review directory")
    loaded: list[HumanReview] = []
    for name in sorted(os.listdir(directory)):
        if not _SAFE_REVIEW_FILE.fullmatch(name):
            continue
        path = directory / name
        loaded.append(load_human_review_path(path))
    loaded.sort(key=lambda item: (item.recorded_at.isoformat(), item.review_id))
    return tuple(loaded)


def load_human_review_path(path: Path) -> HumanReview:
    raw = _read_untrusted_bytes(path, MAX_REVIEW_BYTES)
    if len(raw) > MAX_REVIEW_BYTES:
        raise ConfigurationError(
            f"{path.name} exceeds the {MAX_REVIEW_BYTES} byte review limit"
        )
    if raw.startswith(_SQLITE_MAGIC):
        raise ConfigurationError(f"{path.name} is a SQLite database, not a human review")
    head = raw.lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        raise ConfigurationError(f"{path.name} is an HTML document, not a human review")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid human review {path.name}: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigurationError(f"invalid human review {path.name}")
    version = document.get("schema_version")
    if version != HUMAN_REVIEW_CONTRACT_VERSION:
        raise ConfigurationError(
            f"unsupported human review version {version!r} in {path.name}"
        )
    try:
        review = HumanReview.model_validate_json(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid human review {path.name}: {exc}") from exc
    screen_review_note(review.note)
    if review.reviewer is not None:
        screen_reviewer_label(review.reviewer)
    expected_name = f"{review.review_id}.json"
    if path.name != expected_name:
        raise ConfigurationError(
            f"human review file {path.name} does not match review id {review.review_id}"
        )
    return review


def _encode_review(review: HumanReview) -> bytes:
    payload = (
        json.dumps(
            review.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_REVIEW_BYTES:
        raise ConfigurationError(
            f"human review exceeds the {MAX_REVIEW_BYTES} byte size bound"
        )
    return payload


def _read_untrusted_bytes(path: Path, max_bytes: int) -> bytes:
    if path.is_symlink():
        raise ConfigurationError(f"refusing to follow symlink at {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfigurationError(f"unable to read {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read(max_bytes + 1)
    except OSError as exc:
        raise ConfigurationError(f"unable to read {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


__all__ = [
    "findings_relative_path",
    "hash_findings_file",
    "load_human_review_path",
    "load_reviews_for_run",
    "reviews_directory_relative",
    "screen_review_note",
    "screen_reviewer_label",
    "write_human_review",
]
