from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from .errors import InfrastructureError
from .privacy import redact_artifact


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$")


def new_run_id(*, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{timestamp:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    return redact_artifact(value)


def _encoded_json(value: Any) -> bytes:
    return (
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


class ArtifactStore:
    """Private, local JSON/JSONL artifact writer for one immutable test run."""

    def __init__(self, target_root: Path, artifacts_directory: str, run_id: str) -> None:
        if _SAFE_RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run_id must be a safe single path component")
        target_root = target_root.resolve()
        artifact_root = (target_root / artifacts_directory).resolve()
        try:
            artifact_root.relative_to(target_root)
        except ValueError as exc:
            raise ValueError("artifact directory must remain inside the target") from exc
        runs_root = (artifact_root / "runs").resolve()
        try:
            runs_root.relative_to(target_root)
        except ValueError as exc:
            raise ValueError("artifact runs directory must remain inside the target") from exc
        self.root = runs_root / run_id
        try:
            for directory in (artifact_root, runs_root):
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                os.chmod(directory, 0o700)
            self.root.mkdir(mode=0o700, exist_ok=False)
            os.chmod(self.root, 0o700)
        except OSError as exc:
            raise InfrastructureError(f"unable to create run artifact directory: {exc}") from exc

    def path(self, filename: str) -> Path:
        if Path(filename).name != filename or not filename:
            raise ValueError("artifact filename must be a plain filename")
        return self.root / filename

    def write_json(self, filename: str, value: Any) -> Path:
        return self.write_bytes(filename, _encoded_json(value))

    def write_jsonl(self, filename: str, values: Iterable[Any]) -> Path:
        lines = [
            json.dumps(
                _json_value(value),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for value in values
        ]
        payload = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
        return self.write_bytes(filename, payload)

    def write_text(self, filename: str, value: str) -> Path:
        return self.write_bytes(filename, value.encode("utf-8"))

    def write_bytes(self, filename: str, payload: bytes) -> Path:
        destination = self.path(filename)
        temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(6)}.tmp")
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            os.chmod(destination, 0o600)
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise InfrastructureError(f"unable to write artifact {filename}: {exc}") from exc
        return destination
