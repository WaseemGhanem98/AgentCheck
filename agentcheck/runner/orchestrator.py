"""Parent-side orchestration for one isolated AgentCheck worker process.

The child contract is intentionally small and JSON-only.  Framework objects and
live callables never cross the process boundary; the worker reloads the trusted
entrypoint, replaces its tools, and returns only versioned domain contracts.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Generic, TypeVar, cast

from pydantic import ValidationError

from agentcheck.adapters.base import (
    SupportIssue,
    decode_preflight_report,
    decode_topology,
)
from agentcheck.config import (
    AgentCheckConfig,
    child_environment,
    resolve_entrypoint,
    resolve_python_executable,
)
from agentcheck.domain import (
    AgentSpec,
    CanonicalRun,
    InfrastructureError,
    Scenario,
)
from agentcheck.errors import ConfigurationError
from agentcheck.privacy import redact_log_text


WORKER_REQUEST_VERSION = "agentcheck.worker_request.v1"
WORKER_RESPONSE_VERSION = "agentcheck.worker_response.v1"
_MAX_DIAGNOSTIC_BYTES = 16 * 1024
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]

ResultT = TypeVar("ResultT")


class WorkerProcessError(RuntimeError):
    """Raised by :meth:`ProcessResult.require_value` for failed workers."""

    def __init__(self, error: InfrastructureError) -> None:
        self.error = error
        super().__init__(error.message)


@dataclass(frozen=True)
class ProcessResult(Generic[ResultT]):
    """A worker result that keeps infrastructure failure separate from agent verdicts."""

    value: ResultT | None
    infrastructure_error: InfrastructureError | None
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    worker_pid: int | None = None
    preflight_issues: tuple[SupportIssue, ...] = ()
    topology: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.infrastructure_error is None):
            raise ValueError(
                "a process result must contain exactly one of value or infrastructure_error"
            )

    @property
    def ok(self) -> bool:
        return self.infrastructure_error is None

    def require_value(self) -> ResultT:
        if self.infrastructure_error is not None:
            raise WorkerProcessError(self.infrastructure_error)
        return cast(ResultT, self.value)


def _write_private_json(path: Path, value: Any) -> None:
    """Write one JSON object with owner-only permissions."""

    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
            handle.write("\n")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


class _BoundedCapture:
    """Continuously drain a pipe while retaining only a bounded prefix."""

    def __init__(self) -> None:
        self._data = bytearray()
        self._truncated = False

    def drain(self, stream: IO[bytes]) -> None:
        try:
            while True:
                chunk = stream.read(8 * 1024)
                if not chunk:
                    return
                remaining = _MAX_DIAGNOSTIC_BYTES - len(self._data)
                if remaining > 0:
                    self._data.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self._truncated = True
        except (OSError, ValueError):
            return

    def redacted_text(self) -> str:
        decoded = bytes(self._data).decode("utf-8", errors="replace")
        safe = redact_log_text(decoded)
        if self._truncated:
            safe = f"{safe}\n...[TRUNCATED]"
        return safe


def _read_response(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _probe_worker_python(executable: Path) -> None:
    """Fail closed if the selected interpreter cannot host an AgentCheck worker."""

    probe_config = AgentCheckConfig()
    environment = child_environment(probe_config)
    environment["PYTHONPATH"] = str(_PACKAGE_ROOT)
    try:
        completed = subprocess.run(
            [
                str(executable),
                "-c",
                (
                    "import sys, agentcheck, pydantic, agents\n"
                    "print('%d.%d.%d' % sys.version_info[:3])"
                ),
            ],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except OSError as exc:
        raise ConfigurationError(
            f"python_executable could not be started: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ConfigurationError(
            "python_executable probe exceeded the 10s timeout"
        ) from exc
    if completed.returncode != 0:
        stderr = redact_log_text(
            completed.stderr.decode("utf-8", errors="replace")[:2_000]
        )
        raise ConfigurationError(
            "The selected python_executable could not import AgentCheck and its "
            "runtime dependencies (pydantic, openai-agents). Install AgentCheck "
            "into that environment (`python -m pip install -e '.[agentcheck]'`) "
            "or run AgentCheck from an environment that already contains the "
            "target's dependencies. AgentCheck does not install packages "
            "automatically. "
            f"{stderr}".strip()
        )
    version_text = completed.stdout.decode("utf-8", errors="replace").strip().splitlines()
    version_line = version_text[-1] if version_text else ""
    parts = version_line.split(".")
    try:
        version = tuple(int(part) for part in parts[:3])
    except ValueError:
        version = ()
    if len(version) < 2 or version < (3, 10):
        raise ConfigurationError(
            "python_executable must be Python 3.10 or newer "
            f"(observed {version_line or 'unknown'})"
        )


def _worker_bootstrap(package_root: Path) -> str:
    """Import AgentCheck from its package root even when cwd is the target.

    Starting the worker in the target lets cwd-relative lookups observe that
    tree instead of the AgentCheck checkout. ``python -m`` would then put the
    target first on ``sys.path``, so a target named ``agentcheck`` could
    shadow the worker. This bootstrap strips cwd from ``sys.path`` and inserts
    the package root as a literal, not an environment variable the target
    could read for its own lookups.
    """

    encoded_root = json.dumps(str(package_root), ensure_ascii=False)
    return (
        "import sys\n"
        "sys.path = [p for p in sys.path if p not in ('', '.')]\n"
        f"sys.path.insert(0, {encoded_root})\n"
        "from agentcheck.runner.worker import main\n"
        "raise SystemExit(main())\n"
    )


def _worker_command(root: Path, config: AgentCheckConfig) -> list[str]:
    executable = resolve_python_executable(root, config)
    # Compare the invoked wrapper paths, not symlink targets. Two venvs may
    # share a base interpreter binary but have different site-packages.
    if os.path.normpath(str(executable)) != os.path.normpath(sys.executable):
        _probe_worker_python(executable)
    return [
        str(executable),
        "-c",
        _worker_bootstrap(_PACKAGE_ROOT),
    ]


def _worker_environment(config: AgentCheckConfig) -> dict[str, str]:
    environment = child_environment(config)
    # Never inherit PYTHONPATH from the caller.  This controlled path only makes
    # the AgentCheck package importable when running directly from a checkout.
    environment["PYTHONPATH"] = str(_PACKAGE_ROOT)
    return environment


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name != "posix":  # pragma: no cover - Phase 1 CI is POSIX
        process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if process.poll() is None:
            process.kill()


def _kill_remaining_process_group(process: subprocess.Popen[bytes]) -> None:
    """Remove descendants that retained the worker's fresh process group."""

    if os.name != "posix":  # pragma: no cover - Phase 1 CI is POSIX
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _finish_capture(
    threads: tuple[threading.Thread, threading.Thread],
    streams: tuple[IO[bytes], IO[bytes]],
) -> None:
    for thread in threads:
        thread.join(timeout=1.0)
    for stream in streams:
        try:
            stream.close()
        except OSError:
            pass
    for thread in threads:
        thread.join(timeout=1.0)


def _infrastructure_error(
    *,
    code: str,
    message: str,
    phase: str,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> InfrastructureError:
    return InfrastructureError(
        code=code,
        message=redact_log_text(message)[:4_000] or "AgentCheck worker failed.",
        phase=phase,
        retryable=retryable,
        details=details or {},
    )


def _failed_result(
    error: InfrastructureError,
    *,
    stdout: str,
    stderr: str,
    returncode: int | None,
    timed_out: bool,
    worker_pid: int | None,
) -> ProcessResult[Any]:
    return ProcessResult(
        value=None,
        infrastructure_error=error,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        timed_out=timed_out,
        worker_pid=worker_pid,
    )


def _execute_worker(
    *,
    root: Path,
    config: AgentCheckConfig,
    operation: str,
    timeout_seconds: float,
    scenario: Scenario | None = None,
    run_id: str | None = None,
    expected_target_id: str | None = None,
) -> ProcessResult[Any]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"target root is not a directory: {root}")
    # Validate the containment boundary before starting trusted target code.
    resolve_entrypoint(root, config.entrypoint)

    request: dict[str, Any] = {
        "contract_version": WORKER_REQUEST_VERSION,
        "operation": operation,
        "root": str(root),
        "config": config.model_dump(mode="json"),
    }
    if scenario is not None:
        request["scenario"] = scenario.model_dump(mode="json")
    if run_id is not None:
        request["run_id"] = run_id

    with tempfile.TemporaryDirectory(prefix="agentcheck-worker-") as raw_directory:
        directory = Path(raw_directory)
        directory.chmod(0o700)
        request_path = directory / "request.json"
        response_path = directory / "response.json"
        _write_private_json(request_path, request)
        _write_private_json(
            response_path,
            {
                "contract_version": WORKER_RESPONSE_VERSION,
                "status": "running",
                "phase": "worker_start",
            },
        )

        try:
            process = subprocess.Popen(
                [
                    *_worker_command(root, config),
                    str(request_path),
                    str(response_path),
                ],
                # Target cwd so implicit filesystem lookups observe the
                # application tree, not the AgentCheck checkout. AgentCheck
                # itself is imported from PYTHONPATH plus the bootstrap
                # sys.path insert; cwd is stripped so a target named
                # agentcheck cannot shadow the worker at startup.
                cwd=root,
                env=_worker_environment(config),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            return _failed_result(
                _infrastructure_error(
                    code="worker_start_failed",
                    message=f"Could not start AgentCheck worker: {exc}",
                    phase="worker_start",
                ),
                stdout="",
                stderr="",
                returncode=None,
                timed_out=False,
                worker_pid=None,
            )

        if process.stdout is None or process.stderr is None:  # pragma: no cover
            _kill_process_group(process)
            process.wait()
            return _failed_result(
                _infrastructure_error(
                    code="worker_capture_failed",
                    message="AgentCheck could not capture worker diagnostics.",
                    phase="worker_start",
                ),
                stdout="",
                stderr="",
                returncode=process.returncode,
                timed_out=False,
                worker_pid=process.pid,
            )

        stdout_capture = _BoundedCapture()
        stderr_capture = _BoundedCapture()
        streams: tuple[IO[bytes], IO[bytes]] = (process.stdout, process.stderr)
        capture_threads = (
            threading.Thread(
                target=stdout_capture.drain,
                args=(process.stdout,),
                name="agentcheck-worker-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=stderr_capture.drain,
                args=(process.stderr,),
                name="agentcheck-worker-stderr",
                daemon=True,
            ),
        )
        for thread in capture_threads:
            thread.start()

        timed_out = False
        try:
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_group(process)
                returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                _kill_process_group(process)
                try:
                    process.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            raise
        finally:
            if process.poll() is None:
                _kill_process_group(process)
                try:
                    process.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            _kill_remaining_process_group(process)
            _finish_capture(capture_threads, streams)

        stdout = stdout_capture.redacted_text()
        stderr = stderr_capture.redacted_text()
        response = _read_response(response_path)
        worker_pid = process.pid

        if timed_out:
            phase = "run"
            if response is not None and isinstance(response.get("phase"), str):
                phase = response["phase"]
            return _failed_result(
                _infrastructure_error(
                    code="worker_timeout",
                    message=(
                        f"AgentCheck worker exceeded the {timeout_seconds:g}s wall-clock timeout."
                    ),
                    phase=phase,
                    retryable=True,
                    details={"timeout_seconds": timeout_seconds},
                ),
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                timed_out=True,
                worker_pid=worker_pid,
            )

        if response is None:
            return _failed_result(
                _infrastructure_error(
                    code="invalid_worker_response",
                    message="AgentCheck worker did not produce a valid JSON response.",
                    phase="worker_response",
                    details={"returncode": returncode},
                ),
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                timed_out=False,
                worker_pid=worker_pid,
            )

        response_pid = response.get("worker_pid")
        if isinstance(response_pid, int) and not isinstance(response_pid, bool):
            worker_pid = response_pid
        phase_value = response.get("phase")
        phase = (
            phase_value if isinstance(phase_value, str) and phase_value else "worker"
        )
        if response.get("contract_version") != WORKER_RESPONSE_VERSION:
            return _failed_result(
                _infrastructure_error(
                    code="unsupported_worker_response",
                    message="AgentCheck worker returned an unsupported response contract.",
                    phase=phase,
                    details={"returncode": returncode},
                ),
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                timed_out=False,
                worker_pid=worker_pid,
            )

        if response.get("status") != "ok" or returncode != 0:
            error_value = response.get("error")
            try:
                error = InfrastructureError.model_validate_json(
                    json.dumps(error_value, ensure_ascii=False, allow_nan=False)
                )
            except ValidationError:
                error = _infrastructure_error(
                    code="worker_failed",
                    message="AgentCheck worker failed without valid structured error details.",
                    phase=phase,
                    details={"returncode": returncode},
                )
            return _failed_result(
                error,
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                timed_out=False,
                worker_pid=worker_pid,
            )

        if response.get("operation") != operation:
            return _failed_result(
                _infrastructure_error(
                    code="worker_result_identity_mismatch",
                    message="AgentCheck worker response did not match the requested operation.",
                    phase="worker_response",
                    details={
                        "expected_operation": operation,
                        "observed_operation": response.get("operation"),
                    },
                ),
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                timed_out=False,
                worker_pid=worker_pid,
            )

        result = response.get("result")
        preflight_issues: tuple[SupportIssue, ...] = ()
        topology: dict[str, Any] | None = None
        try:
            encoded_result = json.dumps(
                result, ensure_ascii=False, allow_nan=False, sort_keys=True
            )
            if operation == "inspect":
                value: Any = AgentSpec.model_validate_json(encoded_result)
                if "preflight" not in response:
                    raise ValueError("inspect response requires a preflight report")
                preflight_issues = decode_preflight_report(response["preflight"]).issues
                if "topology" in response:
                    topology = decode_topology(response["topology"])
            elif operation == "run":
                value = CanonicalRun.model_validate_json(encoded_result)
            else:  # pragma: no cover - internal caller invariant
                raise ValueError(f"unsupported operation {operation!r}")
        except (ValidationError, ValueError):
            return _failed_result(
                _infrastructure_error(
                    code="invalid_worker_result",
                    message=f"AgentCheck worker returned an invalid {operation} result.",
                    phase="worker_response",
                    details={"returncode": returncode},
                ),
                stdout=stdout,
                stderr=stderr,
                returncode=returncode,
                timed_out=False,
                worker_pid=worker_pid,
            )

        if isinstance(value, CanonicalRun):
            mismatches: dict[str, Any] = {}
            if value.run_id != run_id:
                mismatches["run_id"] = {
                    "expected": run_id,
                    "observed": value.run_id,
                }
            if scenario is None or value.scenario_id != scenario.scenario_id:
                mismatches["scenario_id"] = {
                    "expected": scenario.scenario_id if scenario is not None else None,
                    "observed": value.scenario_id,
                }
            if expected_target_id is not None and value.target_id != expected_target_id:
                mismatches["target_id"] = {
                    "expected": expected_target_id,
                    "observed": value.target_id,
                }
            if mismatches:
                return _failed_result(
                    _infrastructure_error(
                        code="worker_result_identity_mismatch",
                        message="Canonical run identity did not match the requested case.",
                        phase="worker_response",
                        details={"mismatches": mismatches},
                    ),
                    stdout=stdout,
                    stderr=stderr,
                    returncode=returncode,
                    timed_out=False,
                    worker_pid=worker_pid,
                )

        return ProcessResult(
            value=value,
            infrastructure_error=None,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            timed_out=False,
            worker_pid=worker_pid,
            preflight_issues=preflight_issues,
            topology=topology,
        )


def inspect_in_subprocess(
    root: Path,
    config: AgentCheckConfig,
    *,
    timeout_seconds: float = 30.0,
) -> ProcessResult[AgentSpec]:
    """Inspect one configured target in a fresh child process."""

    return cast(
        ProcessResult[AgentSpec],
        _execute_worker(
            root=root,
            config=config,
            operation="inspect",
            timeout_seconds=timeout_seconds,
        ),
    )


def run_scenario_in_subprocess(
    root: Path,
    config: AgentCheckConfig,
    scenario: Scenario,
    run_id: str,
    *,
    expected_target_id: str | None = None,
) -> ProcessResult[CanonicalRun]:
    """Execute one scenario in a new child process with its own clean world."""

    return cast(
        ProcessResult[CanonicalRun],
        _execute_worker(
            root=root,
            config=config,
            operation="run",
            timeout_seconds=scenario.resource_budgets.wall_clock_seconds,
            scenario=scenario,
            run_id=run_id,
            expected_target_id=expected_target_id,
        ),
    )


__all__ = [
    "ProcessResult",
    "WORKER_REQUEST_VERSION",
    "WORKER_RESPONSE_VERSION",
    "WorkerProcessError",
    "inspect_in_subprocess",
    "run_scenario_in_subprocess",
]
