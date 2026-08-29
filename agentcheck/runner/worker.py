"""Internal JSON worker invoked once per AgentCheck inspect or scenario run."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agentcheck.adapters import (
    CustomAgentAdapter,
    FrameworkAdapter,
    OpenAIAgentsAdapter,
    PydanticAIAdapter,
    UnsupportedTargetError,
    encode_preflight_report,
)
from agentcheck.config import portable_entrypoint, resolve_entrypoint
from agentcheck.domain import (
    FaultType,
    InfrastructureError,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    ToolFixture,
)
from agentcheck.inspect import TargetLoadError, enable_contained_target_imports, load_target
from agentcheck.mcp_manifest import load_mcp_manifest
from agentcheck.privacy import redact_log_text

from .network_guard import install_network_guard
from .tool_gateway import ToolGateway
from .worker_protocol import (
    WORKER_RESPONSE_VERSION,
    WorkerExecutionInput,
    WorkerRequest,
    WorkerRuntimeConfig,
)
from .world import WorldSimulator


def _write_private_json(path: Path, value: Any) -> None:
    """Atomically replace the response while retaining owner-only permissions."""

    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _response(
    *,
    status: str,
    phase: str,
    operation: str | None = None,
    result: Any = None,
    error: InfrastructureError | None = None,
    preflight: Any | None = None,
    topology: Any | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "contract_version": WORKER_RESPONSE_VERSION,
        "status": status,
        "phase": phase,
        "worker_pid": os.getpid(),
    }
    if operation is not None:
        value["operation"] = operation
    if result is not None:
        value["result"] = result
    if error is not None:
        value["error"] = error.model_dump(mode="json")
    if preflight is not None:
        value["preflight"] = preflight
    if topology is not None:
        value["topology"] = topology
    return value


def _safe_details(details: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(value, str):
            safe[str(key)] = redact_log_text(value)[:1_000]
        elif isinstance(value, (int, float, bool)) or value is None:
            safe[str(key)] = value
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            safe[str(key)] = [redact_log_text(item)[:200] for item in value[:16]]
        else:
            safe[str(key)] = redact_log_text(str(value))[:1_000]
    return safe


def _safe_error(exc: BaseException, *, phase: str) -> InfrastructureError:
    if isinstance(exc, TargetLoadError):
        return InfrastructureError(
            code=exc.code,
            message=(redact_log_text(str(exc))[:4_000] or "AgentCheck could not load the target."),
            phase=phase,
            retryable=False,
            details=_safe_details({**exc.details, "error_type": type(exc).__name__}),
        )
    if isinstance(exc, UnsupportedTargetError):
        # Carries the same SupportIssue(s) preflight() would have reported --
        # an adapter's own inspect() raising this (rather than a bare
        # TypeError) keeps that detail (including any "did you mean the other
        # adapter" guess) intact instead of it being discarded below.
        code = exc.issues[0].code if exc.issues else "unsupported_target"
        return InfrastructureError(
            code=code,
            message=(redact_log_text(str(exc))[:4_000] or "AgentCheck worker failed."),
            phase=phase,
            retryable=False,
            details={"error_type": type(exc).__name__},
        )
    if isinstance(exc, ValidationError):
        message = f"Contract validation failed with {exc.error_count()} error(s)."
        code = "worker_execution_failed"
    else:
        try:
            message = str(exc)
        except BaseException:  # pragma: no cover - defensive against hostile __str__
            message = "Exception text was unavailable."
        code = "worker_execution_failed"
    return InfrastructureError(
        code=code,
        message=(redact_log_text(message)[:4_000] or "AgentCheck worker failed."),
        phase=phase,
        retryable=False,
        details={"error_type": type(exc).__name__},
    )


def _read_request(path: Path) -> WorkerRequest:
    try:
        payload = path.read_text(encoding="utf-8")
        raw = json.loads(payload)
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("worker request must be valid UTF-8 JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("worker request must be a JSON object")
    try:
        return WorkerRequest.model_validate_json(payload)
    except ValidationError as exc:
        raise ValueError(
            "worker request does not match the supported contract"
        ) from exc


_ADAPTERS: dict[str, type[FrameworkAdapter]] = {
    "openai_agents": OpenAIAgentsAdapter,
    "pydantic_ai": PydanticAIAdapter,
    "custom": CustomAgentAdapter,
}


def _adapter_for(config: WorkerRuntimeConfig) -> FrameworkAdapter:
    """Resolve the configured adapter.

    The only place the framework name is dispatched on. Everything downstream
    works through the FrameworkAdapter contract.
    """

    factory = _ADAPTERS.get(config.adapter)
    if factory is None:  # pragma: no cover - config narrows this
        raise ValueError(f"unsupported adapter {config.adapter!r}")
    return factory()


def _load_agent(root: Path, config: WorkerRuntimeConfig) -> tuple[Any, str]:
    if config.adapter not in _ADAPTERS:  # pragma: no cover - config narrows this
        raise ValueError(f"unsupported adapter {config.adapter!r}")
    resolve_entrypoint(root, config.entrypoint)
    # AgentCheck is already imported via the parent bootstrap. Add only the
    # configured target root after those imports are complete.
    enable_contained_target_imports(root)
    return load_target(root)


def _fault_fixtures(execution_input: WorkerExecutionInput) -> tuple[ToolFixture, ...]:
    """Turn explicit injected faults into ordinary controlled gateway fixtures."""

    status_by_fault = {
        FaultType.ERROR: SimulatedToolStatus.ERROR,
        FaultType.TIMEOUT: SimulatedToolStatus.TIMEOUT,
        FaultType.MALFORMED_RESPONSE: SimulatedToolStatus.MALFORMED,
        FaultType.EMPTY_RESPONSE: SimulatedToolStatus.EMPTY,
        FaultType.PARTIAL_RESPONSE: SimulatedToolStatus.PARTIAL,
        FaultType.STALE_RESPONSE: SimulatedToolStatus.STALE,
    }
    fixtures: list[ToolFixture] = []
    for fault in execution_input.injected_faults:
        is_error = fault.fault_type in {FaultType.ERROR, FaultType.TIMEOUT}
        fixtures.append(
            ToolFixture(
                fixture_id=f"injected-fault:{fault.fault_id}",
                tool_name=fault.tool_name,
                invocation_index=fault.invocation_index,
                priority=100,
                outcome=SimulatedToolOutcome(
                    status=status_by_fault[fault.fault_type],
                    result=fault.payload,
                    error_code=fault.fault_type.value if is_error else None,
                    error_message=(fault.message or "Injected tool fault.")
                    if is_error
                    else None,
                    latency_ms=fault.latency_ms or 0.0,
                ),
            )
        )
    return tuple(fixtures)


def _controlled_fixtures(
    execution_input: WorkerExecutionInput,
) -> tuple[ToolFixture, ...]:
    """Overlay injected faults on explicit fixtures for the same call slot."""

    faults = _fault_fixtures(execution_input)
    fault_slots = {
        (fault.tool_name, fault.invocation_index)
        for fault in execution_input.injected_faults
    }
    if len(fault_slots) != len(execution_input.injected_faults):
        raise ValueError("multiple injected faults target the same tool invocation")
    baseline = tuple(
        fixture
        for fixture in execution_input.tool_fixtures
        if (fixture.tool_name, fixture.invocation_index) not in fault_slots
    )
    return (*baseline, *faults)


def _inspect(
    root: Path, config: WorkerRuntimeConfig
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    target, source = _load_agent(root, config)
    adapter = _adapter_for(config)
    mcp_manifest = load_mcp_manifest(root)
    # A canonical target-relative locator for the same module `source` points
    # at, carrying no checkout location and no spelling of the same path.
    spec = adapter.inspect(
        target,
        source=source,
        identity_locator=portable_entrypoint(root, config.entrypoint),
        declared_tool_risk=config.tool_risk,
        mcp_manifest=mcp_manifest,
    )
    preflight = encode_preflight_report(adapter.preflight(target, mcp_manifest=mcp_manifest))
    topology = adapter.describe_topology(target, source=source)
    return spec, preflight, topology


def _run(
    root: Path,
    config: WorkerRuntimeConfig,
    execution_input: WorkerExecutionInput,
) -> Any:
    target, source = _load_agent(root, config)
    adapter = _adapter_for(config)
    mcp_manifest = load_mcp_manifest(root)
    spec = adapter.inspect(
        target,
        source=source,
        identity_locator=portable_entrypoint(root, config.entrypoint),
        declared_tool_risk=config.tool_risk,
        mcp_manifest=mcp_manifest,
    )
    tools = tuple(item.value for item in spec.tools.items)
    world = WorldSimulator(execution_input.initial_world_state)
    gateway = ToolGateway(
        tools,
        _controlled_fixtures(execution_input),
        world=world,
        budgets=execution_input.resource_budgets,
        run_id=execution_input.run_id,
    )
    prepared = adapter.prepare(
        target,
        gateway,
        world_state=gateway.world,
        source=source,
        identity_locator=portable_entrypoint(root, config.entrypoint),
        controlled_model=config.controlled_model,
        declared_tool_risk=config.tool_risk,
        mcp_manifest=mcp_manifest,
    )
    return asyncio.run(
        adapter.run(
            prepared,
            execution_input.conversation_turns,
            followup_turns=execution_input.followup_turns,
            run_id=execution_input.run_id,
            max_turns=execution_input.resource_budgets.max_model_turns,
            # The real Scenario ID can directly encode the evaluator's intent.
            # Use the transport run identity inside the child; the parent
            # validates and rebinds it only after accepting the CanonicalRun.
            scenario_id=execution_input.run_id,
            target_id=spec.spec_id,
        )
    )


def execute_request(request_path: Path, response_path: Path) -> int:
    """Execute a worker request, always attempting a structured response."""

    phase = "request"
    operation: str | None = None
    try:
        request = _read_request(request_path)
        operation = request.operation
        root = Path(request.root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("worker target root must be an existing directory")
        config = request.runtime_config
        try:
            request_path.unlink()
        except OSError as exc:
            raise ValueError(
                "worker request could not be hidden before target import"
            ) from exc
        # Deny egress before any target module is imported, so import-time
        # network calls are contained too, not only calls made during a run.
        install_network_guard(
            allow_network=config.allow_network,
            allowlist=config.network_allowlist,
        )
        previous_cwd = Path.cwd()
        os.chdir(root)
        # Restore cwd so an in-process execute_request call cannot leak the
        # target root into later parent-process path resolution.
        try:
            phase = "load"
            _write_private_json(
                response_path,
                _response(status="running", phase=phase, operation=operation),
            )
            preflight: Any | None = None
            topology: Any | None = None
            if operation == "inspect":
                result, preflight, topology = _inspect(root, config)
            else:
                phase = "scenario"
                execution_input = request.execution_input
                if execution_input is None:  # pragma: no cover - contract validator
                    raise ValueError("run operation requires execution_input")
                phase = "run"
                _write_private_json(
                    response_path,
                    _response(status="running", phase=phase, operation=operation),
                )
                result = _run(root, config, execution_input)

            _write_private_json(
                response_path,
                _response(
                    status="ok",
                    phase="complete",
                    operation=operation,
                    result=result.model_dump(mode="json"),
                    preflight=preflight,
                    topology=topology,
                ),
            )
            return 0
        finally:
            os.chdir(previous_cwd)
    except BaseException as exc:
        # KeyboardInterrupt/SystemExit from target code are worker failures too;
        # they never become agent FAIL verdicts in the parent process.
        error = _safe_error(exc, phase=phase)
        try:
            _write_private_json(
                response_path,
                _response(
                    status="error",
                    phase=phase,
                    operation=operation,
                    error=error,
                ),
            )
        except BaseException:
            pass
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 2)
    # The target imports in this process. Remove both private IPC paths from
    # Python-visible arguments before any target code can inspect them.
    sys.argv[:] = [sys.argv[0] if sys.argv else "agentcheck-worker"]
    return execute_request(arguments.request, arguments.response)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    sys.exit(main())


__all__ = ["execute_request", "main"]
