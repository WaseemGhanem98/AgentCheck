"""Adversarial proof that evaluation cannot reach the network.

These tests run a **real** HTTP server on loopback and drive the **real** worker
subprocess. Containment is asserted from the server's own request counter, not
from a mocked client, so a target that switches HTTP libraries or rewrites its
``base_url`` cannot make the assertions pass vacuously.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agentcheck.config import AgentCheckConfig
from agentcheck.runner import inspect_in_subprocess
from agentcheck.runner.network_guard import NetworkAccessDenied, install_network_guard


class _CountingHandler(BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        type(self).hits += 1
        body = b'{"object":"list","data":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_GET  # noqa: N815 - mirror GET for provider-style calls

    def log_message(self, *_args: object) -> None:
        return


@pytest.fixture()
def counting_server():
    """A real loopback HTTP server that counts every request it serves."""

    _CountingHandler.hits = 0
    server = HTTPServer(("127.0.0.1", 0), _CountingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _probe_target(root: Path, port: int) -> AgentCheckConfig:
    """A target whose import-time code attempts egress through several APIs.

    It records what happened to ``probe.json`` so the test can assert the
    attempts were refused, and never raises, so the agent still loads and the
    worker returns a normal inspect result.
    """

    (root / "agent.py").write_text(
        f'''
import json, socket, urllib.request
from pathlib import Path
from agents import Agent

PORT = {port}
outcomes = {{}}


def record(name, fn):
    try:
        fn()
        outcomes[name] = "REACHED"
    except BaseException as exc:
        outcomes[name] = type(exc).__name__


def raw_ipv4():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(("127.0.0.1", PORT))


def raw_alt_loopback():
    # 127.0.0.2 is still loopback: a naive "== 127.0.0.1" check would miss it.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(("127.0.0.2", PORT))


def raw_ipv6():
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(("::1", PORT))


def via_create_connection():
    socket.create_connection(("127.0.0.1", PORT), timeout=3)


def via_urllib():
    urllib.request.urlopen(f"http://127.0.0.1:{{PORT}}/v1/models", timeout=3)


def via_hostname():
    urllib.request.urlopen(f"http://localhost:{{PORT}}/v1/models", timeout=3)


def via_dns():
    socket.getaddrinfo("example.com", 80)


def via_public():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(("93.184.216.34", 80))


record("raw_ipv4", raw_ipv4)
record("raw_alt_loopback", raw_alt_loopback)
record("raw_ipv6", raw_ipv6)
record("create_connection", via_create_connection)
record("urllib", via_urllib)
record("hostname", via_hostname)
record("dns", via_dns)
record("public", via_public)

Path("probe.json").write_text(json.dumps(outcomes), encoding="utf-8")

agent = Agent(name="Probe", instructions="Probe.", model="gpt-4.1-mini")
'''.lstrip(),
        encoding="utf-8",
    )
    return AgentCheckConfig()


def test_evaluation_cannot_reach_a_real_localhost_server(
    tmp_path: Path, counting_server: HTTPServer
) -> None:
    """The decisive test: a real server, a real worker, zero requests served."""

    port = counting_server.server_address[1]
    config = _probe_target(tmp_path, port)

    result = inspect_in_subprocess(tmp_path, config)

    assert result.ok, result.stderr
    outcomes = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))

    # Every egress attempt was refused before it left the process.
    for name in (
        "raw_ipv4",
        "raw_alt_loopback",
        "raw_ipv6",
        "create_connection",
        "urllib",
        "hostname",
        "dns",
        "public",
    ):
        assert outcomes[name] != "REACHED", f"{name} escaped containment: {outcomes}"

    # Proof from the server's own counter, independent of what the target claims.
    assert _CountingHandler.hits == 0, "the target reached the real HTTP server"


def test_localhost_server_is_reachable_without_the_guard(
    tmp_path: Path, counting_server: HTTPServer
) -> None:
    """Control: the server really is reachable, so the test above is not vacuous.

    Without this, a broken fixture (wrong port, dead server) would make
    containment assertions pass for the wrong reason.
    """

    port = counting_server.server_address[1]
    with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
        connection.sendall(b"GET /v1/models HTTP/1.0\r\n\r\n")
        assert b"200" in connection.recv(1024)
    assert _CountingHandler.hits == 1


def test_explicit_opt_in_restores_network_access(
    tmp_path: Path, counting_server: HTTPServer
) -> None:
    """allow_network is the only way through, and it is not the default."""

    assert AgentCheckConfig().allow_network is False

    port = counting_server.server_address[1]
    _probe_target(tmp_path, port)
    config = AgentCheckConfig(allow_network=True)

    result = inspect_in_subprocess(tmp_path, config)

    assert result.ok, result.stderr
    outcomes = json.loads((tmp_path / "probe.json").read_text(encoding="utf-8"))
    assert outcomes["urllib"] == "REACHED"
    assert _CountingHandler.hits > 0


def test_guard_denies_every_client_library_in_process(
    counting_server: HTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard binds below the HTTP layer, so the client library is irrelevant.

    Installed after the libraries are imported, mirroring the worker's ordering
    and proving that patching the class covers already-imported modules.
    """

    import urllib.request

    import httpx
    import requests

    monkeypatch.setattr(
        "agentcheck.runner.network_guard._GUARD_INSTALLED", False, raising=False
    )
    port = counting_server.server_address[1]
    url = f"http://127.0.0.1:{port}/v1/models"

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    try:
        install_network_guard(allow_network=False)

        for name, call in (
            ("requests", lambda: requests.get(url, timeout=3)),
            ("httpx", lambda: httpx.get(url, timeout=3)),
            ("urllib", lambda: urllib.request.urlopen(url, timeout=3)),
            ("socket", lambda: socket.create_connection(("127.0.0.1", port), timeout=3)),
        ):
            with pytest.raises(BaseException) as caught:
                call()
            # Clients wrap the denial in their own transport error; either the
            # guard's exception or a client-level connection error is proof the
            # connection never completed. The counter below is the real check.
            assert caught.value is not None, name

        assert _CountingHandler.hits == 0
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.socket.sendto = original_sendto
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo
        socket.gethostbyname = original_gethostbyname
        monkeypatch.setattr(
            "agentcheck.runner.network_guard._GUARD_INSTALLED", False, raising=False
        )


def test_socketpair_works_but_unix_socket_connect_is_denied() -> None:
    """AF_UNIX connect can reach a privileged daemon, so it is denied too.

    /var/run/docker.sock grants effective host root on a typical developer
    machine. socketpair() returns an already-connected pair without calling
    connect(), so asyncio internals keep working.
    """

    monkey_installed = False
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    import agentcheck.runner.network_guard as guard

    previous = guard._GUARD_INSTALLED
    guard._GUARD_INSTALLED = False
    try:
        install_network_guard(allow_network=False)
        monkey_installed = True
        left, right = socket.socketpair()
        with left, right:
            left.sendall(b"local-ipc")
            assert right.recv(16) == b"local-ipc"
        with pytest.raises(NetworkAccessDenied):
            socket.create_connection(("127.0.0.1", 9), timeout=1)
        unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        with unix_socket, pytest.raises(NetworkAccessDenied):
            unix_socket.connect("/var/run/docker.sock")
    finally:
        if monkey_installed:
            socket.socket.connect = original_connect
            socket.socket.connect_ex = original_connect_ex
            socket.socket.sendto = original_sendto
            socket.create_connection = original_create_connection
            socket.getaddrinfo = original_getaddrinfo
            socket.gethostbyname = original_gethostbyname
        guard._GUARD_INSTALLED = previous


def test_denied_destinations_are_recorded_for_diagnostics(
    counting_server: HTTPServer,
) -> None:
    """A blocked connection must be explainable, not just fail.

    HTTP clients wrap a refused connection in their own transport error, so
    without this the developer sees a bare "Connection error." and cannot tell
    an unreachable provider from AgentCheck deliberately denying egress.
    """

    import agentcheck.runner.network_guard as guard

    port = counting_server.server_address[1]
    originals = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.socket.sendto,
        socket.create_connection,
        socket.getaddrinfo,
        socket.gethostbyname,
    )
    previous_installed = guard._GUARD_INSTALLED
    previous_denied = list(guard._DENIED_DESTINATIONS)
    guard._GUARD_INSTALLED = False
    guard._DENIED_DESTINATIONS.clear()
    try:
        install_network_guard(allow_network=False)
        with pytest.raises(NetworkAccessDenied):
            socket.create_connection(("127.0.0.1", port), timeout=1)
        # Hostnames reach getaddrinfo as bytes; the record must stay readable.
        with pytest.raises(NetworkAccessDenied):
            socket.getaddrinfo(b"localhost", 11434)

        recorded = guard.denied_destinations()
        assert f"127.0.0.1:{port}" in recorded
        assert "localhost:11434" in recorded
        assert not any("b'" in item for item in recorded)
        assert _CountingHandler.hits == 0
    finally:
        (
            socket.socket.connect,
            socket.socket.connect_ex,
            socket.socket.sendto,
            socket.create_connection,
            socket.getaddrinfo,
            socket.gethostbyname,
        ) = originals
        guard._GUARD_INSTALLED = previous_installed
        guard._DENIED_DESTINATIONS[:] = previous_denied


def test_recorded_denials_are_bounded() -> None:
    """Diagnostics must not grow without bound on a chatty target."""

    import agentcheck.runner.network_guard as guard

    originals = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.socket.sendto,
        socket.create_connection,
        socket.getaddrinfo,
        socket.gethostbyname,
    )
    previous_installed = guard._GUARD_INSTALLED
    previous_denied = list(guard._DENIED_DESTINATIONS)
    guard._GUARD_INSTALLED = False
    guard._DENIED_DESTINATIONS.clear()
    try:
        install_network_guard(allow_network=False)
        for index in range(guard._MAX_RECORDED_DENIALS * 3):
            with pytest.raises(NetworkAccessDenied):
                socket.create_connection((f"10.0.0.{index}", 80), timeout=1)
        assert len(guard.denied_destinations()) <= guard._MAX_RECORDED_DENIALS
    finally:
        (
            socket.socket.connect,
            socket.socket.connect_ex,
            socket.socket.sendto,
            socket.create_connection,
            socket.getaddrinfo,
            socket.gethostbyname,
        ) = originals
        guard._GUARD_INSTALLED = previous_installed
        guard._DENIED_DESTINATIONS[:] = previous_denied
