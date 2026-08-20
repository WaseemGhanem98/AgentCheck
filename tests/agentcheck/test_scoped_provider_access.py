"""A real model may reach its provider; the target still may not reach anything else.

Evaluating with a real reasoning model needs one network destination. Granting
it by turning the guard off would hand the target the whole network, so the
allowlist names exactly the endpoints the worker may open and refuses the rest
at the same chokepoint that already refuses everything today.
"""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentcheck.config import AgentCheckConfig
from agentcheck.runner.network_guard import (
    NetworkAccessDenied,
    install_network_guard,
    normalize_allowlist,
)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *_args: object) -> None:
        return


@pytest.fixture()
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


class _Guard:
    """Install the guard and restore the socket module afterwards."""

    def __init__(self, allowlist: tuple[str, ...]) -> None:
        self.allowlist = allowlist

    def __enter__(self):
        import agentcheck.runner.network_guard as guard

        self.guard = guard
        self.saved = (
            socket.socket.connect,
            socket.socket.connect_ex,
            socket.socket.sendto,
            socket.create_connection,
            socket.getaddrinfo,
            socket.gethostbyname,
        )
        self.installed = guard._GUARD_INSTALLED
        self.allowed = guard._ALLOWED_DESTINATIONS
        guard._GUARD_INSTALLED = False
        install_network_guard(allow_network=False, allowlist=self.allowlist)
        return self

    def __exit__(self, *_exc: object) -> None:
        (
            socket.socket.connect,
            socket.socket.connect_ex,
            socket.socket.sendto,
            socket.create_connection,
            socket.getaddrinfo,
            socket.gethostbyname,
        ) = self.saved
        self.guard._GUARD_INSTALLED = self.installed
        self.guard._ALLOWED_DESTINATIONS = self.allowed


def test_allowlisted_endpoint_is_reachable_and_everything_else_is_not(server) -> None:
    """The whole point: one destination opens, the rest stay closed."""

    port = server.server_address[1]

    with _Guard((f"127.0.0.1:{port}",)):
        with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
            connection.sendall(b"GET / HTTP/1.0\r\n\r\n")
            assert b"200" in connection.recv(256)

        # A different port on the same host is a different destination.
        with pytest.raises(NetworkAccessDenied):
            socket.create_connection(("127.0.0.1", port + 1), timeout=2)
        with pytest.raises(NetworkAccessDenied):
            socket.create_connection(("93.184.216.34", 80), timeout=2)
        with pytest.raises(NetworkAccessDenied):
            socket.getaddrinfo("example.com", 80)


def test_resolution_of_an_allowlisted_endpoint_is_permitted(server) -> None:
    """A client resolves before it connects, so resolution must not be refused."""

    port = server.server_address[1]

    with _Guard((f"localhost:{port}",)):
        assert socket.getaddrinfo("localhost", port)
        with pytest.raises(NetworkAccessDenied):
            socket.getaddrinfo("example.com", port)


def test_loopback_spellings_are_equivalent() -> None:
    """One entry covers the same endpoint however the client spells it."""

    assert normalize_allowlist(("localhost:11434",)) == normalize_allowlist(
        ("127.0.0.1:11434",)
    )
    assert normalize_allowlist(("LOCALHOST:11434",)) == {"localhost:11434"}
    # A different host is still a different destination.
    assert normalize_allowlist(("example.com:443",)) == {"example.com:443"}


def test_malformed_entries_are_rejected() -> None:
    for bad in ("nonsense", "host:", "host:notaport", ":80"):
        with pytest.raises(ValueError, match="host:port"):
            normalize_allowlist((bad,))


def test_empty_allowlist_denies_everything() -> None:
    """Default behaviour is unchanged: no entry means no destination."""

    assert AgentCheckConfig().network_allowlist == ()
    with _Guard(()):
        with pytest.raises(NetworkAccessDenied):
            socket.create_connection(("127.0.0.1", 9), timeout=1)


def test_config_rejects_a_malformed_allowlist() -> None:
    with pytest.raises(ValueError):
        AgentCheckConfig(network_allowlist=("not-a-destination",))


def test_scenario_wall_clock_budget_is_opt_in_and_bounded() -> None:
    """A slower real model needs a longer budget, but not an unbounded one."""

    assert AgentCheckConfig().scenario_wall_clock_seconds is None
    assert AgentCheckConfig(scenario_wall_clock_seconds=180).scenario_wall_clock_seconds == 180
    for invalid in (0, -1, 601):
        with pytest.raises(ValueError):
            AgentCheckConfig(scenario_wall_clock_seconds=invalid)


def test_raising_the_budget_rebuilds_scenarios_honestly() -> None:
    """Worker timeout and evaluated budget move together, and identity changes.

    Raising only the worker timeout would leave the evaluated budget behind and
    turn slow-but-correct behaviour into a budget failure.
    """

    from agentcheck.generate.templates import build_account_support_suite

    default = build_account_support_suite(seed=1729)
    raised = build_account_support_suite(seed=1729, wall_clock_seconds=180)

    assert default[0].resource_budgets.wall_clock_seconds == 10
    assert raised[0].resource_budgets.wall_clock_seconds == 180
    assert len(default) == len(raised)
    # A different budget is a different scenario, so the fingerprint must move.
    assert default[0].fingerprint != raised[0].fingerprint
    assert default[0].fingerprint == build_account_support_suite(seed=1729)[0].fingerprint
