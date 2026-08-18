"""A hostname allowlist entry must cover the addresses that hostname resolves to.

The guard sees whatever `connect()` is given, which for a hosted provider is a
resolved address, not the hostname the operator wrote. `api.openai.com:443`
therefore never matched, and every real provider was unreachable while loopback
worked only because its spellings fold together.

Resolution is authorisation here, so these tests pin the boundary: only
addresses DNS returns for an explicitly allowlisted hostname *and port* are
admitted, and nothing else becomes reachable.

DNS is faked so the tests are deterministic and touch no live infrastructure.
"""

from __future__ import annotations

import socket

import pytest

import agentcheck.runner.network_guard as guard
from agentcheck.runner.network_guard import NetworkAccessDenied, install_network_guard


def _addrinfo(*addresses: str, port: int) -> list:
    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
        for address in addresses
    ]


class _Guard:
    """Install the guard with a fake resolver, then restore everything."""

    def __init__(self, allowlist: tuple[str, ...], resolver=None) -> None:
        self.allowlist = allowlist
        self.resolver = resolver

    def __enter__(self) -> "_Guard":
        self.saved = (
            socket.socket.connect,
            socket.socket.connect_ex,
            socket.socket.sendto,
            socket.create_connection,
            socket.getaddrinfo,
            socket.gethostbyname,
        )
        self.state = (
            guard._GUARD_INSTALLED,
            guard._ALLOWED_DESTINATIONS,
            dict(guard._ALLOWED_HOSTNAMES),
            set(guard._RESOLVED_ADDRESSES),
            guard._real_getaddrinfo,
        )
        guard._GUARD_INSTALLED = False
        if self.resolver is not None:
            # Patched before install so install-time resolution uses the fake.
            socket.getaddrinfo = self.resolver
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
        (
            guard._GUARD_INSTALLED,
            guard._ALLOWED_DESTINATIONS,
            hostnames,
            resolved,
            guard._real_getaddrinfo,
        ) = self.state
        guard._ALLOWED_HOSTNAMES = hostnames
        guard._RESOLVED_ADDRESSES.clear()
        guard._RESOLVED_ADDRESSES.update(resolved)


def test_resolved_address_of_an_allowlisted_hostname_is_admitted() -> None:
    """The regression: the operator names a hostname, the client connects to an IP."""

    def resolver(host, port, *a, **k):
        return _addrinfo("203.0.113.10", port=port)

    with _Guard(("provider.example:443",), resolver):
        assert guard._is_allowed("203.0.113.10", 443)
        # The hostname itself still matches, which is what lets resolution run.
        assert guard._is_allowed("provider.example", 443)


def test_all_addresses_behind_one_hostname_are_admitted() -> None:
    """A CDN answers with several addresses and may hand back any of them."""

    def resolver(host, port, *a, **k):
        return _addrinfo("203.0.113.10", "203.0.113.11", "2001:db8::1", port=port)

    with _Guard(("provider.example:443",), resolver):
        for address in ("203.0.113.10", "203.0.113.11", "2001:db8::1"):
            assert guard._is_allowed(address, 443), address


def test_unrelated_address_stays_blocked() -> None:
    """Only what DNS returns for the named hostname is admitted."""

    def resolver(host, port, *a, **k):
        return _addrinfo("203.0.113.10", port=port)

    with _Guard(("provider.example:443",), resolver):
        assert not guard._is_allowed("198.51.100.7", 443)
        with pytest.raises(NetworkAccessDenied):
            socket.create_connection(("198.51.100.7", 443), timeout=1)


def test_port_scoping_survives_resolution() -> None:
    """A hostname allowed on one port must not become reachable on another."""

    def resolver(host, port, *a, **k):
        return _addrinfo("203.0.113.10", port=port)

    with _Guard(("provider.example:443",), resolver):
        assert guard._is_allowed("203.0.113.10", 443)
        assert not guard._is_allowed("203.0.113.10", 80)
        assert not guard._is_allowed("provider.example", 80)


def test_rotated_address_is_admitted_only_after_dns_confirms_it() -> None:
    """Provider addresses rotate, so a miss re-resolves the named hostname."""

    current = {"address": "203.0.113.10"}
    def resolver(host, port, *a, **k):
        return _addrinfo(current["address"], port=port)

    with _Guard(("provider.example:443",), resolver):
        assert guard._is_allowed("203.0.113.10", 443)
        # DNS rotates to a new address the guard has never seen.
        current["address"] = "203.0.113.99"
        assert guard._is_allowed("203.0.113.99", 443)
        # An address DNS never returned is still refused.
        assert not guard._is_allowed("203.0.113.55", 443)


def test_unresolvable_hostname_fails_closed() -> None:
    """No answer means no address, so nothing becomes reachable."""

    def resolver(host, port, *a, **k):
        raise socket.gaierror("no such host")

    with _Guard(("broken.example:443",), resolver):
        assert not guard._is_allowed("203.0.113.10", 443)
        with pytest.raises(NetworkAccessDenied):
            socket.create_connection(("203.0.113.10", 443), timeout=1)


def test_re_resolution_is_bounded() -> None:
    """A target must not turn one hostname into an unbounded DNS side channel."""

    calls = {"n": 0}

    def resolver(host, port, *a, **k):
        calls["n"] += 1
        return _addrinfo("203.0.113.10", port=port)

    with _Guard(("provider.example:443",), resolver):
        for octet in range(60):  # every one misses the cache
            guard._is_allowed(f"198.51.100.{octet}", 443)
        assert calls["n"] <= guard._MAX_RESOLUTIONS


def test_literal_ip_entries_are_unchanged() -> None:
    """Literal entries must not need DNS at all."""

    def resolver(host, port, *a, **k):
        raise AssertionError("a literal IP entry must not trigger resolution")

    with _Guard(("203.0.113.10:443",), resolver):
        assert guard._is_allowed("203.0.113.10", 443)
        assert not guard._is_allowed("203.0.113.11", 443)
        assert not guard._is_allowed("203.0.113.10", 80)


def test_loopback_behaviour_is_unchanged() -> None:
    """The Ollama path must keep working exactly as before."""

    with _Guard(("localhost:11434",)):
        assert guard._is_allowed("localhost", 11434)
        assert guard._is_allowed("127.0.0.1", 11434)
        assert guard._is_allowed("::1", 11434)
        assert not guard._is_allowed("127.0.0.1", 11435)
        assert not guard._is_allowed("203.0.113.10", 11434)


def test_empty_allowlist_still_denies_everything() -> None:
    """Deny-by-default is untouched by any of this."""

    with _Guard(()):
        assert not guard._is_allowed("203.0.113.10", 443)
        assert not guard._is_allowed("localhost", 11434)
        with pytest.raises(NetworkAccessDenied):
            socket.create_connection(("127.0.0.1", 9), timeout=1)
