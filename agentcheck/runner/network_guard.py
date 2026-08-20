"""Deny-by-default network egress containment for AgentCheck worker processes.

Stripping ``OPENAI_API_KEY`` is not a security boundary. A target that
configures its own provider ``base_url`` (an Ollama or vLLM endpoint, a staging
host, an internal service) reaches that endpoint for real during evaluation,
because nothing below the URL layer refuses the connection. Earlier OpenAI
targets stayed offline only incidentally: the OpenAI client raises client-side
when no credential is present, before it opens a socket. A target that needs no
real credential never hits that check.

This guard closes that gap at the socket layer rather than the URL layer. Every
Python networking API in practical use -- ``urllib``, ``requests``, ``httpx``
(sync and async, the path the OpenAI Agents SDK itself takes), ``aiohttp``, and
raw ``socket`` -- ultimately calls ``socket.socket.connect`` /
``socket.create_connection``. Refusing there cannot be sidestepped by choosing a
different HTTP client or a different ``base_url``, which is precisely the
weakness of a URL allowlist.

Scope and honest limits. This is an in-process control, not an OS sandbox. It
denies the realistic failure mode for AgentCheck's stated trust model -- a
trusted-but-misconfigured local agent causing incidental network side effects
during evaluation. It does not contain deliberately hostile code, which could
issue raw syscalls through ``ctypes``, ship a C extension, or spawn a
subprocess. Containing that requires OS-level isolation (network namespaces,
seccomp), which is platform-specific and privileged; AgentCheck already
describes its worker as "process isolation, not a security sandbox".

``AF_UNIX`` *connect* is refused as well. A Unix socket cannot reach a network
peer, but it can reach a privileged local daemon -- ``/var/run/docker.sock`` is
present on a typical developer machine and grants effective host root -- which
is exactly the kind of external side effect evaluation must not cause.
``socket.socketpair()`` is untouched because it returns an already-connected
pair without calling ``connect``; asyncio's event loop and its subprocess
support were verified to keep working under this rule.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any


class NetworkAccessDenied(RuntimeError):
    """Raised when target code attempts network access during evaluation."""


_GUARD_INSTALLED = False
_DENIED_DESTINATIONS: list[str] = []
_ALLOWED_DESTINATIONS: frozenset[str] = frozenset()
# Hostnames the operator named, keyed by the port they were named for, plus the
# addresses DNS has returned for them. A hostname entry cannot be matched
# directly: the client resolves it and then connects to an address, so the
# guard only ever sees the address. These two maps are what let an address be
# traced back to the hostname that authorised it.
_ALLOWED_HOSTNAMES: dict[int, frozenset[str]] = {}
_RESOLVED_ADDRESSES: set[str] = set()
_LOOPBACK_ALIASES = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", ""})
_MAX_RECORDED_DENIALS = 16
# Provider addresses rotate, so a miss re-resolves. Bounded so a target cannot
# turn a single allowlisted hostname into an unbounded DNS side channel.
_MAX_RESOLUTIONS = 32
_resolutions_performed = 0
_real_getaddrinfo = socket.getaddrinfo

_DENIAL_HINT = (
    "AgentCheck blocks network access during evaluation so a target cannot "
    "cause external side effects. Set allow_network to true in agentcheck.json "
    "only when you intend the evaluation to reach that endpoint."
)


def denied_destinations() -> tuple[str, ...]:
    """Destinations refused so far in this process, oldest first.

    HTTP clients wrap a failed connection in their own transport error, so the
    guard's message is otherwise lost and the developer sees only something
    like "Connection error." Callers use this to say *why* the connection
    failed. Recording is diagnostic only and never widens what is allowed.
    """

    return tuple(_DENIED_DESTINATIONS)


def _canonical_host(host: Any) -> str:
    """Fold loopback spellings together so one entry covers the same endpoint.

    A client resolves ``localhost`` and then connects to ``127.0.0.1``, so an
    allowlist matched literally would have to name both. Every alias here refers
    to this machine, so folding them changes reachability by nothing.
    """

    text = _text(host).strip().strip("[]").lower()
    return "localhost" if text in _LOOPBACK_ALIASES else text


def normalize_allowlist(entries: Any) -> frozenset[str]:
    """Parse ``host:port`` entries into canonical match keys."""

    parsed: set[str] = set()
    for entry in entries or ():
        text = _text(entry).strip()
        if not text:
            continue
        host, separator, port = text.rpartition(":")
        # An empty host is rejected rather than folded to loopback: a typo must
        # not quietly allowlist an endpoint nobody named.
        if not separator or not port.isdigit() or not host.strip():
            raise ValueError(
                f"network allowlist entry {text!r} must use the form host:port"
            )
        parsed.add(f"{_canonical_host(host)}:{int(port)}")
    return frozenset(parsed)


def _is_literal_address(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _resolve_hostnames(port: int) -> None:
    """Record the addresses DNS returns for hostnames allowlisted on ``port``.

    Resolution is authorisation here, so only hostnames the operator explicitly
    named are ever resolved, and only the addresses DNS returns for that exact
    hostname and port are admitted. Reverse DNS is never consulted: an address
    is admitted because a named hostname resolves *to* it, never because it
    claims to belong to one.

    Uses the pre-patch ``getaddrinfo`` so the guard cannot recurse into itself.
    """

    global _resolutions_performed
    for hostname in _ALLOWED_HOSTNAMES.get(port, frozenset()):
        if _resolutions_performed >= _MAX_RESOLUTIONS:
            return
        _resolutions_performed += 1
        try:
            infos = _real_getaddrinfo(hostname, port)
        except Exception:
            # Unresolvable stays unreachable: no addresses recorded, so the
            # caller denies.
            continue
        for info in infos:
            address = info[4]
            if isinstance(address, tuple) and address:
                _RESOLVED_ADDRESSES.add(f"{_canonical_host(address[0])}:{port}")


def _is_allowed(host: Any, port: Any) -> bool:
    if not _ALLOWED_DESTINATIONS:
        return False
    try:
        canonical = _canonical_host(host)
        port_number = int(port)
    except (TypeError, ValueError):
        return False
    key = f"{canonical}:{port_number}"
    # Literal entries: an allowlisted IP, a loopback spelling, or the hostname
    # itself as seen by getaddrinfo before the client resolves it.
    if key in _ALLOWED_DESTINATIONS:
        return True
    if key in _RESOLVED_ADDRESSES:
        return True
    # A miss on a port that has hostname entries may simply be DNS rotation, so
    # re-resolve those hostnames and admit the address only if it is now among
    # what they return.
    if _ALLOWED_HOSTNAMES.get(port_number):
        _resolve_hostnames(port_number)
        return key in _RESOLVED_ADDRESSES
    return False


def _text(value: Any) -> str:
    """Render one address component; hostnames may arrive as bytes."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _describe(address: Any) -> str:
    """Render a destination without trusting a hostile ``__repr__``."""

    if isinstance(address, tuple) and len(address) >= 2:
        return f"{_text(address[0])}:{_text(address[1])}"[:200]
    if isinstance(address, (str, bytes)):
        return _text(address)[:200]
    return type(address).__name__


def install_network_guard(
    *, allow_network: bool, allowlist: Any = ()
) -> None:
    """Deny network egress in this process unless the target explicitly opted in.

    Installed once, before any target module is imported. Patching the methods
    on ``socket.socket`` (rather than wrapping instances) means the guard binds
    regardless of import order: Python resolves the method on the class at call
    time, so modules imported earlier are covered too.
    """

    global _GUARD_INSTALLED, _ALLOWED_DESTINATIONS, _ALLOWED_HOSTNAMES
    global _real_getaddrinfo, _resolutions_performed
    if allow_network or _GUARD_INSTALLED:
        return
    _ALLOWED_DESTINATIONS = normalize_allowlist(allowlist)
    # Capture the real resolver before patching so re-resolution cannot recurse.
    _real_getaddrinfo = socket.getaddrinfo
    _resolutions_performed = 0
    _RESOLVED_ADDRESSES.clear()
    hostnames: dict[int, set[str]] = {}
    for entry in _ALLOWED_DESTINATIONS:
        entry_host, _, entry_port = entry.rpartition(":")
        if entry_host and not _is_literal_address(entry_host):
            hostnames.setdefault(int(entry_port), set()).add(entry_host)
    _ALLOWED_HOSTNAMES = {port: frozenset(names) for port, names in hostnames.items()}
    for port in _ALLOWED_HOSTNAMES:
        _resolve_hostnames(port)
    _GUARD_INSTALLED = True

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def _deny(destination: Any) -> None:
        described = _describe(destination)
        if (
            len(_DENIED_DESTINATIONS) < _MAX_RECORDED_DENIALS
            and described not in _DENIED_DESTINATIONS
        ):
            _DENIED_DESTINATIONS.append(described)
        raise NetworkAccessDenied(f"network access to {described} was blocked. {_DENIAL_HINT}")

    def connect(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(address, tuple) and len(address) >= 2 and _is_allowed(*address[:2]):
            return real_connect(self, address, *args, **kwargs)
        _deny(address)

    def connect_ex(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(address, tuple) and len(address) >= 2 and _is_allowed(*address[:2]):
            return real_connect_ex(self, address, *args, **kwargs)
        _deny(address)

    def sendto(self: socket.socket, data: Any, *args: Any, **kwargs: Any) -> Any:
        # Connectionless UDP (including DNS queries) never calls connect().
        _deny(args[-1] if args else "datagram peer")

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        # Modules that did `from socket import create_connection` hold their own
        # reference, so the module attribute is replaced as well as the methods.
        if isinstance(address, tuple) and len(address) >= 2 and _is_allowed(*address[:2]):
            return real_create_connection(address, *args, **kwargs)
        _deny(address)

    def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        # DNS resolution is itself an outbound query and leaks the hostname even
        # when the subsequent connect() is refused. Resolving an allowlisted
        # endpoint has to work, because the client resolves before it connects.
        if _is_allowed(host, port):
            return real_getaddrinfo(host, port, *args, **kwargs)
        _deny((host, port) if port is not None else host)

    def gethostbyname(hostname: Any) -> Any:
        _deny(hostname)

    socket.socket.connect = connect  # type: ignore[assignment]
    socket.socket.connect_ex = connect_ex  # type: ignore[assignment]
    socket.socket.sendto = sendto  # type: ignore[assignment]
    socket.create_connection = create_connection
    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname


__all__ = [
    "NetworkAccessDenied",
    "denied_destinations",
    "install_network_guard",
    "normalize_allowlist",
]
