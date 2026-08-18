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

import socket
from typing import Any


class NetworkAccessDenied(RuntimeError):
    """Raised when target code attempts network access during evaluation."""


_GUARD_INSTALLED = False
_DENIED_DESTINATIONS: list[str] = []
_ALLOWED_DESTINATIONS: frozenset[str] = frozenset()
_LOOPBACK_ALIASES = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", ""})
_MAX_RECORDED_DENIALS = 16

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


def _is_allowed(host: Any, port: Any) -> bool:
    if not _ALLOWED_DESTINATIONS:
        return False
    try:
        key = f"{_canonical_host(host)}:{int(port)}"
    except (TypeError, ValueError):
        return False
    return key in _ALLOWED_DESTINATIONS


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

    global _GUARD_INSTALLED, _ALLOWED_DESTINATIONS
    if allow_network or _GUARD_INSTALLED:
        return
    _ALLOWED_DESTINATIONS = normalize_allowlist(allowlist)
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
