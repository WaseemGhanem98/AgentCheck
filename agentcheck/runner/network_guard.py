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

``AF_UNIX`` sockets stay available: they are local IPC and cannot reach a
network peer, and asyncio/multiprocessing internals may rely on them.
"""

from __future__ import annotations

import socket
from typing import Any


class NetworkAccessDenied(RuntimeError):
    """Raised when target code attempts network access during evaluation."""


_GUARD_INSTALLED = False

_DENIAL_HINT = (
    "AgentCheck blocks network access during evaluation so a target cannot "
    "cause external side effects. Set allow_network to true in agentcheck.json "
    "only when you intend the evaluation to reach that endpoint."
)


def _is_local_ipc(family: int) -> bool:
    """AF_UNIX is local IPC; it cannot reach a network peer."""

    unix_family = getattr(socket, "AF_UNIX", None)
    return unix_family is not None and family == unix_family


def _describe(address: Any) -> str:
    """Render a destination without trusting a hostile ``__repr__``."""

    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    if isinstance(address, (str, bytes)):
        return str(address)[:200]
    return type(address).__name__


def install_network_guard(*, allow_network: bool) -> None:
    """Deny network egress in this process unless the target explicitly opted in.

    Installed once, before any target module is imported. Patching the methods
    on ``socket.socket`` (rather than wrapping instances) means the guard binds
    regardless of import order: Python resolves the method on the class at call
    time, so modules imported earlier are covered too.
    """

    global _GUARD_INSTALLED
    if allow_network or _GUARD_INSTALLED:
        return
    _GUARD_INSTALLED = True

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto

    def _deny(destination: Any) -> None:
        raise NetworkAccessDenied(
            f"network access to {_describe(destination)} was blocked. {_DENIAL_HINT}"
        )

    def connect(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_local_ipc(self.family):
            return real_connect(self, address, *args, **kwargs)
        _deny(address)

    def connect_ex(self: socket.socket, address: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_local_ipc(self.family):
            return real_connect_ex(self, address, *args, **kwargs)
        _deny(address)

    def sendto(self: socket.socket, data: Any, *args: Any, **kwargs: Any) -> Any:
        # Connectionless UDP (including DNS queries) never calls connect().
        if _is_local_ipc(self.family):
            return real_sendto(self, data, *args, **kwargs)
        _deny(args[-1] if args else "datagram peer")

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        # Modules that did `from socket import create_connection` hold their own
        # reference, so the module attribute is replaced as well as the methods.
        _deny(address)

    def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        # DNS resolution is itself an outbound query and leaks the hostname even
        # when the subsequent connect() is refused.
        _deny(f"{host}:{port}" if port is not None else host)

    def gethostbyname(hostname: Any) -> Any:
        _deny(hostname)

    socket.socket.connect = connect  # type: ignore[assignment]
    socket.socket.connect_ex = connect_ex  # type: ignore[assignment]
    socket.socket.sendto = sendto  # type: ignore[assignment]
    socket.create_connection = create_connection
    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname


__all__ = ["NetworkAccessDenied", "install_network_guard"]
