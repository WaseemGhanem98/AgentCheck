"""Shared isolation for the AgentCheck tests.

``agentcheck.runner.worker.execute_request`` installs the network guard by
patching process-global ``socket`` functions, and it deliberately never removes
them: inside a real worker process the guard must outlive the target import it
protects. Several tests call that entry point in-process, which leaves the guard
installed for the rest of the pytest worker. A later test that expects an
unguarded process then fails -- and because pytest-xdist assigns tests to
workers dynamically, whether that happens depends on scheduling rather than on
anything the tests did.

This restores the guard's process-global state around every test, so ordering
cannot decide the outcome. It is deliberately test-side: the product behaviour
is correct as it stands, and making the worker uninstall its own guard would
weaken the containment it exists to provide.
"""

from __future__ import annotations

import socket
from typing import Iterator

import pytest

import agentcheck.runner.network_guard as network_guard


@pytest.fixture(autouse=True)
def restore_network_guard_state() -> Iterator[None]:
    saved_socket = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.socket.sendto,
        socket.create_connection,
        socket.getaddrinfo,
        socket.gethostbyname,
    )
    saved_guard = (
        network_guard._GUARD_INSTALLED,
        network_guard._ALLOWED_DESTINATIONS,
        list(network_guard._DENIED_DESTINATIONS),
        dict(network_guard._ALLOWED_HOSTNAMES),
        set(network_guard._RESOLVED_ADDRESSES),
        network_guard._real_getaddrinfo,
        network_guard._resolutions_performed,
    )
    try:
        yield
    finally:
        (
            socket.socket.connect,
            socket.socket.connect_ex,
            socket.socket.sendto,
            socket.create_connection,
            socket.getaddrinfo,
            socket.gethostbyname,
        ) = saved_socket
        (
            network_guard._GUARD_INSTALLED,
            network_guard._ALLOWED_DESTINATIONS,
            denied,
            hostnames,
            resolved,
            network_guard._real_getaddrinfo,
            network_guard._resolutions_performed,
        ) = saved_guard
        network_guard._DENIED_DESTINATIONS[:] = denied
        network_guard._ALLOWED_HOSTNAMES.clear()
        network_guard._ALLOWED_HOSTNAMES.update(hostnames)
        network_guard._RESOLVED_ADDRESSES.clear()
        network_guard._RESOLVED_ADDRESSES.update(resolved)
