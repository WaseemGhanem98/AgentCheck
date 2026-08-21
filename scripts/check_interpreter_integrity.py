"""Fail fast when the interpreter chosen for a job is not self-consistent.

A relocatable CPython records where its own ``libpython`` lives. The prebuilt
interpreters ``actions/setup-python`` downloads hard-code that location as
``/opt/hostedtoolcache/Python/<version>/x64/lib``, a path that exists only on a
GitHub-hosted VM. A self-hosted runner keeps its tool cache somewhere else, so
the loader never finds the bundled library, falls back to whatever ``libpython``
the host publishes, and produces an interpreter whose core and standard library
come from different releases.

That mismatch is invisible to most pure-Python work and fatal to anything that
touches an affected C extension, so it reached AgentCheck as SIGSEGV inside the
worker subprocesses: twenty-one behavioural failures that were one
infrastructure fault. AgentCheck refuses to report an infrastructure fault as a
product failure, and its own CI owes the suite the same guarantee -- before the
suite spends thirteen minutes producing verdicts nobody should trust.

The check has to run in a child process, not just in-process. Workers inherit a
deliberately minimal environment that drops ``LD_LIBRARY_PATH`` (see
``agentcheck.config.child_environment``), so a loader override set at the job
level would hide the fault from the only process whose health matters. The
child below is therefore spawned without loader overrides, exactly as a worker
is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# Extensions whose ABI moves between patch releases, so a core/stdlib mismatch
# shows up here as an unresolved symbol rather than as a later crash.
_PROBE_MODULES = ("ctypes", "ssl", "hashlib", "sqlite3", "socket", "decimal")

# Variables that let a caller redirect the dynamic loader. A worker never sees
# them, so neither does this check.
_LOADER_OVERRIDES = ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH")

_CHILD_PROGRAM = (
    "import json, sys\n"
    f"for name in {_PROBE_MODULES!r}:\n"
    "    __import__(name)\n"
    "print(json.dumps(list(sys.version_info[:3])))\n"
)


def _expected_version(argument: str) -> tuple[int, int]:
    parts = argument.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise SystemExit(f"expected a MAJOR.MINOR version, got {argument!r}")
    return int(parts[0]), int(parts[1])


def _child_version() -> tuple[int, int, int]:
    """Report the version a worker-shaped child actually runs."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _LOADER_OVERRIDES
    }
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{sys.executable} cannot start a working child process "
            f"(exit {completed.returncode}). This interpreter is not usable:\n"
            f"{completed.stderr.strip() or '<no diagnostics>'}"
        )
    major, minor, micro = json.loads(completed.stdout)
    return int(major), int(minor), int(micro)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_interpreter_integrity.py MAJOR.MINOR")
    expected = _expected_version(sys.argv[1])

    running = sys.version_info[:3]
    if running[:2] != expected:
        raise SystemExit(
            f"{sys.executable} reports {'.'.join(map(str, running))}, "
            f"but this job asked for {sys.argv[1]}. The interpreter is loading "
            "a standard library and a core from different installations."
        )

    child = _child_version()
    if child != running:
        raise SystemExit(
            f"{sys.executable} reports {'.'.join(map(str, running))} in process "
            f"but {'.'.join(map(str, child))} in a child that has no loader "
            "overrides. AgentCheck workers run with that same minimal "
            "environment, so this interpreter would crash them."
        )

    print(f"interpreter {'.'.join(map(str, running))} is self-consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
