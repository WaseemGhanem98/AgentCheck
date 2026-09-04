"""Qualify the existing release wheel, never a rebuild or a source install.

The release build calls this before uploading its wheel/sdist. Dependency
downloads use pip; installed checks are credential-free and network-denied.
This is trusted release-code validation, not a hostile-package sandbox.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile
from typing import Any
from urllib.parse import urldefrag


EXTRAS = ("", "openai-agents", "pydantic-ai")
SEMANTIC_CASES = (
    "withheld-no-call", "withheld-call", "absent-no-call", "absent-call",
    "prose-not-consent", "established-consent",
)
SCRIPT = Path(__file__).resolve()


def require(condition: bool, message: str) -> None:
    # These are release gates, so python -O must not disable them.
    if not condition:
        raise ValueError(message)


def artifact_hashes(dist: Path, version: str) -> dict[str, dict[str, Any]]:
    require(bool(re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]*", version)), "invalid version")
    expected = {
        f"agentcheck_ai-{version}-py3-none-any.whl",
        f"agentcheck_ai-{version}.tar.gz",
    }
    require({path.name for path in dist.iterdir()} == expected, "artifact set mismatch")
    result = {}
    for name in sorted(expected):
        path = dist / name
        require(path.is_file() and not path.is_symlink(), f"not a regular artifact: {name}")
        data = path.read_bytes()
        result[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    return result


def clean_environment(scratch: Path) -> dict[str, str]:
    # Do not inherit credentials, Python paths, proxy/index settings or pip
    # configuration. Runtime children get this same allowlist, not the host env.
    return {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "TMPDIR": str(scratch),
        "PIP_CONFIG_FILE": os.devnull,
        "NETRC": os.devnull,
        "PIP_KEYRING_PROVIDER": "disabled",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
    }


def run(command: list[str], scratch: Path) -> str:
    result = subprocess.run(
        command, cwd=scratch, env=clean_environment(scratch),
        check=True, capture_output=True, text=True, timeout=180,
    )
    return result.stdout.strip()


def deny_probe_network() -> Callable[[], None]:
    """Bootstrap tripwire for trusted probes, NOT a hostile-code sandbox.

    Runs before package imports. Native/syscall/subprocess bypasses are outside
    this Python audit coverage; the product guard is also retained afterward.
    """
    denied = False
    seen = False

    def audit(event: str, args: tuple[Any, ...]) -> None:
        nonlocal denied, seen
        if event == "agentcheck.release_probe.tripwire":
            seen = True
        if event in {
            "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg",
            "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
            "socket.getnameinfo",
        }:
            denied = True
            raise ValueError(f"release probe network attempt: {event}")

    sys.addaudithook(audit)
    sys.audit("agentcheck.release_probe.tripwire")
    require(seen, "bootstrap tripwire was not installed")
    return lambda: require(not denied, "release probe swallowed a network attempt")


def check_download(info: dict[str, Any], wheel: Path, digest: str) -> None:
    require(urldefrag(info.get("url", ""))[0] == wheel.as_uri(), "wrong installed wheel URL")
    hashes = info.get("archive_info", {}).get("hashes", {})
    require(hashes.get("sha256") == digest, "wrong installed wheel SHA-256")


def check_install_report(path: Path, wheel: Path, digest: str, version: str) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    entries = [
        item for item in report["install"]
        if re.sub(r"[-_.]+", "-", item["metadata"]["name"]).lower() == "agentcheck-ai"
    ]
    require(len(entries) == 1, "install receipt must identify exactly one AgentCheck")
    item = entries[0]
    require(item.get("is_direct") is True and item.get("requested") is True,
            "AgentCheck was not installed from the explicit wheel requirement")
    require(item["metadata"]["version"] == version, "install receipt version mismatch")
    check_download(item["download_info"], wheel, digest)


def check_identity(wheel: Path, digest: str, version: str, environment: Path) -> None:
    require(sys.flags.isolated == 1, "installed probe must use python -I")
    require(Path(sys.prefix).resolve() == environment.resolve(), "wrong Python environment")
    site = Path(sysconfig.get_path("purelib")).resolve()
    require(site.is_relative_to(environment.resolve()), "site-packages outside environment")
    # Check the spec before importing: a same-version source checkout is not
    # installed-wheel evidence, even when its code would pass every smoke.
    spec = importlib.util.find_spec("agentcheck")
    if spec is None or spec.origin is None:
        raise ValueError("AgentCheck import missing")
    origin = Path(spec.origin).resolve()
    require(origin.is_relative_to(site), "AgentCheck import shadows the installed wheel")
    dist = metadata.distribution("agentcheck-ai")
    require(dist.version == version, "distribution version mismatch")
    require(Path(str(dist.locate_file("agentcheck/__init__.py"))).resolve() == origin,
            "module and distribution origins disagree")
    metadata_files = [p for p in dist.files or () if p.name == "METADATA"]
    require(len(metadata_files) == 1, "distribution metadata missing or ambiguous")
    require(Path(str(dist.locate_file(metadata_files[0]))).resolve().is_relative_to(site),
            "distribution metadata outside environment")
    direct = dist.read_text("direct_url.json")
    if direct is None:
        raise ValueError("installed direct-wheel receipt missing")
    check_download(json.loads(direct), wheel, digest)
    import agentcheck

    require(agentcheck.__version__ == version, "package version mismatch")


def check_frameworks(extra: str) -> None:
    from agentcheck.adapters import AdapterDependencyError

    for name, module in (("openai-agents", "agents"), ("pydantic-ai", "pydantic_ai")):
        adapter_name = "openai_agents" if name == "openai-agents" else "pydantic_ai"
        adapter = __import__(f"agentcheck.adapters.{adapter_name}", fromlist=[adapter_name])
        present = importlib.util.find_spec(module) is not None
        require(present == (extra == name), f"framework missing or leaked: {module}")
        if present:
            adapter._require_sdk()
            require(adapter._supported_sdk_version(adapter._sdk_version()),
                    f"unsupported installed extra: {name}")
        else:
            try:
                adapter._require_sdk()
            except AdapterDependencyError as exc:
                require("pip install" in str(exc), f"unactionable missing extra: {name}")
            else:
                raise ValueError(f"missing extra did not refuse: {name}")


def semantic_smoke() -> list[str]:
    # Replaced only with fixtures coordinated against the frozen confirmation
    # contract. Until then qualification must fail, never silently skip semantics.
    raise ValueError("confirmation semantic fixture not yet qualified")


def probe_receipt(digest: str, version: str, extra: str, cases: list[str]) -> dict[str, Any]:
    return {"wheel_sha256": digest, "version": version, "extra": extra, "semantic_cases": cases}


def installed_probe(
    wheel: Path, digest: str, version: str, environment: Path, extra: str,
) -> dict[str, Any]:
    check_network = deny_probe_network()
    check_identity(wheel, digest, version, environment)
    from agentcheck.runner.network_guard import denied_destinations, install_network_guard

    install_network_guard(allow_network=False)
    check_frameworks(extra)
    cases = semantic_smoke()
    check_network()
    require(not denied_destinations(), "installed smoke attempted network access")
    return probe_receipt(digest, version, extra, cases)


CLI_PROBE = """
import runpy, sys
helper = runpy.run_path(sys.argv.pop(1))
check_network = helper['deny_probe_network']()
from agentcheck.runner.network_guard import denied_destinations, install_network_guard
install_network_guard(allow_network=False)
entry = sys.argv.pop(1)
try:
    runpy.run_path(entry, run_name='__main__')
finally:
    check_network()
    if denied_destinations():
        raise RuntimeError('CLI smoke attempted network access')
"""


def qualify(dist: Path, version: str, source_sha: str) -> dict[str, Any]:
    require(bool(re.fullmatch(r"[0-9a-f]{40}", source_sha)), "invalid source SHA")
    dist = dist.resolve(strict=True)
    before = artifact_hashes(dist, version)
    wheel = dist / f"agentcheck_ai-{version}-py3-none-any.whl"
    digest = before[wheel.name]["sha256"]
    with tempfile.TemporaryDirectory(prefix="agentcheck-release-qualification-") as temporary:
        scratch = Path(temporary).resolve()
        require(not scratch.is_relative_to(SCRIPT.parent.parent), "probe must leave checkout")
        for extra in EXTRAS:
            environment = scratch / (extra or "base")
            run([sys.executable, "-I", "-m", "venv", str(environment)], scratch)
            python = str(environment / "bin" / "python")
            receipt = scratch / f"{extra or 'base'}-install.json"
            requirement = f"agentcheck-ai{f'[{extra}]' if extra else ''} @ {wheel.as_uri()}#sha256={digest}"
            run([
                python, "-I", "-m", "pip", "install", "--no-cache-dir",
                "--index-url", "https://pypi.org/simple", "--only-binary=:all:",
                "--report", str(receipt), requirement,
            ], scratch)
            check_install_report(receipt, wheel, digest, version)
            run([python, "-I", "-m", "pip", "check"], scratch)
            proof = run([
                python, "-I", str(SCRIPT), "--probe", "--wheel", str(wheel),
                "--sha256", digest, "--version", version, "--environment", str(environment),
                "--extra", extra,
            ], scratch)
            require(json.loads(proof) == probe_receipt(digest, version, extra, list(SEMANTIC_CASES)),
                    "installed probe receipt incomplete or mismatched")
            cli = [python, "-I", "-c", CLI_PROBE, str(SCRIPT), str(environment / "bin" / "agentcheck")]
            require(run([*cli, "--version"], scratch) == f"agentcheck {version}",
                    "CLI version mismatch")
            run([*cli, "--help"], scratch)
            require(artifact_hashes(dist, version) == before, "artifacts changed during qualification")
    require(artifact_hashes(dist, version) == before, "artifacts changed after qualification")
    return {
        "source_sha": source_sha, "version": version, "artifacts": before,
        "environments": [extra or "base" for extra in EXTRAS],
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "qualification": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--source-sha")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wheel", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--sha256", help=argparse.SUPPRESS)
    parser.add_argument("--environment", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--extra", choices=EXTRAS, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.probe:
            require(all((args.wheel, args.sha256, args.environment)), "probe identity incomplete")
            print(json.dumps(installed_probe(
                args.wheel, args.sha256, args.version, args.environment, args.extra,
            ), sort_keys=True))
        else:
            require(args.source_sha is not None, "source SHA required")
            print(json.dumps(qualify(args.dist_dir, args.version, args.source_sha), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"FAIL: release artifact qualification: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError):
            print(exc.stdout or "", file=sys.stderr)
            print(exc.stderr or "", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
