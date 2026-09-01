"""Static safety contract for the review-only maintained-runtime smoke."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pathlib
import re
import subprocess
from typing import Any

import yaml


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github/workflows/maintained-runtime-smoke.yml"
INPUTS_PATH = (
    REPOSITORY_ROOT / "research/environment_containment/hosted_smoke/frozen-inputs.json"
)
CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "research/environment_containment/hosted_smoke/EVIDENCE_CONTRACT.md"
)


def _load_workflow() -> dict[Any, Any]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow


def _load_inputs() -> dict[str, Any]:
    inputs = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    assert isinstance(inputs, dict)
    return inputs


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    raw = workflow.get("on", workflow.get(True))
    assert isinstance(raw, dict)
    return {str(key): value for key, value in raw.items()}


def _smoke_script(workflow: dict[Any, Any]) -> str:
    steps = workflow["jobs"]["harmless-smoke"]["steps"]
    assert len(steps) == 1
    script = steps[0]["run"]
    assert isinstance(script, str)
    return script


def _job_env() -> dict[str, str]:
    raw = _load_workflow()["jobs"]["harmless-smoke"]["env"]
    return {str(key): str(value) for key, value in raw.items()}


def _python_block_after(script: str, marker: str) -> str:
    tail = script[script.index(marker) :]
    match = re.search(r"<<'PY'\n(.*?)\nPY", tail, re.DOTALL)
    assert match is not None, f"missing Python block after {marker}"
    return match.group(1)


def _bash_function(script: str, name: str) -> str:
    match = re.search(rf"(?ms)^{re.escape(name)}\(\) \{{\n.*?^\}}\n", script)
    assert match is not None, f"missing Bash function {name}"
    return match.group(0)


def test_workflow_is_manual_tokenless_and_actionless() -> None:
    workflow = _load_workflow()
    assert _triggers(workflow) == {"workflow_dispatch": None}
    assert workflow["permissions"] == {}
    assert set(workflow["jobs"]) == {"harmless-smoke"}

    job = workflow["jobs"]["harmless-smoke"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert "permissions" not in job
    assert "environment" not in job
    assert 1 <= job["timeout-minutes"] <= 30
    assert all("uses" not in step for step in job["steps"])

    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "${{",
        "GITHUB_TOKEN",
        "secrets.",
        "github.token",
        "id-token",
        "pull_request_target",
        "repository_dispatch",
        "schedule:",
        "actions/checkout",
        "actions/upload-artifact",
    ):
        assert forbidden not in text
    assert re.search(r"\blatest\b", text, re.IGNORECASE) is None


def test_workflow_matches_every_frozen_input() -> None:
    workflow = _load_workflow()
    inputs = _load_inputs()
    env = workflow["jobs"]["harmless-smoke"]["env"]

    assert inputs["status"] == "review-only-unexecuted-candidate"
    assert inputs["agentcheck_base_commit"] == (
        "124cbc59232e5532130dd93a92856aec11d1d1d4"
    )
    assert inputs["agentcheck_base_tree"] == (
        "4089365742529b1cd41dae4a215a08540fcb5e56"
    )
    assert inputs["runner"] == {
        "label": "ubuntu-24.04",
        "os": "Linux",
        "architecture": "X64",
        "image_os": "ubuntu24",
    }

    docker = inputs["docker"]
    assert env["DOCKER_KEY_URL"] == docker["signing_key"]["url"]
    assert env["DOCKER_KEY_SHA256"] == docker["signing_key"]["sha256"]
    assert (
        env["DOCKER_KEY_PRIMARY_FINGERPRINT"]
        == (docker["signing_key"]["primary_fingerprint"])
    )
    assert (
        env["DOCKER_KEY_SIGNING_FINGERPRINT"]
        == (docker["signing_key"]["signing_subkey_fingerprint"])
    )
    metadata = docker["signed_metadata"]
    assert env["DOCKER_INRELEASE_URL"] == metadata["inrelease_url"]
    assert "DOCKER_INRELEASE_SHA256" not in env
    assert "inrelease_sha256" not in metadata
    assert env["DOCKER_RELEASE_SUITE"] == metadata["suite"] == "noble"
    assert env["DOCKER_RELEASE_ARCHITECTURE"] == metadata["architecture"] == "amd64"
    assert env["DOCKER_PACKAGES_PATH"] == metadata["packages_path"]
    assert env["DOCKER_PACKAGES_URL"] == metadata["packages_url"]
    assert env["DOCKER_PACKAGES_SHA256"] == metadata["packages_sha256"]
    assert int(env["DOCKER_PACKAGES_SIZE"]) == metadata["packages_size"]

    package_env_names = {
        "containerd.io": (
            "CONTAINERD_VERSION",
            "CONTAINERD_URL",
            "CONTAINERD_SHA256",
            "CONTAINERD_SIZE",
        ),
        "docker-ce-cli": (
            "DOCKER_CLI_VERSION",
            "DOCKER_CLI_URL",
            "DOCKER_CLI_SHA256",
            "DOCKER_CLI_SIZE",
        ),
        "docker-ce": (
            "DOCKER_CE_VERSION",
            "DOCKER_CE_URL",
            "DOCKER_CE_SHA256",
            "DOCKER_CE_SIZE",
        ),
    }
    assert {package["package"] for package in docker["packages"]} == set(
        package_env_names
    )
    for package in docker["packages"]:
        version_name, url_name, sha_name, size_name = package_env_names[
            package["package"]
        ]
        assert env[version_name] == package["version"]
        assert env[url_name] == package["url"]
        assert env[sha_name] == package["sha256"]
        assert int(env[size_name]) == package["size"]
        assert package["architecture"] == "amd64"
    assert env["REQUIRED_DOCKER_VERSION"] == docker["required_client_version"]
    assert docker["required_client_version"] == docker["required_server_version"]

    gvisor = inputs["gvisor"]
    assert env["GVISOR_RELEASE"] == gvisor["release"]
    assert env["GVISOR_COMMIT"] == gvisor["commit"]
    assert env["GVISOR_ARCHIVE_URL"] == gvisor["archive_url"]
    assert int(env["GVISOR_ARCHIVE_SIZE"]) == gvisor["archive_size"]
    assert env["GVISOR_ARCHIVE_SHA256"] == gvisor["archive_sha256"]
    assert env["GVISOR_ARCHIVE_SHA512"] == gvisor["archive_sha512"]
    assert env["GVISOR_CHECKSUM_URL"] == gvisor["checksum_url"]
    assert env["GVISOR_CHECKSUM_SHA256"] == gvisor["checksum_file_sha256"]
    assert env["RUNTIME_NAME"] == gvisor["runtime_name"]

    oci = inputs["oci_image"]
    assert env["OCI_IMAGE"] == oci["reference"]
    assert env["OCI_MANIFEST_DIGEST"] == oci["manifest_digest"]
    assert env["OCI_CONFIG_DIGEST"] == oci["config_digest"]
    assert oci["os"] == "linux"
    assert oci["architecture"] == "amd64"
    assert env["CONTAINER_USER"] == oci["execution_user"] == "65532:65532"
    assert env["CONTAINER_TMPFS"] == oci["private_tmpfs"]


def test_script_is_syntactically_valid_and_has_no_fallback_path() -> None:
    workflow = _load_workflow()
    script = _smoke_script(workflow)
    result = subprocess.run(
        ["bash", "-n"],
        input=script,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    python_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script, re.DOTALL)
    assert len(python_blocks) == 7
    for index, block in enumerate(python_blocks, start=1):
        compile(block, f"maintained-runtime-smoke-heredoc-{index}.py", "exec")
    assert script.startswith("set -Eeuo pipefail\n")
    assert "curl --retry" not in script
    assert "|| true" not in script
    assert "set +u" not in script
    assert "set +o pipefail" not in script
    assert re.search(r"\brunsc\s+do\b", script) is None
    assert re.search(r"--runtime(?:=|\s+)runc\b", script) is None
    assert "agentcheck gate" not in script
    assert "spikes/environment_containment/targets" not in script
    assert "--download-sidecars=NEVER" in script
    assert "--require-sidecars=ALWAYS" in script
    assert "--clobber=false" in script
    assert "-- --platform=systrap" in script
    assert '--runtime "$RUNTIME_NAME"' in script
    assert 'docker pull --platform linux/amd64 "$OCI_IMAGE"' in script
    assert "--network none" in script
    assert "--read-only" in script
    assert '--user "$CONTAINER_USER"' in script
    assert "--cap-drop ALL" in script
    assert "--security-opt no-new-privileges" in script
    assert '--tmpfs "$CONTAINER_TMPFS"' in script
    assert "os.getuid(), os.getgid()" in script
    assert 'tempfile.mkstemp(dir="/tmp")' in script
    assert "--mount" not in script
    assert "--volume" not in script
    assert "dmesg" not in script
    assert "DOCKER_INRELEASE_SHA256" not in script
    assert script.index("gpgv --keyring") < script.index('release_values="$(python3')
    assert "docker_signed_packages_sha256_expected" in script
    assert "docker_signed_packages_sha256_actual" in script
    assert "docker_packages_index_sha256_expected" not in script
    assert '"${evidence_key}_sha256_expected"' in script
    assert '"${evidence_key}_sha256_actual"' in script


def test_complete_gvisor_archive_and_host_identity_are_fail_closed() -> None:
    inputs = _load_inputs()
    script = _smoke_script(_load_workflow())
    members = inputs["gvisor"]["members"]
    expected_paths = {
        "containerd-shim-runsc-v1",
        "runsc",
        "gvisor-bin/checkpointgofer",
        "gvisor-bin/gvisor-sentry-prewarmer",
        "gvisor-bin/gvisor_sentry",
        "gvisor-bin/runsc-metric-server",
    }
    assert {member["path"] for member in members} == expected_paths
    assert len({member["sha512"] for member in members}) == len(expected_paths)
    for member in members:
        assert re.fullmatch(r"[0-9a-f]{128}", member["sha512"])
        assert member["path"] in script
        assert member["sha512"] in script

    for required in (
        "Docker runtime registration mismatch",
        "registered runtime absent from Docker info",
        "Docker reports an unexpected container runtime",
        "container host PID does not execute a frozen gVisor binary",
        "live host-side gVisor executable hash mismatch",
        "missing live host-side gVisor process evidence",
        "gVisor process remained after container removal",
    ):
        assert required in script


def test_evidence_and_cleanup_contract_are_complete_and_bounded() -> None:
    script = _smoke_script(_load_workflow())
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    normalized_contract = " ".join(contract.split())
    required_evidence = {
        "runner_image_os",
        "runner_image_version",
        "docker_initial_client",
        "docker_initial_server",
        "docker_inrelease_signature",
        "docker_inrelease_sha256_actual",
        "docker_release_suite_actual",
        "docker_release_architecture_actual",
        "docker_signed_packages_sha256_actual",
        "docker_signed_packages_size_actual",
        "docker_final_client",
        "docker_final_server",
        "gvisor_release",
        "gvisor_commit",
        "gvisor_sidecar_policy",
        "docker_runtime_name",
        "docker_runtime_path",
        "docker_runtime_platform",
        "daemon_config_before_state",
        "daemon_backup_before_state",
        "oci_manifest_digest",
        "oci_config_digest",
        "oci_platform",
        "container_runtime",
        "container_host_pid",
        "container_host_executable",
        "container_host_executable_sha512",
        "live_gvisor_processes",
        "container_user",
        "container_cap_drop",
        "container_security_opt",
        "container_tmpfs",
        "container_exit",
        "container_removed",
        "post_container_gvisor_processes",
        "sentinel_before_sha256",
        "sentinel_after_sha256",
        "sentinel_unchanged",
        "cleanup_container_absent",
        "cleanup_residual_gvisor_processes",
        "cleanup_runtime_absent",
        "cleanup_daemon_config_restored",
        "cleanup_daemon_backup_restored",
        "cleanup_image_absent",
        "cleanup_gvisor_files_absent",
        "cleanup_work_dir_absent",
        "cleanup_sentinel_unchanged",
        "smoke_original_exit",
        "smoke_cleanup_exit",
    }
    emitted = set(re.findall(r"^\s*emit\s+([a-z0-9_]+)\b", script, re.MULTILINE))
    assert required_evidence <= emitted

    assert "trap cleanup EXIT" in script
    assert "docker container rm --force" in script
    assert "runsc uninstall --runtime" in script
    assert "docker image rm" in script
    assert "count_gvisor_processes" in script
    assert 'rm -rf -- "$WORK_DIR"' in script
    assert "cleanup_work_dir_absent" in script
    assert "host sentinel changed" in script
    assert "original_rc" in script and "cleanup_rc" in script

    for limitation in (
        "UNEXECUTED REVIEW CANDIDATE",
        "does not establish containment",
        "does not run AgentCheck",
        "zero mutation outside the one exact sentinel",
        "whole `InRelease` file is deliberately not byte-pinned",
        "package-index and package bytes still match their frozen hashes and sizes",
        "token-unreferenced workflow",
        "In-container `dmesg` is neither invoked nor accepted",
        "all hosted/runtime outcomes are **NOT PROVEN**",
    ):
        assert limitation in normalized_contract


def test_preexisting_image_failure_receipt_is_false_and_job_stays_failed(
    tmp_path: pathlib.Path,
) -> None:
    script = _smoke_script(_load_workflow())
    prefix, separator, _ = script.partition("trap cleanup EXIT\n")
    assert separator

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ "$1 $2" == "container inspect" ]]; then
  exit 1
fi
if [[ "$1" == "info" ]]; then
  printf '%s\\n' '{"runc":{}}'
  exit 0
fi
if [[ "$1 $2" == "image inspect" ]]; then
  exit 0
fi
printf 'unexpected fake docker invocation: %s\\n' "$*" >&2
exit 97
""",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    harness = (
        prefix
        + separator
        + "count_gvisor_processes() { printf '0\\n'; }\n"
        + 'if docker image inspect "$OCI_IMAGE" >/dev/null 2>&1; then\n'
        + '  fail "frozen OCI image unexpectedly exists before the smoke"\n'
        + "fi\n"
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_RUN_ID": "1",
            "GITHUB_RUN_ATTEMPT": "1",
            "RUNTIME_NAME": "ac-smoke-runsc",
            "OCI_IMAGE": "docker.io/library/python@sha256:" + "a" * 64,
        }
    )
    result = subprocess.run(
        ["bash"],
        input=harness,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "AC_SMOKE_EVIDENCE_V1\tcleanup_image_absent\tfalse" in result.stdout
    assert "AC_SMOKE_EVIDENCE_V1\tsmoke_original_exit\t1" in result.stdout
    assert "AC_SMOKE_EVIDENCE_V1\tsmoke_cleanup_exit\t0" in result.stdout
    assert not list(tmp_path.glob("ac-maintained-runtime-smoke.*"))


def test_daemon_state_restore_helper_preserves_prior_or_absent_state(
    tmp_path: pathlib.Path,
) -> None:
    script = _smoke_script(_load_workflow())
    state_function = _bash_function(script, "daemon_path_state")
    restore_function = _bash_function(script, "restore_daemon_path")
    config = tmp_path / "daemon.json"
    snapshot = tmp_path / "daemon.json.before"
    backup = tmp_path / "daemon.json~"
    snapshot.write_text('{"prior":true}\n', encoding="utf-8")
    snapshot.chmod(0o640)
    config.write_text('{"mutated":true}\n', encoding="utf-8")
    backup.write_text('{"run-created":true}\n', encoding="utf-8")

    harness = f"""set -Eeuo pipefail
sudo() {{ "$@"; }}
{state_function}
{restore_function}
expected="$(daemon_path_state {snapshot})"
restore_daemon_path {config} {snapshot} "$expected"
[[ "$(daemon_path_state {config})" == "$expected" ]]
restore_daemon_path {backup} {tmp_path / "unused"} absent
[[ ! -e {backup} ]]
"""
    result = subprocess.run(
        ["bash"],
        input=harness,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    cleanup = _bash_function(script, "cleanup")
    assert cleanup.index("runsc uninstall --runtime") < cleanup.index(
        "restore_daemon_path"
    )
    assert "$DOCKER_DAEMON_BACKUP" in cleanup
    assert "daemon_config_restored=true" in cleanup
    assert "daemon_backup_restored=true" in cleanup
    for required in (
        'daemon_config_before_state="$(daemon_path_state',
        'daemon_backup_before_state="$(daemon_path_state',
        '"$WORK_DIR/daemon.json.before"',
        '"$WORK_DIR/daemon.json-backup.before"',
        "daemon_state_captured=1",
    ):
        assert required in script


def _synthetic_inrelease(
    digest: str,
    size: int,
    *,
    target_count: int = 1,
    unrelated: str = "unrelated-v1",
) -> str:
    target = " stable/binary-amd64/Packages.gz"
    relationships = [f" {digest} {size}{target}"] * target_count
    return "\n".join(
        [
            "-----BEGIN PGP SIGNED MESSAGE-----",
            "Hash: SHA512",
            "",
            "Architectures: amd64 arm64",
            f"Date: {unrelated}",
            "Suite: noble",
            "SHA256:",
            *relationships,
            f" {'f' * 64} 9 test/binary-arm64/Packages.gz",
            "-----BEGIN PGP SIGNATURE-----",
            "inert-test-signature",
            "-----END PGP SIGNATURE-----",
            "",
        ]
    )


def _metadata_gate_result(
    tmp_path: pathlib.Path,
    *,
    inrelease: str,
    packages: bytes = b"frozen-index",
    expected_digest: str | None = None,
    expected_size: int | None = None,
    key_sha256: str | None = None,
    primary_fingerprint: str | None = None,
    signature_valid: bool = True,
) -> subprocess.CompletedProcess[str]:
    work_dir = tmp_path / "work"
    fake_bin = tmp_path / "bin"
    work_dir.mkdir(parents=True)
    fake_bin.mkdir()
    key = work_dir / "docker.asc"
    release = work_dir / "InRelease"
    index = work_dir / "Packages.gz"
    key.write_bytes(b"frozen-key")
    release.write_text(inrelease, encoding="utf-8")
    index.write_bytes(packages)

    fake_gpg = fake_bin / "gpg"
    fake_gpg.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
if [[ " $* " == *" --show-keys "* ]]; then
  printf 'fpr:::::::::%s:\\n' "$FAKE_PRIMARY_FINGERPRINT"
  printf 'fpr:::::::::%s:\\n' "$FAKE_SIGNING_FINGERPRINT"
  exit 0
fi
output=""
while (($#)); do
  if [[ "$1" == "--output" ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done
[[ -n "$output" ]]
printf 'test-keyring\\n' > "$output"
""",
        encoding="utf-8",
    )
    fake_gpg.chmod(0o755)
    fake_gpgv = fake_bin / "gpgv"
    fake_gpgv.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
printf 'called\\n' >> "$FAKE_GPGV_LOG"
[[ "$FAKE_SIGNATURE_VALID" == "1" ]]
""",
        encoding="utf-8",
    )
    fake_gpgv.chmod(0o755)

    script = _smoke_script(_load_workflow())
    harness = "\n".join(
        [
            "set -Eeuo pipefail",
            'emit() { printf \'TEST_RECEIPT\\t%s\\t%s\\n\' "$1" "$2"; }',
            "fail() { printf 'FAIL: %s\\n' \"$*\" >&2; return 1; }",
            _bash_function(script, "require_sha256"),
            _bash_function(script, "require_size"),
            _bash_function(script, "verify_docker_repository_metadata"),
            f"verify_docker_repository_metadata {key} {release} {index}",
        ]
    )
    env = os.environ.copy()
    env.update(_job_env())
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "WORK_DIR": str(work_dir),
            "DOCKER_KEY_SHA256": key_sha256
            or hashlib.sha256(key.read_bytes()).hexdigest(),
            "DOCKER_PACKAGES_SHA256": expected_digest
            or hashlib.sha256(packages).hexdigest(),
            "DOCKER_PACKAGES_SIZE": str(
                len(packages) if expected_size is None else expected_size
            ),
            "FAKE_PRIMARY_FINGERPRINT": primary_fingerprint
            or env["DOCKER_KEY_PRIMARY_FINGERPRINT"],
            "FAKE_SIGNING_FINGERPRINT": env["DOCKER_KEY_SIGNING_FINGERPRINT"],
            "FAKE_SIGNATURE_VALID": "1" if signature_valid else "0",
            "FAKE_GPGV_LOG": str(tmp_path / "gpgv.log"),
        }
    )
    return subprocess.run(
        ["bash"],
        input=harness,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_docker_source_gate_rejects_bad_key_fingerprint_and_signature(
    tmp_path: pathlib.Path,
) -> None:
    packages = b"frozen-index"
    digest = hashlib.sha256(packages).hexdigest()
    release = _synthetic_inrelease(digest, len(packages))

    bad_key = _metadata_gate_result(
        tmp_path / "bad-key", inrelease=release, key_sha256="0" * 64
    )
    assert bad_key.returncode == 1
    assert "docker_key_sha256_expected" in bad_key.stdout
    assert "docker_key_sha256_actual" in bad_key.stdout
    assert not (tmp_path / "bad-key/gpgv.log").exists()

    bad_fingerprint = _metadata_gate_result(
        tmp_path / "bad-fingerprint",
        inrelease=release,
        primary_fingerprint="1" * 40,
    )
    assert bad_fingerprint.returncode == 1
    assert "docker_key_primary_fingerprint_expected" in bad_fingerprint.stdout
    assert "docker_key_primary_fingerprint_actual" in bad_fingerprint.stdout
    assert not (tmp_path / "bad-fingerprint/gpgv.log").exists()

    bad_signature = _metadata_gate_result(
        tmp_path / "bad-signature", inrelease=release, signature_valid=False
    )
    assert bad_signature.returncode == 1
    assert (tmp_path / "bad-signature/gpgv.log").read_text() == "called\n"
    assert "docker_release_suite_actual" not in bad_signature.stdout


def test_docker_source_gate_rejects_signed_relation_and_index_mismatches(
    tmp_path: pathlib.Path,
) -> None:
    packages = b"frozen-index"
    digest = hashlib.sha256(packages).hexdigest()

    wrong_relation = _metadata_gate_result(
        tmp_path / "wrong-relation",
        inrelease=_synthetic_inrelease("e" * 64, len(packages)),
        packages=packages,
    )
    assert wrong_relation.returncode == 1
    assert "docker_signed_packages_sha256_expected" in wrong_relation.stdout
    assert "docker_signed_packages_sha256_actual" in wrong_relation.stdout

    wrong_signed_size = _metadata_gate_result(
        tmp_path / "wrong-signed-size",
        inrelease=_synthetic_inrelease(digest, len(packages) + 1),
        packages=packages,
    )
    assert wrong_signed_size.returncode == 1
    assert "docker_signed_packages_size_expected" in wrong_signed_size.stdout
    assert "docker_signed_packages_size_actual" in wrong_signed_size.stdout

    wrong_index_hash = _metadata_gate_result(
        tmp_path / "wrong-index-hash",
        inrelease=_synthetic_inrelease(digest, len(packages)),
        packages=b"changed-index",
        expected_digest=digest,
        expected_size=len(packages),
    )
    assert wrong_index_hash.returncode == 1
    assert "docker_packages_index_sha256_expected" in wrong_index_hash.stdout
    assert "docker_packages_index_sha256_actual" in wrong_index_hash.stdout

    wrong_index_size = _metadata_gate_result(
        tmp_path / "wrong-index-size",
        inrelease=_synthetic_inrelease(digest, len(packages) + 1),
        packages=packages,
        expected_digest=digest,
        expected_size=len(packages) + 1,
    )
    assert wrong_index_size.returncode == 1
    assert "docker_packages_index_size_expected" in wrong_index_size.stdout
    assert "docker_packages_index_size_actual" in wrong_index_size.stdout


def test_unrelated_signed_inrelease_drift_passes_exact_source_gate(
    tmp_path: pathlib.Path,
) -> None:
    packages = b"frozen-index"
    digest = hashlib.sha256(packages).hexdigest()
    first = _metadata_gate_result(
        tmp_path / "first",
        inrelease=_synthetic_inrelease(digest, len(packages), unrelated="date-v1"),
        packages=packages,
    )
    second = _metadata_gate_result(
        tmp_path / "second",
        inrelease=_synthetic_inrelease(digest, len(packages), unrelated="date-v2"),
        packages=packages,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    first_inrelease = (tmp_path / "first/work/InRelease").read_bytes()
    second_inrelease = (tmp_path / "second/work/InRelease").read_bytes()
    assert (
        hashlib.sha256(first_inrelease).digest()
        != hashlib.sha256(second_inrelease).digest()
    )


def _package_stanza(package: dict[str, Any]) -> str:
    filename = package["url"].removeprefix("https://download.docker.com/linux/ubuntu/")
    return "\n".join(
        [
            f"Package: {package['package']}",
            f"Version: {package['version']}",
            f"Architecture: {package['architecture']}",
            f"Filename: {filename}",
            f"Size: {package['size']}",
            f"SHA256: {package['sha256']}",
        ]
    )


def _run_package_stanza_gate(
    tmp_path: pathlib.Path, paragraphs: list[str]
) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir(parents=True)
    index = tmp_path / "Packages.gz"
    index.write_bytes(gzip.compress(("\n\n".join(paragraphs) + "\n").encode()))
    block = _python_block_after(
        _smoke_script(_load_workflow()), "# DOCKER_PACKAGES_STANZA_GATE"
    )
    env = os.environ.copy()
    env.update(_job_env())
    return subprocess.run(
        ["python3", "-", str(index)],
        input=block,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_signed_package_stanzas_reject_duplicate_and_missing_entries(
    tmp_path: pathlib.Path,
) -> None:
    paragraphs = [
        _package_stanza(package) for package in _load_inputs()["docker"]["packages"]
    ]
    duplicate = _run_package_stanza_gate(
        tmp_path / "duplicate", [*paragraphs, paragraphs[0]]
    )
    assert duplicate.returncode == 1
    assert "duplicate package stanza" in duplicate.stderr

    missing = _run_package_stanza_gate(tmp_path / "missing", paragraphs[1:])
    assert missing.returncode == 1
    assert "frozen package set mismatch" in missing.stderr


def test_package_bytes_and_control_fields_fail_closed(tmp_path: pathlib.Path) -> None:
    script = _smoke_script(_load_workflow())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_dpkg_deb = fake_bin / "dpkg-deb"
    fake_dpkg_deb.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
case "$3" in
  Package) printf '%s\\n' "$FAKE_PACKAGE" ;;
  Version) printf '%s\\n' "$FAKE_VERSION" ;;
  Architecture) printf '%s\\n' "$FAKE_ARCHITECTURE" ;;
  *) exit 97 ;;
esac
""",
        encoding="utf-8",
    )
    fake_dpkg_deb.chmod(0o755)
    package_path = tmp_path / "package.deb"
    package_path.write_bytes(b"exact-package-bytes")
    actual_sha = hashlib.sha256(package_path.read_bytes()).hexdigest()
    functions = "\n".join(
        [
            'emit() { printf \'TEST_RECEIPT\\t%s\\t%s\\n\' "$1" "$2"; }',
            "fail() { printf 'FAIL: %s\\n' \"$*\" >&2; return 1; }",
            _bash_function(script, "require_sha256"),
            _bash_function(script, "require_size"),
            _bash_function(script, "verify_deb"),
        ]
    )
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_PACKAGE": "containerd.io",
            "FAKE_VERSION": "expected-version",
            "FAKE_ARCHITECTURE": "amd64",
        }
    )
    bad_bytes = subprocess.run(
        ["bash"],
        input=(
            "set -Eeuo pipefail\n"
            + functions
            + f"\nverify_deb {package_path} {'0' * 64} {package_path.stat().st_size} "
            "containerd.io expected-version package_test\n"
        ),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert bad_bytes.returncode == 1
    assert "package_test_sha256_expected" in bad_bytes.stdout
    assert "package_test_sha256_actual" in bad_bytes.stdout

    env["FAKE_VERSION"] = "wrong-version"
    bad_control = subprocess.run(
        ["bash"],
        input=(
            "set -Eeuo pipefail\n"
            + functions
            + f"\nverify_deb {package_path} {actual_sha} {package_path.stat().st_size} "
            "containerd.io expected-version package_test\n"
        ),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert bad_control.returncode == 1
    assert "package_test_control_version_expected" in bad_control.stdout
    assert "package_test_control_version_actual" in bad_control.stdout
    assert "package control version mismatch" in bad_control.stderr
