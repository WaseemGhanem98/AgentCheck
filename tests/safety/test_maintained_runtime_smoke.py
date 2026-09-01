"""Static safety contract for the review-only maintained-runtime smoke."""

from __future__ import annotations

import json
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
    assert env["DOCKER_INRELEASE_URL"] == docker["signed_metadata"]["inrelease_url"]
    assert (
        env["DOCKER_INRELEASE_SHA256"]
        == (docker["signed_metadata"]["inrelease_sha256"])
    )
    assert env["DOCKER_PACKAGES_URL"] == docker["signed_metadata"]["packages_url"]
    assert (
        env["DOCKER_PACKAGES_SHA256"] == (docker["signed_metadata"]["packages_sha256"])
    )

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
    assert len(python_blocks) == 5
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
    assert "--cap-drop ALL" in script
    assert "--security-opt no-new-privileges" in script
    assert "--mount" not in script
    assert "--volume" not in script
    assert "dmesg" not in script


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
        "docker_final_client",
        "docker_final_server",
        "gvisor_release",
        "gvisor_commit",
        "gvisor_sidecar_policy",
        "docker_runtime_name",
        "docker_runtime_path",
        "docker_runtime_platform",
        "oci_manifest_digest",
        "oci_config_digest",
        "oci_platform",
        "container_runtime",
        "container_host_pid",
        "container_host_executable",
        "container_host_executable_sha512",
        "live_gvisor_processes",
        "container_exit",
        "container_removed",
        "post_container_gvisor_processes",
        "sentinel_before_sha256",
        "sentinel_after_sha256",
        "sentinel_unchanged",
        "cleanup_container_absent",
        "cleanup_residual_gvisor_processes",
        "cleanup_runtime_absent",
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
        "In-container `dmesg` is neither invoked nor accepted",
        "all hosted/runtime outcomes are **NOT PROVEN**",
    ):
        assert limitation in normalized_contract
