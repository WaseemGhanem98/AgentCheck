"""The cross-version compatibility suite: what runs on every supported Python.

The full suite runs once, on the newest supported interpreter. Running the entire
suite on all three interpreters would repeat most process-heavy pure-logic tests
that cannot behave differently between 3.10 and 3.12.

What *can* differ is narrow and identifiable. The package contains no
``sys.version_info`` branching at all, so cross-version risk is not in its
control flow -- it is in the places where the package meets the interpreter:

* ``agentcheck/domain/base.py`` builds its models on a **recursive**
  ``TypeAliasType`` and an ``Annotated`` validator from ``typing_extensions``,
  and defines ``canonical_hash``. Every contract, fingerprint and serialized
  artifact in the product sits on top of that file.
* Scenarios execute in **child processes** launched through ``subprocess`` with
  an explicit interpreter and ``PYTHONPATH``. Spawn behaviour, environment
  propagation and import bootstrapping are interpreter concerns.
* Adapters read **framework-private attributes** from third-party SDKs, which
  are themselves compiled against a Python version.
* The CLI is an installed **entry point** parsed by ``argparse``.

So the compatibility suite is chosen by *where the interpreter can reach*, not
by how fast a file runs. Each category below names the property that would be
silently lost if its files stopped running on 3.10 and 3.11.

``test_compat_manifest.py`` enforces this: every file must exist, every category
must be populated, and each file must still contain the symbols that make it
evidence for its category. Removing coverage therefore fails a test rather than
quietly shrinking the matrix.
"""

from __future__ import annotations


# category -> (why it is interpreter-sensitive, test files, symbols that prove it)
COMPATIBILITY_SUITE: dict[str, dict[str, object]] = {
    "domain-and-serialization": {
        "why": (
            "Recursive TypeAliasType and Annotated validators from "
            "typing_extensions underpin every contract; canonical_hash turns "
            "them into fingerprints."
        ),
        "files": (
            "tests/agentcheck/test_domain.py",
            "tests/agentcheck/test_artifacts.py",
            "tests/agentcheck/test_suite_compatibility.py",
        ),
        "symbols": ("fingerprint", "model_dump", "IncompatibleSuiteError", "redact"),
    },
    "worker-and-isolation": {
        "why": (
            "Scenarios run in child processes launched with an explicit "
            "interpreter and PYTHONPATH; spawn and import bootstrap are "
            "interpreter behaviour."
        ),
        "files": (
            "tests/agentcheck/test_worker_process.py",
            "tests/agentcheck/test_network_containment.py",
        ),
        "symbols": ("subprocess", "worker"),
    },
    "tool-gateway-and-fail-closed": {
        "why": (
            "Unknown tools must fail closed and original handlers must never "
            "execute, on every interpreter the product claims to support."
        ),
        "files": (
            "tests/agentcheck/test_world_gateway.py",
            "tests/agentcheck/test_schema_safety.py",
        ),
        "symbols": ("ToolGateway", "unknown", "UnsafeSchema"),
    },
    "prerequisites-confirmation-followup": {
        "why": (
            "The interactive path drives a multi-stage run through the "
            "orchestrator, so it exercises staged child-process execution end "
            "to end."
        ),
        "files": (
            "tests/agentcheck/test_prerequisite_fixtures.py",
            "tests/agentcheck/test_interactive_scenarios.py",
            "tests/agentcheck/test_confirmation_variant_cases.py",
        ),
        "symbols": ("prerequisite", "followup_turns", "confirmation"),
    },
    "replay-and-source-integrity": {
        "why": (
            "Replay reloads serialized manifests and re-binds a source "
            "fileset, which is deserialization plus filesystem behaviour."
        ),
        "files": (
            "tests/agentcheck/test_replay.py",
            "tests/agentcheck/test_source_fileset.py",
        ),
        "symbols": ("manifest", "fileset"),
    },
    "adapters": {
        "why": (
            "Adapters read framework-private attributes from SDKs built "
            "against a specific Python version."
        ),
        "files": (
            "tests/agentcheck/test_openai_adapter.py",
            "tests/agentcheck/test_pydantic_ai_adapter.py",
            "tests/agentcheck/test_controlled_model.py",
            # The custom adapter reads no SDK internals, but it is the one
            # adapter whose preflight decides support from `inspect.signature`
            # and whose target code runs in the worker unwrapped by a framework.
            # Both are interpreter surfaces rather than product logic.
            "tests/agentcheck/test_custom_agent_adapter.py",
        ),
        "symbols": ("adapter", "Adapter"),
    },
    "cli-and-import": {
        "why": (
            "Console entry point, argparse wiring, target import machinery and "
            "the meta-path boundary check are all interpreter surfaces."
        ),
        "files": (
            "tests/agentcheck/test_cli_e2e.py",
            "tests/agentcheck/test_target_loading.py",
            "tests/agentcheck/test_package_boundary.py",
            # Protocol runtime-checkability and typing behaviour differ between
            # interpreters, and this contract must hold on every version the
            # package claims to support.
            "tests/agentcheck/test_custom_agent_contract.py",
        ),
        "symbols": ("main", "import"),
    },
}


def compatibility_paths() -> list[str]:
    """Every test file in the cross-version suite, deduplicated and ordered."""

    seen: dict[str, None] = {}
    for category in COMPATIBILITY_SUITE.values():
        for path in category["files"]:  # type: ignore[union-attr]
            seen.setdefault(str(path), None)
    return list(seen)


if __name__ == "__main__":
    print("\n".join(compatibility_paths()))
