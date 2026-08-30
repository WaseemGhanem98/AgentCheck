"""Target identity must describe the agent, not where it happens to live.

A developer generates a suite locally and CI validates it from a different
absolute checkout. Those two locations are the same agent, so they must inspect
to the same identity. A real change to the declared behavioral contract must
still change it.

Every test here is offline and executes no declared tool handler.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from agentcheck.adapters import OpenAIAgentsAdapter
from agentcheck.application import inspect_target


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"


def _copy_example(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        EXAMPLE,
        destination,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    return destination


def _agents_sdk() -> Any:
    return pytest.importorskip("agents")


def _weather_agent(agents: Any, *, instructions: str = "You report weather.") -> Any:
    @agents.function_tool
    def get_weather(city: str) -> str:
        """Return weather."""

        return "sunny"

    return agents.Agent(
        name="Pin Agent", instructions=instructions, tools=[get_weather]
    )


# --------------------------------------------------------------------------
# Regression: identity used to depend on the absolute checkout location.
# --------------------------------------------------------------------------


def test_identical_agent_at_two_absolute_paths_inspects_identically(
    tmp_path: Path,
) -> None:
    """The reported P1: same bytes, two checkouts, one identity.

    ``/home/user/project`` and ``/home/runner/work/project`` are the same agent.
    This asserts the property the product needs, so before the portable identity
    contract existed it failed purely because the absolute path differed.
    """

    local = _copy_example(tmp_path / "home" / "developer" / "refund-agent")
    ci = _copy_example(
        tmp_path / "runner" / "work" / "refund-agent" / "refund-agent"
    )

    _, _, local_result = inspect_target(local)
    _, _, ci_result = inspect_target(ci)

    assert local_result.value.spec_id == ci_result.value.spec_id
    # The difference under test really is location only.
    assert local != ci
    assert (
        local_result.value.identity.name.value
        == ci_result.value.identity.name.value
    )


def test_identity_survives_a_different_username_and_root(tmp_path: Path) -> None:
    """A username, a temp root, and a CI workspace root are not the agent."""

    first = _copy_example(tmp_path / "home" / "waseem" / "proj" / "agent-under-test")
    second = _copy_example(tmp_path / "opt" / "ci" / "ws" / "agent-under-test")
    third = _copy_example(tmp_path / "var" / "tmp" / "x1y2z3" / "agent-under-test")

    ids = {
        inspect_target(path)[2].value.spec_id for path in (first, second, third)
    }

    assert len(ids) == 1


def test_the_difference_between_the_two_checkouts_is_only_location(
    tmp_path: Path,
) -> None:
    """Prove the reproduction isolates path, not content."""

    local = _copy_example(tmp_path / "a" / "refund-agent")
    ci = _copy_example(tmp_path / "b" / "deep" / "refund-agent")

    def surface(spec: Any) -> Any:
        return (
            spec.identity.name.value,
            spec.identity.framework.value,
            spec.identity.framework_version.value,
            spec.instructions.system.value,
            tuple(
                (item.value.name, item.value.input_schema)
                for item in spec.tools.items
            ),
        )

    local_spec = inspect_target(local)[2].value
    ci_spec = inspect_target(ci)[2].value

    assert surface(local_spec) == surface(ci_spec)
    assert local_spec.spec_id == ci_spec.spec_id
    # Evidence keeps the real location for diagnostics; identity does not.
    assert str(local) in local_spec.interface.entrypoint.value
    assert str(ci) in ci_spec.interface.entrypoint.value
    assert local_spec.interface.entrypoint.value != ci_spec.interface.entrypoint.value


def test_portable_identity_is_deterministic_across_repeated_inspections(
    tmp_path: Path,
) -> None:
    target = _copy_example(tmp_path / "repeat" / "agent")

    ids = {inspect_target(target)[2].value.spec_id for _ in range(3)}

    assert len(ids) == 1


def test_validation_evidence_describes_portable_identity() -> None:
    documentation = (REPOSITORY_ROOT / "docs" / "validation-evidence.md").read_text(
        encoding="utf-8"
    )
    prose = " ".join(documentation.split())

    assert "identity are portable across checkout locations" in prose
    assert "entrypoint relative to the target root" in prose
    assert "identical complete generation inputs" in prose
    assert "provider realization" in prose
    assert (
        "Artifacts created before portable target identity remain location-bound"
        in prose
    )
    assert "## Fingerprints are per-location" not in documentation


# --------------------------------------------------------------------------
# Semantic changes must still move identity.
# --------------------------------------------------------------------------


def _identity(agents: Any, agent: Any) -> str:
    return (
        OpenAIAgentsAdapter()
        .inspect(agent, source="/abs/a/agent.py:agent", identity_locator="agent.py:agent")
        .spec_id
    )


def test_changed_system_instructions_change_identity() -> None:
    agents = _agents_sdk()

    baseline = _identity(agents, _weather_agent(agents))
    changed = _identity(
        agents, _weather_agent(agents, instructions="You report tides.")
    )

    assert baseline != changed


def test_changed_tool_schema_changes_identity() -> None:
    agents = _agents_sdk()

    @agents.function_tool
    def get_weather(city: str) -> str:
        """Return weather."""

        return "sunny"

    @agents.function_tool
    def get_weather_wider(city: str, unit: str) -> str:  # noqa: D401
        """Return weather."""

        return "sunny"

    narrow = agents.Agent(
        name="Pin Agent", instructions="You report weather.", tools=[get_weather]
    )
    wide = agents.Agent(
        name="Pin Agent",
        instructions="You report weather.",
        tools=[get_weather_wider],
    )

    assert _identity(agents, narrow) != _identity(agents, wide)


def test_added_or_removed_tool_changes_identity() -> None:
    agents = _agents_sdk()

    @agents.function_tool
    def get_weather(city: str) -> str:
        """Return weather."""

        return "sunny"

    @agents.function_tool
    def get_tide(city: str) -> str:
        """Return tide."""

        return "high"

    one = agents.Agent(
        name="Pin Agent", instructions="You report weather.", tools=[get_weather]
    )
    two = agents.Agent(
        name="Pin Agent",
        instructions="You report weather.",
        tools=[get_weather, get_tide],
    )

    assert _identity(agents, one) != _identity(agents, two)


def test_a_different_relative_entrypoint_is_a_different_target() -> None:
    """Two files with the same basename under different roots are not one target."""

    agents = _agents_sdk()
    agent = _weather_agent(agents)
    adapter = OpenAIAgentsAdapter()

    first = adapter.inspect(
        agent, source="/abs/a/src/agent.py:agent", identity_locator="src/agent.py:agent"
    )
    second = adapter.inspect(
        agent,
        source="/abs/a/other/agent.py:agent",
        identity_locator="other/agent.py:agent",
    )

    assert first.spec_id != second.spec_id


# --------------------------------------------------------------------------
# Every adapter derives identity the same way.
# --------------------------------------------------------------------------


def _two_locations(adapter: Any, agent: Any) -> tuple[Any, Any]:
    """Inspect one agent as if it lived in two unrelated checkouts."""

    local = adapter.inspect(
        agent,
        source="/home/developer/proj/agent.py:agent",
        identity_locator="agent.py:agent",
    )
    ci = adapter.inspect(
        agent,
        source="/home/runner/work/proj/proj/agent.py:agent",
        identity_locator="agent.py:agent",
    )
    return local, ci


def test_openai_agents_adapter_identity_is_portable() -> None:
    agents = _agents_sdk()

    local, ci = _two_locations(OpenAIAgentsAdapter(), _weather_agent(agents))

    assert local.spec_id == ci.spec_id
    # Each side still records the location-bound identity it would have had.
    assert local.legacy_spec_id is not None
    assert ci.legacy_spec_id is not None
    assert local.legacy_spec_id != ci.legacy_spec_id


def test_custom_adapter_identity_is_portable() -> None:
    from agentcheck.adapters import CustomAgentAdapter

    from tests.agentcheck.test_custom_agent_adapter import SupportAgent

    local, ci = _two_locations(CustomAgentAdapter(), SupportAgent())

    assert local.spec_id == ci.spec_id
    assert local.legacy_spec_id != ci.legacy_spec_id


def test_pydantic_ai_adapter_identity_is_portable() -> None:
    pytest.importorskip("pydantic_ai")
    from agentcheck.adapters import PydanticAIAdapter
    from pydantic_ai import Agent as PydanticAgent
    from pydantic_ai.models.test import TestModel

    agent = PydanticAgent(TestModel(), name="portable", instructions="Be brief.")

    local, ci = _two_locations(PydanticAIAdapter(), agent)

    assert local.spec_id == ci.spec_id
    assert local.legacy_spec_id != ci.legacy_spec_id


def test_identity_without_a_portable_locator_stays_location_bound() -> None:
    """A caller that established no target root cannot claim portability."""

    agents = _agents_sdk()
    adapter = OpenAIAgentsAdapter()
    agent = _weather_agent(agents)

    first = adapter.inspect(agent, source="/home/a/agent.py:agent")
    second = adapter.inspect(agent, source="/home/b/agent.py:agent")

    assert first.spec_id != second.spec_id
    # There is no separate legacy identity to record in that mode.
    assert first.legacy_spec_id is None
    assert second.legacy_spec_id is None


def test_the_pinned_location_bound_identity_is_reproduced_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy identity must be the old contract's bytes, not an approximation.

    Pinned against the fingerprinting algorithm, not the installed SDK: the
    identity deliberately bakes in ``framework_version`` (see
    SUPPORTED_SDK_MINOR_RANGE's own docstring), so this pin is only
    meaningful against one fixed version -- 0.20.0, whatever it was computed
    against -- regardless of which minor within the adapter's verified range
    (0.20-0.22) actually happens to be installed when the suite runs.
    """

    import agentcheck.adapters.openai_agents as openai_agents_adapter

    monkeypatch.setattr(openai_agents_adapter, "_sdk_version", lambda: "0.20.0")

    agents = _agents_sdk()
    agent = _weather_agent(agents)

    spec = OpenAIAgentsAdapter().inspect(
        agent, source="pin:agent.py:agent", identity_locator="agent.py:agent"
    )

    # Same value the pre-portable contract produced for this exact input.
    assert spec.legacy_spec_id == "agentspec-482269daa366d4ff8f81b74e"


# --------------------------------------------------------------------------
# End-to-end: generate in one checkout, validate from another.
# --------------------------------------------------------------------------


def _relocate(source_target: Path, destination: Path) -> Path:
    """Copy a whole target, artifacts included, to a new absolute location."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_target, destination)
    return destination


def test_a_suite_generated_locally_is_accepted_from_a_ci_checkout(
    tmp_path: Path,
) -> None:
    import agentcheck.application as application

    local = _copy_example(tmp_path / "home" / "dev" / "refund-agent")
    application.generate_suite(local, seed=1729, force=True)

    ci = _relocate(local, tmp_path / "runner" / "work" / "refund-agent" / "refund-agent")

    # The suite travels with the checkout and is accepted at the new path.
    execution = application.execute_suite(
        ci, run_id="ci-portable", persist_store=False
    )

    assert execution.frozen_suite is not None
    assert execution.scenarios


def test_a_baseline_created_locally_gates_a_run_from_a_ci_checkout(
    tmp_path: Path,
) -> None:
    import agentcheck.application as application
    from agentcheck.baseline.service import check_baseline, create_baseline

    local = _copy_example(tmp_path / "home" / "dev" / "refund-agent")
    application.execute_suite(local, run_id="local-run", persist_store=False)
    create_baseline(local, run_id="local-run", out="agentcheck-baseline.json")

    ci = _relocate(local, tmp_path / "runner" / "work" / "refund-agent" / "refund-agent")
    application.execute_suite(ci, run_id="ci-run", persist_store=False)

    checked = check_baseline(
        ci, baseline_path="agentcheck-baseline.json", run_id="ci-run"
    )

    assert checked.comparison.baseline_scenario_count > 0
    assert checked.exit_code == 0


def test_a_relocated_checkout_still_rejects_modified_source(tmp_path: Path) -> None:
    """Portability must not weaken what a suite is bound to."""

    import agentcheck.application as application
    from agentcheck.errors import ConfigurationError

    local = _copy_example(tmp_path / "home" / "dev" / "refund-agent")
    application.generate_suite(local, seed=1729, force=True)
    ci = _relocate(local, tmp_path / "runner" / "work" / "refund-agent" / "refund-agent")

    # A real change to the authoritative instruction surface, at the new path.
    agent_file = ci / "agent.py"
    agent_file.write_text(
        agent_file.read_text(encoding="utf-8").replace(
            "Use only exact account identifiers.",
            "Use approximate account identifiers.",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError) as excinfo:
        application.execute_suite(ci, run_id="ci-tampered", persist_store=False)

    message = str(excinfo.value)
    assert "re-run agentcheck generate" in message
    # The migration path is offered conditionally; it never claims to know
    # whether the cause was a relocation or a real contract change.
    assert "If this artifact predates portable target identity" in message


# --------------------------------------------------------------------------
# Legacy artifacts: usable where they were made, never elsewhere.
# --------------------------------------------------------------------------


def _rebind_suite_to(path: Path, spec_id: str) -> None:
    """Rewrite a frozen suite as if the location-bound contract had made it."""

    import json

    from agentcheck.generate.suite import FrozenSuite, write_frozen_suite

    document = json.loads(path.read_text(encoding="utf-8"))
    document["spec_id"] = spec_id
    document.pop("fingerprint", None)
    document.pop("suite_id", None)
    write_frozen_suite(
        path, FrozenSuite.model_validate_json(json.dumps(document)), force=True
    )


def _legacy_spec_id(target: Path) -> str:
    spec = inspect_target(target)[2].value
    assert spec.legacy_spec_id is not None
    return spec.legacy_spec_id


def test_a_legacy_suite_is_accepted_at_the_location_that_created_it(
    tmp_path: Path,
) -> None:
    """Recognizing it proves this inspection reproduces the recorded identity."""

    import agentcheck.application as application
    from agentcheck.generate.suite import DEFAULT_SUITE_FILENAME

    target = _copy_example(tmp_path / "home" / "dev" / "legacy-agent")
    application.generate_suite(target, seed=1729, force=True)
    _rebind_suite_to(target / DEFAULT_SUITE_FILENAME, _legacy_spec_id(target))

    execution = application.execute_suite(
        target, run_id="legacy-same-path", persist_store=False
    )

    assert execution.frozen_suite is not None
    assert execution.scenarios


def test_a_legacy_suite_is_refused_from_a_different_location(tmp_path: Path) -> None:
    """Its identity was never evidence about any other directory."""

    import agentcheck.application as application
    from agentcheck.errors import ConfigurationError
    from agentcheck.generate.suite import DEFAULT_SUITE_FILENAME

    target = _copy_example(tmp_path / "home" / "dev" / "legacy-agent")
    application.generate_suite(target, seed=1729, force=True)
    _rebind_suite_to(target / DEFAULT_SUITE_FILENAME, _legacy_spec_id(target))

    moved = _relocate(target, tmp_path / "runner" / "work" / "legacy-agent")

    with pytest.raises(ConfigurationError) as excinfo:
        application.execute_suite(moved, run_id="legacy-moved", persist_store=False)

    assert "If this artifact predates portable target identity" in str(excinfo.value)


def test_a_legacy_baseline_is_accepted_at_its_own_location(tmp_path: Path) -> None:
    import json

    import agentcheck.application as application
    from agentcheck.baseline.contract import EvaluationBaseline
    from agentcheck.baseline.service import check_baseline, create_baseline

    target = _copy_example(tmp_path / "home" / "dev" / "legacy-agent")
    application.execute_suite(target, run_id="legacy-run", persist_store=False)
    created = create_baseline(
        target, run_id="legacy-run", out="agentcheck-baseline.json"
    )

    document = json.loads(created.path.read_text(encoding="utf-8"))
    document["spec_id"] = _legacy_spec_id(target)
    document.pop("fingerprint", None)
    document.pop("baseline_id", None)
    rebound = EvaluationBaseline.model_validate_json(json.dumps(document))
    created.path.write_text(
        json.dumps(rebound.model_dump(mode="json")), encoding="utf-8"
    )

    checked = check_baseline(
        target, baseline_path="agentcheck-baseline.json", run_id="legacy-run"
    )

    assert checked.exit_code == 0


def test_a_legacy_baseline_is_refused_from_a_different_location(
    tmp_path: Path,
) -> None:
    import json

    import agentcheck.application as application
    from agentcheck.baseline.contract import EvaluationBaseline
    from agentcheck.baseline.service import check_baseline, create_baseline
    from agentcheck.errors import ConfigurationError

    target = _copy_example(tmp_path / "home" / "dev" / "legacy-agent")
    application.execute_suite(target, run_id="legacy-run", persist_store=False)
    created = create_baseline(
        target, run_id="legacy-run", out="agentcheck-baseline.json"
    )
    document = json.loads(created.path.read_text(encoding="utf-8"))
    document["spec_id"] = _legacy_spec_id(target)
    document.pop("fingerprint", None)
    document.pop("baseline_id", None)
    rebound = EvaluationBaseline.model_validate_json(json.dumps(document))
    created.path.write_text(
        json.dumps(rebound.model_dump(mode="json")), encoding="utf-8"
    )

    moved = _relocate(target, tmp_path / "runner" / "work" / "legacy-agent")
    application.execute_suite(moved, run_id="moved-run", persist_store=False)

    with pytest.raises(ConfigurationError) as excinfo:
        check_baseline(
            moved, baseline_path="agentcheck-baseline.json", run_id="moved-run"
        )

    assert "does not match the current run" in str(excinfo.value)


# --------------------------------------------------------------------------
# Locator canonicalization: separators, spellings, symlinks.
# --------------------------------------------------------------------------


def _write_config(target: Path, entrypoint: str) -> None:
    import json

    (target / "agentcheck.json").write_text(
        json.dumps({"entrypoint": entrypoint, "adapter": "openai_agents"}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "entrypoint", ("./agent.py:agent", ".//agent.py:agent")
)
def test_equivalent_entrypoint_spellings_are_one_identity(
    tmp_path: Path, entrypoint: str
) -> None:
    """Naming the same module differently is not a different agent."""

    canonical = _copy_example(tmp_path / "canonical" / "agent")
    _write_config(canonical, "agent.py:agent")
    spelled = _copy_example(tmp_path / "spelled" / "agent")
    _write_config(spelled, entrypoint)

    assert (
        inspect_target(canonical)[2].value.spec_id
        == inspect_target(spelled)[2].value.spec_id
    )


def test_portable_entrypoint_is_posix_normalized(tmp_path: Path) -> None:
    """The locator is separator-independent, so a Windows config agrees."""

    from agentcheck.config import portable_entrypoint

    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "agent.py").write_text("agent = object()\n", encoding="utf-8")

    canonical = portable_entrypoint(root, "src/agent.py:agent")

    assert canonical == "src/agent.py:agent"
    assert "\\" not in canonical
    assert portable_entrypoint(root, "./src/agent.py:agent") == canonical
    assert portable_entrypoint(root, "src//agent.py:agent") == canonical
    assert portable_entrypoint(root, "src/agent.py:agent()") == "src/agent.py:agent()"


def test_a_symlinked_checkout_resolves_to_the_same_identity(tmp_path: Path) -> None:
    real = _copy_example(tmp_path / "real" / "refund-agent")
    link = tmp_path / "linked-agent"
    link.symlink_to(real, target_is_directory=True)

    assert (
        inspect_target(real)[2].value.spec_id
        == inspect_target(link)[2].value.spec_id
    )


def test_two_different_agents_sharing_a_filename_are_not_equivalent(
    tmp_path: Path,
) -> None:
    """Identity is the contract, not the basename."""

    first = _copy_example(tmp_path / "one" / "agent-a")
    second = _copy_example(tmp_path / "two" / "agent-b")
    agent_file = second / "agent.py"
    agent_file.write_text(
        agent_file.read_text(encoding="utf-8").replace(
            "Use only exact account identifiers.",
            "Use approximate account identifiers.",
            1,
        ),
        encoding="utf-8",
    )

    assert (
        inspect_target(first)[2].value.spec_id
        != inspect_target(second)[2].value.spec_id
    )


# --------------------------------------------------------------------------
# Replay and containment are unchanged by portability.
# --------------------------------------------------------------------------


def test_replay_binds_across_checkouts_but_still_rejects_changed_source(
    tmp_path: Path,
) -> None:
    """A portable spec binding must not become a weaker source binding."""

    import agentcheck.application as application
    from agentcheck.errors import ConfigurationError

    local = _copy_example(tmp_path / "home" / "dev" / "refund-agent")
    execution = application.execute_suite(
        local, run_id="replay-src", persist_store=False
    )
    assert execution.replay_manifest_path is not None
    manifest_name = execution.replay_manifest_path.name

    ci = _relocate(local, tmp_path / "runner" / "work" / "refund-agent")
    manifest = f".agentcheck/replay/{manifest_name}"

    replayed = application.replay_suite(
        ci, manifest, run_id="replay-ci", persist_store=False
    )
    assert replayed.scenarios

    # Same manifest, same new location, changed source: still refused.
    agent_file = ci / "agent.py"
    agent_file.write_text(
        agent_file.read_text(encoding="utf-8").replace(
            "Use only exact account identifiers.",
            "Use approximate account identifiers.",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError):
        application.replay_suite(
            ci, manifest, run_id="replay-bad", persist_store=False
        )


def test_inspection_across_locations_executes_no_declared_handler(
    tmp_path: Path,
) -> None:
    """Portable identity is derived from the declared contract, never by running it."""

    marker = tmp_path / "handler-ran"
    target = _copy_example(tmp_path / "home" / "dev" / "refund-agent")
    agent_file = target / "agent.py"
    agent_file.write_text(
        agent_file.read_text(encoding="utf-8").replace(
            "def lookup_account(",
            f"def lookup_account(  # noqa\n"
            f"    *_probe_args, **_probe_kwargs\n"
            f") -> str:\n"
            f"    open({str(marker)!r}, 'w').close()\n"
            f"    raise AssertionError('handler executed')\n"
            f"def _unused_lookup_account(",
            1,
        ),
        encoding="utf-8",
    )

    try:
        inspect_target(target)
    except Exception:
        # A malformed probe target is fine; the assertion under test is that no
        # declared handler body ran during inspection.
        pass

    assert not marker.exists()


def test_a_declared_policy_pack_is_bound_separately_from_spec_identity(
    tmp_path: Path,
) -> None:
    """Packs are declared for a run, not read out of the agent.

    No adapter populates ``tool_policies``, so the authoritative policy surface
    today is the declared pack set. It is bound explicitly by the replay spec
    binding rather than folded into ``spec_id``, and that binding is what
    refuses a run whose packs changed.
    """

    from types import SimpleNamespace

    from agentcheck.errors import ConfigurationError
    from agentcheck.replay.bind import verify_replay_spec_bindings
    from agentcheck.replay.manifest import SpecBinding

    target = _copy_example(tmp_path / "home" / "dev" / "refund-agent")
    spec = inspect_target(target)[2].value
    assert spec.tool_policies.items == ()

    binding = SpecBinding(
        spec_id=spec.spec_id,
        adapter="openai_agents",
        entrypoint="agent.py:agent",
        policy_pack_ids=("declared-pack",),
    )
    manifest = SimpleNamespace(
        spec_binding=binding,
        source_binding=SimpleNamespace(
            framework=spec.identity.framework.value,
            framework_version=spec.identity.framework_version.value,
        ),
    )

    # The portable identity matches, so only the pack binding can refuse this.
    with pytest.raises(ConfigurationError, match="policy packs"):
        verify_replay_spec_bindings(manifest, spec=spec, policy_pack_ids=())

    verify_replay_spec_bindings(
        manifest, spec=spec, policy_pack_ids=("declared-pack",)
    )


def test_editing_a_suite_identity_breaks_its_fingerprint(tmp_path: Path) -> None:
    """Portable identity does not make a hand-edited artifact acceptable."""

    import json

    import agentcheck.application as application
    from agentcheck.errors import ConfigurationError
    from agentcheck.generate.suite import DEFAULT_SUITE_FILENAME

    target = _copy_example(tmp_path / "home" / "dev" / "refund-agent")
    application.generate_suite(target, seed=1729, force=True)
    suite_path = target / DEFAULT_SUITE_FILENAME

    document = json.loads(suite_path.read_text(encoding="utf-8"))
    document["spec_id"] = "agentspec-0000000000000000000000ff"
    suite_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ConfigurationError):
        application.execute_suite(target, run_id="tampered", persist_store=False)


# --------------------------------------------------------------------------
# Adversarial: the legacy bridge must never become a downgrade path.
# --------------------------------------------------------------------------


def _portable_and_legacy(**overrides: Any) -> tuple[str, str | None]:
    from agentcheck.adapters.base import portable_identity

    fingerprint: dict[str, Any] = {
        "name": "agent",
        "framework_version": "0.20.0",
        "model": "scripted",
        "instructions_hash": "sha256:aaa",
        "tools": [
            {
                "name": "act",
                "input_schema": {"type": "object"},
                "output_schema": None,
                "state_changing": True,
                "destructive": False,
                "replaceable": True,
            }
        ],
    }
    fingerprint.update(overrides)
    return portable_identity(
        fingerprint,
        location_locator="/home/dev/proj/agent.py:agent",
        identity_locator="agent.py:agent",
    )


def test_every_semantic_input_moves_portable_identity() -> None:
    """Each declared input the fingerprint carries is load-bearing."""

    base, _ = _portable_and_legacy()

    changed = {
        "framework_version": _portable_and_legacy(framework_version="0.21.0")[0],
        "model": _portable_and_legacy(model="other")[0],
        "name": _portable_and_legacy(name="renamed")[0],
        "instructions": _portable_and_legacy(instructions_hash="sha256:bbb")[0],
        "output_schema": _portable_and_legacy(
            tools=[
                {
                    "name": "act",
                    "input_schema": {"type": "object"},
                    "output_schema": {"type": "string"},
                    "state_changing": True,
                    "destructive": False,
                    "replaceable": True,
                }
            ]
        )[0],
        "destructive_flag": _portable_and_legacy(
            tools=[
                {
                    "name": "act",
                    "input_schema": {"type": "object"},
                    "output_schema": None,
                    "state_changing": True,
                    "destructive": True,
                    "replaceable": True,
                }
            ]
        )[0],
    }

    for label, value in changed.items():
        assert value != base, f"{label} did not move portable identity"
    assert len(set(changed.values())) == len(changed)


def test_the_legacy_identity_binds_the_same_semantic_surface() -> None:
    """A legacy id is the stricter binding, never a looser one."""

    _, legacy_base = _portable_and_legacy()
    _, legacy_changed = _portable_and_legacy(instructions_hash="sha256:bbb")

    assert legacy_base is not None
    assert legacy_changed is not None
    assert legacy_base != legacy_changed


def test_a_forged_identity_in_a_suite_is_refused(tmp_path: Path) -> None:
    """An attacker-chosen value matches neither identity."""

    import agentcheck.application as application
    from agentcheck.errors import ConfigurationError
    from agentcheck.generate.suite import DEFAULT_SUITE_FILENAME

    target = _copy_example(tmp_path / "home" / "dev" / "refund-agent")
    application.generate_suite(target, seed=1729, force=True)
    _rebind_suite_to(
        target / DEFAULT_SUITE_FILENAME, "agentspec-deadbeefdeadbeefdeadbeef"
    )

    with pytest.raises(ConfigurationError, match="re-run agentcheck generate"):
        application.execute_suite(target, run_id="forged", persist_store=False)


def test_a_legacy_suite_from_another_agent_at_this_path_is_refused(
    tmp_path: Path,
) -> None:
    """The legacy bridge still binds the behavioral contract, not just the path."""

    import agentcheck.application as application
    from agentcheck.errors import ConfigurationError
    from agentcheck.generate.suite import DEFAULT_SUITE_FILENAME

    target = _copy_example(tmp_path / "home" / "dev" / "refund-agent")
    application.generate_suite(target, seed=1729, force=True)

    # The legacy identity of a *different* contract at this very path.
    from agentcheck.adapters.base import portable_identity
    from agentcheck.config import portable_entrypoint

    live = inspect_target(target)[2].value
    _, foreign_legacy = portable_identity(
        {
            "name": "some other agent",
            "framework_version": live.identity.framework_version.value,
            "model": None,
            "instructions_hash": "sha256:unrelated",
            "tools": [],
        },
        location_locator=str(target / "agent.py") + ":agent",
        identity_locator=portable_entrypoint(target, "agent.py:agent"),
    )
    assert foreign_legacy is not None
    _rebind_suite_to(target / DEFAULT_SUITE_FILENAME, foreign_legacy)

    with pytest.raises(ConfigurationError, match="re-run agentcheck generate"):
        application.execute_suite(target, run_id="foreign-legacy", persist_store=False)


def test_a_stale_legacy_identity_in_a_stored_spec_cannot_rescue_a_suite(
    tmp_path: Path,
) -> None:
    """The suite gate inspects live, so a persisted legacy value is never consulted."""

    import json

    import agentcheck.application as application
    from agentcheck.errors import ConfigurationError
    from agentcheck.generate.suite import DEFAULT_SUITE_FILENAME

    target = _copy_example(tmp_path / "home" / "dev" / "refund-agent")
    application.generate_suite(target, seed=1729, force=True)
    application.execute_suite(target, run_id="seed-run", persist_store=False)

    forged = "agentspec-0123456789abcdef01234567"
    spec_path = target / ".agentcheck" / "runs" / "seed-run" / "agent-spec.json"
    document = json.loads(spec_path.read_text(encoding="utf-8"))
    document["legacy_spec_id"] = forged
    spec_path.write_text(json.dumps(document), encoding="utf-8")
    _rebind_suite_to(target / DEFAULT_SUITE_FILENAME, forged)

    with pytest.raises(ConfigurationError, match="re-run agentcheck generate"):
        application.execute_suite(target, run_id="after-forgery", persist_store=False)


def test_a_nested_entrypoint_is_portable_and_distinct(tmp_path: Path) -> None:
    """A deeper relative entrypoint travels, and is not the same as a shallow one."""

    import json

    def build(root: Path, relative: str) -> Path:
        module = root / relative
        module.parent.mkdir(parents=True, exist_ok=True)
        for parent in list(module.parents)[:-1]:
            if parent == root:
                break
            (parent / "__init__.py").write_text("", encoding="utf-8")
        shutil.copyfile(EXAMPLE / "agent.py", module)
        (root / "agentcheck.json").write_text(
            json.dumps(
                {"entrypoint": f"{relative}:agent", "adapter": "openai_agents"}
            ),
            encoding="utf-8",
        )
        return root

    nested_a = build(tmp_path / "one", "src/deep/agent.py")
    nested_b = build(tmp_path / "elsewhere" / "two", "src/deep/agent.py")
    shallow = build(tmp_path / "three", "agent.py")

    assert (
        inspect_target(nested_a)[2].value.spec_id
        == inspect_target(nested_b)[2].value.spec_id
    )
    assert (
        inspect_target(nested_a)[2].value.spec_id
        != inspect_target(shallow)[2].value.spec_id
    )
