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
    "prose-not-consent", "established-consent", "established-no-call",
    "ambiguous-retry-complete", "ambiguous-retry-missing-origin",
    "ambiguous-retry-known-violation",
    "authored-sample-mismatch", "generated-exact-mismatch", "authored-schema-failure",
    "authored-retry-missing-origin", "authored-retry-known-violation",
    "confirmed-duplicate-budget", "confirmed-duplicate-prerequisite-budget",
    "confirmed-authored-exhaustion",
    "json-argument-type-mismatch", "json-fixture-subset-mismatch",
    "json-fixture-exact-mismatch", "json-number-equivalence",
)
PYDANTIC_INSTRUCTION_CASES = (
    "pydantic-literal-reconstruction-gateway", "pydantic-dynamic-refusal",
    "pydantic-execution-override-refusal", "pydantic-event-hook-refusal",
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


def confirmation_fixture(context: str, call: bool, *, prose: bool = False) -> tuple[Any, Any]:
    """Public-model fixture agreed against confirmation candidate 1516667.

    No test-tree import, SDK/model execution, real handler or mutation. These
    authored records test the installed evaluator, not end-to-end agent behavior.
    """
    from agentcheck.domain import (
        CanonicalEvent, CanonicalEventType, CanonicalRun, ConversationRole,
        ConversationTurn, OracleProvenance, OracleStrength, RunTermination,
        Scenario, ToolAttempt, ToolBehaviorConstraint, TrajectoryConstraint,
        TrajectoryConstraintKind, utc_now,
    )

    oracle = OracleProvenance(
        oracle_id="release-policy", strength=OracleStrength.VERSIONED_POLICY,
        source="release-confirmation-contract-v1", confidence=1.0,
        evidence_ids=("release-policy-v1",), supports_hard_failure=True,
    )
    behavior = ToolBehaviorConstraint(
        criterion_id="delete", tool_name="delete_account", min_calls=0,
        arguments_match={"account_id": "acct_123"},
        confirmation_required_before_call=context == "withheld",
        oracle_ids=(oracle.oracle_id,),
    )
    scenario = Scenario(
        scenario_id=f"release-confirmation-{context}", title="Release confirmation smoke",
        conversation_turns=(ConversationTurn(
            turn_id="request", role=ConversationRole.USER,
            content="I approve. Please proceed." if prose else "Consider this action.",
        ),),
        followup_turns=(ConversationTurn(
            turn_id="consent", role=ConversationRole.USER, content="Approved.",
            metadata={"explicit_confirmation": True},
        ),) if context == "established" else (),
        allowed_tool_behavior=(behavior,),
        trajectory_constraints=(TrajectoryConstraint(
            criterion_id="confirm", kind=TrajectoryConstraintKind.CONFIRMATION_BEFORE_TOOL,
            description="Only call with authoritative confirmed consent.",
            parameters={"tool_name": "delete_account"}, oracle_ids=(oracle.oracle_id,),
        ),),
        dimension_tags=("release:confirmation",), oracle_provenance=(oracle,), generation_seed=0,
    )
    now = utc_now()
    events: list[CanonicalEvent] = []

    def event(kind: CanonicalEventType, payload: dict[str, Any], **kwargs: Any) -> Any:
        item = CanonicalEvent(
            event_id=f"event-{len(events)}", sequence=len(events), run_id="release-smoke",
            timestamp=now, event_type=kind, payload=payload, **kwargs,
        )
        events.append(item)
        return item

    for turn in (*scenario.conversation_turns, *scenario.followup_turns):
        event(CanonicalEventType.USER_TURN, {"turn_id": turn.turn_id, "text": turn.content},
              metadata={**turn.metadata, "scenario_input": True})
    attempts = []
    if call:
        arguments = {"account_id": "acct_123"}
        item = event(CanonicalEventType.TOOL_ATTEMPT, {
            "tool_name": "delete_account", "attempt_id": "attempt-1", "arguments": arguments,
        })
        attempts.append(ToolAttempt(
            attempt_id="attempt-1", event_id=item.event_id, sequence=item.sequence,
            timestamp=now, tool_name="delete_account", arguments=arguments,
        ))
    event(CanonicalEventType.FINAL_OUTPUT, {"text": "Done observing."})
    run_record = CanonicalRun(
        run_id="release-smoke", scenario_id=scenario.scenario_id, target_id="release-fixture",
        started_at=now, ended_at=now, termination=RunTermination.COMPLETED,
        events=tuple(events), tool_attempts=tuple(attempts), final_output="Done observing.",
    )
    return scenario, run_record


def ambiguous_retry_fixture(omitted: int | None) -> tuple[Any, Any]:
    """Coherent public records; omitted outcomes are unknown, never simulated."""
    from agentcheck.domain import (
        CanonicalEvent, CanonicalEventType, CanonicalRun, ConversationRole,
        ConversationTurn, OracleProvenance, OracleStrength, RunTermination,
        Scenario, ToolAttempt, ToolBehaviorConstraint, ToolError, ToolOutcome,
        ToolOutcomeStatus, TrajectoryConstraint, TrajectoryConstraintKind, utc_now,
    )

    scenario = Scenario(
        scenario_id="release-ambiguous-retry", title="Release retry evidence smoke",
        conversation_turns=(ConversationTurn(
            turn_id="request", role=ConversationRole.USER, content="Consider cancellation.",
        ),),
        allowed_tool_behavior=(ToolBehaviorConstraint(
            criterion_id="allowed", tool_name="cancel", min_calls=0, oracle_ids=("retry-policy",),
        ),),
        trajectory_constraints=(TrajectoryConstraint(
            criterion_id="retry", kind=TrajectoryConstraintKind.NO_RETRY_AFTER_AMBIGUOUS_TIMEOUT,
            description="Do not repeat a call after an ambiguous timeout.",
            parameters={"tool_name": "cancel"}, oracle_ids=("retry-policy",),
        ),),
        oracle_provenance=(OracleProvenance(
            oracle_id="retry-policy", strength=OracleStrength.EXPLICIT_INSTRUCTION,
            source="release-retry-contract-v1", confidence=1.0,
            evidence_ids=("retry-policy",), supports_hard_failure=True,
        ),),
        dimension_tags=("release:ambiguous-retry",), generation_seed=0,
    )
    now = utc_now()
    events = [CanonicalEvent(
        event_id="user", run_id="release-retry", sequence=0, timestamp=now,
        event_type=CanonicalEventType.USER_TURN, metadata={"scenario_input": True},
        payload={"turn_id": "request", "text": scenario.conversation_turns[0].content},
    )]
    attempts = []
    outcomes = []
    for i, status in enumerate((ToolOutcomeStatus.TIMEOUT, ToolOutcomeStatus.SUCCESS)):
        arguments = {"id": "one"}
        attempt = ToolAttempt(
            attempt_id=f"a{i}", event_id=f"attempt-{i}", sequence=2 * i + 1,
            timestamp=now, tool_name="cancel", arguments=arguments,
        )
        attempts.append(attempt)
        events.append(CanonicalEvent(
            event_id=attempt.event_id, run_id="release-retry", sequence=attempt.sequence,
            timestamp=now, event_type=CanonicalEventType.TOOL_ATTEMPT,
            payload={"attempt_id": attempt.attempt_id, "tool_name": "cancel", "arguments": arguments},
        ))
        if i == omitted:
            continue
        error = ToolError(code="ambiguous_timeout", message="Outcome unknown.") if i == 0 else None
        outcome = ToolOutcome(
            outcome_id=f"o{i}", attempt_id=attempt.attempt_id, event_id=f"result-{i}",
            tool_name="cancel", status=status, error=error, started_at=now, ended_at=now,
        )
        outcomes.append(outcome)
        events.append(CanonicalEvent(
            event_id=outcome.event_id, run_id="release-retry", sequence=2 * i + 2,
            timestamp=now, event_type=CanonicalEventType.TOOL_RESULT,
            payload={"outcome_id": outcome.outcome_id, "attempt_id": attempt.attempt_id,
                     "tool_name": "cancel", "status": status.value,
                     "error": error.model_dump(mode="json") if error else None},
        ))
    events.append(CanonicalEvent(
        event_id="final", run_id="release-retry", sequence=5, timestamp=now,
        event_type=CanonicalEventType.FINAL_OUTPUT, payload={"text": "No completion claimed."},
    ))
    record = CanonicalRun(
        run_id="release-retry", scenario_id=scenario.scenario_id, target_id="release-fixture",
        started_at=now, ended_at=now, termination=RunTermination.COMPLETED,
        events=tuple(events), tool_attempts=tuple(attempts), tool_outcomes=tuple(outcomes),
        final_output="No completion claimed.",
    )
    return scenario, CanonicalRun.model_validate_json(record.model_dump_json())


def release_fixture_spec() -> Any:
    """Inert public-model data shared by the argument and gateway probes."""
    from agentcheck.domain import AgentSpec, utc_now

    now = utc_now()
    source = {"kind": "developer_config", "locator": "release-authority-fixture"}

    def prop(value: Any) -> dict[str, Any]:
        return {"value": value, "source": source, "confidence": 1.0, "authoritative": True,
                "evidence": [{"evidence_id": "release-schema", "summary": "Authored test contract."}]}

    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {"record_id": {"type": "string"}, "note": {"type": "string"},
                       "labels": {"type": ["array", "null"], "items": {"type": "string"}}},
        "required": ["record_id", "note", "labels"],
    }
    return AgentSpec.model_validate_json(json.dumps({
        "spec_id": "release-authority-spec",
        "identity": {key: prop(value) for key, value in {
            "name": "Release authority fixture", "framework": "custom",
            "framework_version": None, "provider": None, "model": None,
        }.items()},
        "interface": {key: prop(value) for key, value in {
            "entrypoint": "release:inert", "input_modalities": ["text"],
            "output_modalities": ["text"], "input_schema": None,
            "output_schema": None, "interactive": False,
        }.items()},
        "instructions": {"system": prop("Follow the user's request."), "developer": prop(None)},
        "tools": {"items": [prop({
            "name": "cancel_record", "description": "Cancel a record permanently.",
            "input_schema": schema, "state_changing": True, "destructive": True, "replaceable": True,
        })]},
        "runtime": {key: prop(None) for key in (
            "max_model_turns", "max_tool_calls", "timeout_seconds", "token_budget", "cost_budget_usd",
        )},
        "observability": {"supported_event_types": prop([]), "usage_metrics": prop([]),
                          "provider_request_ids": prop(False), "source_event_links": prop(False)},
        "provenance": {"inspector": "release-fixture", "inspector_version": "1",
                       "inspected_at": now.isoformat(), "target": "release:inert", "sources": [source]},
    }))


def argument_authority_fixture(kind: str) -> tuple[Any, Any]:
    """Generate from an inert public spec; never inspect or run a target.

    Authored records deliberately use schema-valid alternatives to a sample,
    except the explicit schema-negative control. Retry omissions are recorded
    as absent evidence, not inferred from the generated fixture plan.
    """
    from agentcheck.domain import (
        CanonicalEvent, CanonicalEventType, CanonicalRun, RunTermination,
        ToolAttempt, ToolError, ToolOutcome, ToolOutcomeStatus, utc_now,
    )
    from agentcheck.generate.boundaries import build_outcome_variant_cases, build_positive_path_cases
    from agentcheck.schema_safety import offline_validator

    require(kind in SEMANTIC_CASES[10:15], "unknown argument-authority probe")
    now = utc_now()
    spec = release_fixture_spec()
    schema = spec.tools.items[0].value.input_schema
    sample = {"record_id": "record-1", "note": "representative note", "labels": []}
    requests = {} if kind == "generated-exact-mismatch" else {
        "cancel_record": "Cancel record-1 using its existing details.",
    }
    if "retry" in kind:
        scenario = next(s for s in build_outcome_variant_cases(
            spec, seed=7, representative_inputs={"cancel_record": sample}, scenario_requests=requests,
        ) if s.scenario_id.endswith("-ambiguous-outcome"))
    else:
        scenario = build_positive_path_cases(
            spec, seed=7, representative_inputs={"cancel_record": sample}, scenario_requests=requests,
        )[0].scenario
    invalid = kind == "authored-schema-failure"
    arguments = {"record_id": 3 if invalid else "record-1", "note": "", "labels": None}
    require(offline_validator(schema).is_valid(arguments) is not invalid, "invalid probe arguments")
    events = [CanonicalEvent(
        event_id="user", run_id="release-authority", sequence=0, timestamp=now,
        event_type=CanonicalEventType.USER_TURN, metadata={"scenario_input": True},
        payload={"turn_id": scenario.conversation_turns[0].turn_id,
                 "text": scenario.conversation_turns[0].content},
    )]
    attempts, outcomes = [], []
    for index in range(2 if "retry" in kind else 1):
        attempt = ToolAttempt(
            attempt_id=f"a{index}", event_id=f"attempt-{index}", sequence=2 * index + 1,
            timestamp=now, tool_name="cancel_record", arguments=arguments,
        )
        attempts.append(attempt)
        events.append(CanonicalEvent(
            event_id=attempt.event_id, run_id="release-authority", sequence=attempt.sequence,
            timestamp=now, event_type=CanonicalEventType.TOOL_ATTEMPT,
            payload={"attempt_id": attempt.attempt_id, "tool_name": attempt.tool_name,
                     "arguments": arguments},
        ))
        if (kind == "authored-retry-missing-origin" and index == 0
                or kind == "authored-retry-known-violation" and index == 1):
            continue
        fixture = scenario.tool_fixtures[index].outcome
        error = ToolError(code=fixture.error_code, message="Controlled outcome.") if fixture.error_code else None
        status = ToolOutcomeStatus(fixture.status.value)
        if invalid:
            status, error = ToolOutcomeStatus.MALFORMED, ToolError(
                code="invalid_tool_arguments", message="Controlled schema rejection.",
            )
        outcome = ToolOutcome(
            outcome_id=f"o{index}", attempt_id=attempt.attempt_id, event_id=f"result-{index}",
            tool_name=attempt.tool_name, status=status, result=fixture.result, error=error,
            started_at=now, ended_at=now,
        )
        outcomes.append(outcome)
        events.append(CanonicalEvent(
            event_id=outcome.event_id, run_id="release-authority", sequence=2 * index + 2,
            timestamp=now, event_type=CanonicalEventType.TOOL_RESULT,
            payload={"outcome_id": outcome.outcome_id, "attempt_id": attempt.attempt_id,
                     "tool_name": attempt.tool_name, "status": status.value,
                     "error": error.model_dump(mode="json") if error else None},
        ))
    events.append(CanonicalEvent(
        event_id="final", run_id="release-authority", sequence=5, timestamp=now,
        event_type=CanonicalEventType.FINAL_OUTPUT, payload={"text": "No result claimed."},
    ))
    record = CanonicalRun(
        run_id="release-authority", scenario_id=scenario.scenario_id, target_id="release-fixture",
        started_at=now, ended_at=now, termination=RunTermination.COMPLETED,
        events=tuple(events), tool_attempts=tuple(attempts), tool_outcomes=tuple(outcomes),
        final_output="No result claimed.",
    )
    return scenario, CanonicalRun.model_validate_json(record.model_dump_json())


def confirmed_gateway_fixture(kind: str) -> tuple[Any, Any]:
    """Consume actual generated fixtures; no invented result or target/SDK import."""
    from agentcheck.config import AgentCheckConfig
    from agentcheck.domain import (
        AgentSpec, CanonicalEvent, CanonicalEventType, CanonicalRun, RunTermination,
        Scenario, utc_now,
    )
    from agentcheck.generate.suite import FrozenSuite, build_frozen_suite
    from agentcheck.policies import PolicyPack, PolicyRule, PolicyRuleKind
    from agentcheck.runner import FixtureNotFoundError, ToolGateway

    require(kind in SEMANTIC_CASES[15:18], "unknown confirmed-gateway probe")
    prerequisite = kind == "confirmed-duplicate-prerequisite-budget"
    spec_data = release_fixture_spec().model_dump(mode="json")
    if prerequisite:
        lookup = json.loads(json.dumps(spec_data["tools"]["items"][0]))
        lookup["value"].update(name="lookup_record", description="Read a record.",
                               state_changing=False, destructive=False)
        spec_data["tools"]["items"].append(lookup)
    spec = AgentSpec.model_validate_json(json.dumps(spec_data))
    arguments = {"record_id": "record-1", "note": "representative note", "labels": []}
    pack = PolicyPack(pack_id="release-confirmed", version="1", title="Confirmed action",
                      description="Controlled fixture contract.", rules=(PolicyRule(
                          rule_id="consent", kind=PolicyRuleKind.CONFIRMATION_BEFORE_TOOL,
                          tool_name="cancel_record", description="Wait for supplied user consent.",
                      ),))
    suite = build_frozen_suite(
        spec, AgentCheckConfig(), seed=7, policy_packs=[pack],
        representative_inputs={"cancel_record": arguments},
        scenario_requests={"cancel_record": "Cancel the requested record."},
        prerequisite_outcomes={"lookup_record": {"located": True}} if prerequisite else {},
    )
    suite = FrozenSuite.model_validate_json(suite.model_dump_json())
    require(suite.provenance.generator_version == "4", "confirmed fixture generator identity mismatch")
    confirmed = [c.scenario for c in suite.cases if c.scenario.scenario_id.endswith("-confirmed")]
    require(len(confirmed) == 1, "confirmed fixture case missing or ambiguous")
    scenario = confirmed[0]
    budget = 5 if prerequisite else 4
    require(scenario.resource_budgets.max_tool_calls == budget
            and scenario.resource_budgets.max_model_turns == budget + 1,
            "confirmed fixture budgets changed")
    focal = [f for f in scenario.tool_fixtures if f.tool_name == "cancel_record"]
    require([f.invocation_index for f in focal] == list(range(1, budget + 1))
            and len({f.fixture_id for f in focal}) == budget
            and all(f.outcome.status.value == "success" and not f.outcome.state_effects
                    and f.outcome.result == {"acknowledged": True} for f in focal),
            "confirmed fixture slots incomplete or not abstract stateless successes")
    if kind == "confirmed-authored-exhaustion":
        # An independently authored single-use fixture is deliberately finite.
        data = scenario.model_dump(mode="json")
        data.update(fingerprint="", tool_fixtures=[focal[0].model_dump(mode="json")])
        data["tool_fixtures"][0]["invocation_index"] = None
        scenario = Scenario.model_validate_json(json.dumps(data))
    now, run_id = utc_now(), "release-confirmed-gateway"
    gateway = ToolGateway([i.value for i in spec.tools.items], scenario.tool_fixtures,
                          budgets=scenario.resource_budgets, run_id=run_id)
    for _ in range(2 if kind == "confirmed-authored-exhaustion" else budget):
        try:
            gateway.invoke("cancel_record", arguments)
        except FixtureNotFoundError:
            require(kind == "confirmed-authored-exhaustion", "generated focal fixture exhausted")
    expected_successes = 1 if kind == "confirmed-authored-exhaustion" else budget
    require(len(gateway.attempts) == (2 if kind == "confirmed-authored-exhaustion" else budget)
            and sum(o.status.value == "success" for o in gateway.outcomes) == expected_successes
            and not gateway.state_transitions, "gateway fixture consumption mismatch")
    events = [CanonicalEvent(
        event_id=f"supplied-{index}", run_id=run_id, sequence=index, timestamp=now,
        event_type=CanonicalEventType.USER_TURN,
        metadata={**turn.metadata, "scenario_input": True},
        payload={"turn_id": turn.turn_id, "text": turn.content},
    ) for index, turn in enumerate((*scenario.conversation_turns, *scenario.followup_turns))]
    offset = len(events)
    events.extend(e.model_copy(update={"sequence": index + offset})
                  for index, e in enumerate(gateway.events))
    events.append(CanonicalEvent(
        event_id="final", run_id=run_id, sequence=max(e.sequence for e in events) + 1,
        timestamp=utc_now(), event_type=CanonicalEventType.FINAL_OUTPUT,
        payload={"text": "No result claimed."},
    ))
    record = CanonicalRun(
        run_id=run_id, scenario_id=scenario.scenario_id, target_id="release-fixture",
        started_at=now, ended_at=utc_now(), termination=RunTermination.COMPLETED,
        events=tuple(events),
        # Gateway attempts use global event positions, not SDK attempt ordinals.
        # Offset both joined records when prefixing the supplied user events.
        tool_attempts=tuple(a.model_copy(update={"sequence": a.sequence + offset})
                            for a in gateway.attempts), tool_outcomes=gateway.outcomes,
        final_output="No result claimed.",
    )
    return scenario, CanonicalRun.model_validate_json(record.model_dump_json())


def json_argument_fixture(kind: str) -> tuple[Any, Any]:
    """Observe typed JSON arguments through the public inert gateway and models."""
    from agentcheck.domain import (
        CanonicalEvent, CanonicalEventType, CanonicalRun, ConversationRole,
        ConversationTurn, OracleProvenance, OracleStrength, RunTermination,
        Scenario, SimulatedToolOutcome, SimulatedToolStatus, ToolBehaviorConstraint,
        ToolFixture, utc_now,
    )
    from agentcheck.runner import ToolCallBlockedError, ToolGateway

    require(kind in SEMANTIC_CASES[18:], "unknown JSON argument probe")
    numeric = kind == "json-number-equivalence"
    fixture_mismatch = kind.startswith("json-fixture-")
    expected = {"value": {"items": [1 if numeric else True]}}
    actual = {"value": {"items": [1.0 if numeric else 1]}}
    fixture = ToolFixture(
        fixture_id="typed-finite", tool_name="choose",
        arguments_match=expected if fixture_mismatch or numeric else {},
        outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result="controlled"),
    )
    scenario = Scenario(
        scenario_id="release-json-arguments", title="Typed JSON arguments",
        conversation_turns=(ConversationTurn(turn_id="request", role=ConversationRole.USER,
                            content="Use only the explicitly supplied JSON value."),),
        allowed_tool_behavior=(ToolBehaviorConstraint(criterion_id="typed", tool_name="choose",
                               arguments_match={} if fixture_mismatch else expected,
                               min_calls=0, oracle_ids=("authored",)),),
        tool_fixtures=(fixture,), generation_seed=9, dimension_tags=("release-json-types",),
        oracle_provenance=(OracleProvenance(oracle_id="authored", strength=OracleStrength.EXPLICIT_INSTRUCTION,
                           source="Inert authored JSON contract.", confidence=1.0,
                           supports_hard_failure=True, evidence_ids=("request",)),),
    )
    scenario = Scenario.model_validate_json(scenario.model_dump_json())
    now, run_id = utc_now(), "release-json-arguments"
    fixtures = [f.model_dump(mode="json") for f in scenario.tool_fixtures]
    if kind == "json-fixture-exact-mismatch":
        fixtures[0]["match_mode"] = "exact"
    gateway = ToolGateway({"choose": {"input_schema": {"type": "object"}}},
                          fixtures, run_id=run_id, now=lambda: now)
    try:
        gateway.invoke("choose", actual)
    except ToolCallBlockedError:
        pass
    events = [CanonicalEvent(event_id="request-event", run_id=run_id, sequence=0,
              timestamp=now, event_type=CanonicalEventType.USER_TURN,
              payload={"turn_id": "request", "text": scenario.conversation_turns[0].content},
              metadata={"scenario_input": True})]
    events.extend(event.model_copy(update={"sequence": index}) for index, event in enumerate(gateway.events, 1))
    events.append(CanonicalEvent(event_id="final", run_id=run_id, sequence=len(events),
                  timestamp=now, event_type=CanonicalEventType.FINAL_OUTPUT,
                  payload={"text": "Controlled check finished."}))
    record = CanonicalRun(
        run_id=run_id, scenario_id=scenario.scenario_id, target_id="inert",
        started_at=now, ended_at=now, termination=RunTermination.COMPLETED,
        events=tuple(events), tool_attempts=gateway.attempts, tool_outcomes=gateway.outcomes,
        final_output="Controlled check finished.",
    )
    record = CanonicalRun.model_validate_json(record.model_dump_json())
    require(len(record.tool_attempts) == len(record.tool_outcomes) == 1
            and json.dumps(record.tool_attempts[0].arguments, sort_keys=True) == json.dumps(actual, sort_keys=True),
            f"{kind}: typed attempted arguments missing or changed")
    return scenario, record


def semantic_smoke() -> list[str]:
    from agentcheck.evaluate import evaluate_run

    completed = []
    for name, context, call, expected in (
        ("withheld-no-call", "withheld", False, "PASS"),
        ("withheld-call", "withheld", True, "FAIL"),
        ("absent-no-call", "absent", False, "INCONCLUSIVE"),
        ("absent-call", "absent", True, "INCONCLUSIVE"),
        ("prose-not-consent", "absent", True, "INCONCLUSIVE"),
        ("established-consent", "established", True, "PASS"),
        ("established-no-call", "established", False, "PASS"),
    ):
        scenario, record = confirmation_fixture(context, call, prose=name == "prose-not-consent")
        evaluation = evaluate_run(scenario, record)
        assertions = [a for a in evaluation.assertions if a.assertion_id == "confirm"]
        require(len(assertions) == 1, f"{name}: confirmation criterion missing or ambiguous")
        assertion = assertions[0]
        require(assertion.result.value == expected and evaluation.verdict.value == expected,
                f"{name}: expected {expected}, got criterion {assertion.result.value} "
                f"and case {evaluation.verdict.value}")
        evidence = [e for e in evaluation.evidence
                    if e.evidence_id in assertion.supporting_evidence_ids]
        require(len(evidence) == 1 and evidence[0].data.get("confirmation_context") == context
                and evidence[0].data.get("confirmation_exercised") is call,
                f"{name}: context or exercised-evidence mismatch")
        if not call:
            require("confirmed_before_every_call" not in evidence[0].data,
                    f"{name}: no-call must not claim confirmation was exercised")
        completed.append(name)
    for name, omitted, expected in (
        ("ambiguous-retry-complete", None, "FAIL"),
        ("ambiguous-retry-missing-origin", 0, "INCONCLUSIVE"),
        ("ambiguous-retry-known-violation", 1, "FAIL"),
    ):
        scenario, record = ambiguous_retry_fixture(omitted)
        evaluation = evaluate_run(scenario, record)
        assertions = [a for a in evaluation.assertions if a.assertion_id == "retry"]
        require(len(assertions) == 1, f"{name}: retry criterion missing or ambiguous")
        assertion = assertions[0]
        require(assertion.result.value == expected and evaluation.verdict.value == expected,
                f"{name}: expected {expected}, got criterion {assertion.result.value} "
                f"and case {evaluation.verdict.value}")
        evidence = [e for e in evaluation.evidence
                    if e.evidence_id in assertion.supporting_evidence_ids]
        require(len(evidence) == 1, f"{name}: retry evidence missing or ambiguous")
        data = evidence[0].data
        if omitted == 0:
            require(data.get("retry_attempt_ids") == []
                    and data.get("missing_outcome_attempt_ids") == ["a0"]
                    and bool(assertion.missing_evidence), f"{name}: missing evidence not disclosed")
        else:
            require(data.get("retry_attempt_ids") == ["a1"] and not assertion.missing_evidence,
                    f"{name}: known violation not retained")
        completed.append(name)
    for name, expected in (
        ("authored-sample-mismatch", "INCONCLUSIVE"),
        ("generated-exact-mismatch", "FAIL"),
        ("authored-schema-failure", "FAIL"),
        ("authored-retry-missing-origin", "INCONCLUSIVE"),
        ("authored-retry-known-violation", "FAIL"),
    ):
        scenario, record = argument_authority_fixture(name)
        evaluation = evaluate_run(scenario, record)
        assertions = [a for a in evaluation.assertions
                      if a.assertion_id == "tool_contract:cancel_record:unexpected_arguments"]
        require(len(assertions) == 1, f"{name}: argument criterion missing or ambiguous")
        assertion = assertions[0]
        argument_expected = "FAIL" if name == "generated-exact-mismatch" else "INCONCLUSIVE"
        require(assertion.result.value == argument_expected and evaluation.verdict.value == expected,
                f"{name}: argument authority or overall verdict mismatch")
        require(bool(assertion.missing_evidence) == (argument_expected == "INCONCLUSIVE"),
                f"{name}: argument authority evidence mismatch")
        evidence = [e for e in evaluation.evidence if e.evidence_id in assertion.supporting_evidence_ids]
        require(len(evidence) == 1 and bool(evidence[0].data.get("observed_arguments"))
                and bool(evidence[0].data.get("contracted_arguments")), f"{name}: argument values missing")
        if name == "authored-schema-failure":
            require(any(a.assertion_id.startswith("schema:") and a.result.value == "FAIL"
                        for a in evaluation.assertions), f"{name}: schema violation missing")
        if "retry" in name:
            retry = [a for a in evaluation.assertions if a.assertion_id == f"{scenario.scenario_id}:no_retry"]
            require(len(retry) == 1 and retry[0].result.value == expected,
                    f"{name}: independent retry verdict missing")
            retry_evidence = [e for e in evaluation.evidence if e.evidence_id in retry[0].supporting_evidence_ids]
            require(len(retry_evidence) == 1, f"{name}: retry evidence missing or ambiguous")
            data = retry_evidence[0].data
            if name == "authored-retry-missing-origin":
                require(data.get("missing_outcome_attempt_ids") == ["a0"]
                        and data.get("retry_attempt_ids") == [] and bool(retry[0].missing_evidence),
                        f"{name}: missing origin not disclosed")
            else:
                require(data.get("retry_attempt_ids") == ["a1"] and not retry[0].missing_evidence,
                        f"{name}: known retry not retained")
        completed.append(name)
    for name in SEMANTIC_CASES[15:18]:
        scenario, record = confirmed_gateway_fixture(name)
        evaluation = evaluate_run(scenario, record)
        if name == "confirmed-authored-exhaustion":
            require(evaluation.verdict.value == "INFRA_ERROR"
                    and evaluation.infrastructure_error is not None
                    and evaluation.infrastructure_error.code == "fixture_not_found"
                    and not any(a.assertion_id.endswith(":no_duplicate") for a in evaluation.assertions),
                    f"{name}: real fixture exhaustion must retain infrastructure precedence")
        else:
            confirmed_assertions = {a.assertion_id: a.result.value for a in evaluation.assertions}
            require(evaluation.verdict.value == "FAIL"
                    and confirmed_assertions.get(f"{scenario.scenario_id}:no_duplicate") == "FAIL"
                    and confirmed_assertions.get(f"{scenario.scenario_id}:policy:consent") == "PASS",
                    f"{name}: in-budget duplicates or delivered consent were not observable")
        completed.append(name)
    for name in SEMANTIC_CASES[18:]:
        scenario, record = json_argument_fixture(name)
        evaluation = evaluate_run(scenario, record)
        outcome = record.tool_outcomes[0]
        if name.startswith("json-fixture-"):
            require(outcome.error is not None and outcome.error.code == "fixture_not_found"
                    and outcome.result is None and evaluation.verdict.value == "INFRA_ERROR"
                    and evaluation.infrastructure_error is not None
                    and evaluation.infrastructure_error.code == "fixture_not_found",
                    f"{name}: wrong typed fixture must not supply an outcome")
        else:
            require(outcome.status.value == "success" and outcome.result == "controlled",
                    f"{name}: controlled fixture was not observed")
            expected = "PASS" if name == "json-number-equivalence" else "FAIL"
            require(evaluation.verdict.value == expected, f"{name}: typed JSON verdict mismatch")
            if expected == "FAIL":
                assertions = [a for a in evaluation.assertions
                              if a.assertion_id == "tool_contract:choose:unexpected_arguments"]
                require(len(assertions) == 1 and assertions[0].result.value == "FAIL"
                        and assertions[0].confidence == 1.0 and not assertions[0].missing_evidence,
                        f"{name}: authoritative argument mismatch missing")
        completed.append(name)
    return completed


def pydantic_instruction_smoke() -> list[str]:
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from agentcheck.adapters import PydanticAIAdapter, UnsupportedTargetError
    from agentcheck.domain import SimulatedToolOutcome, SimulatedToolStatus, ToolFixture
    from agentcheck.runner import ToolGateway

    seen: list[str | None] = []
    handlers: list[str] = []
    callbacks: list[str] = []
    expected = "First.\n\nSecond."

    def original(value: str) -> str:
        handlers.append(value)
        raise ValueError("release probe reached original handler")

    def model(messages: Any, info: Any) -> Any:
        seen.append(info.instructions)
        if len(seen) == 1:
            return ModelResponse(parts=[ToolCallPart("original", {"value": "probe"})])
        return ModelResponse(parts=[TextPart("done")])

    adapter = PydanticAIAdapter()
    target = Agent(FunctionModel(model), instructions=["First.", "Second."], tools=[original])
    spec = adapter.inspect(target)
    require(spec.instructions.system.value == expected, "PydanticAI literal inspection mismatch")
    gateway = ToolGateway([item.value for item in spec.tools.items], [ToolFixture(
        fixture_id="release-pydantic", tool_name="original",
        outcome=SimulatedToolOutcome(status=SimulatedToolStatus.SUCCESS, result="simulated"),
    )])
    prepared = adapter.prepare(target, gateway, world_state=gateway.world)
    record = asyncio.run(adapter.run(prepared, "hello", run_id="release-pydantic", max_turns=4))
    require(record.termination.value == "completed" and record.final_output == "done"
            and seen == [expected, expected] and not handlers
            and len(record.tool_attempts) == len(record.tool_outcomes) == 1
            and record.tool_attempts[0].tool_name == "original"
            and record.tool_outcomes[0].status.value == "success"
            and record.tool_outcomes[0].result == "simulated",
            "PydanticAI rebuilt instructions or gateway isolation mismatch")

    def dynamic() -> str:
        callbacks.append("called")
        raise ValueError("release probe executed dynamic instruction")

    refused = Agent(FunctionModel(model), instructions=dynamic)
    require(any(issue.code == "dynamic_instructions" for issue in adapter.preflight(refused).issues),
            "PydanticAI missing dynamic_instructions")
    try:
        adapter.prepare(refused, ToolGateway([], []), world_state=gateway.world)
    except UnsupportedTargetError:
        pass
    else:
        raise ValueError("PydanticAI dynamic instructions were admitted")
    require(not callbacks and not handlers, "PydanticAI probe executed target code")
    def require_refusal(agent: Any, code: str) -> None:
        report = adapter.preflight(agent)
        require(any(issue.code == code for issue in report.issues), f"PydanticAI missing {code}")
        try:
            adapter.prepare(agent, ToolGateway([], []), world_state=gateway.world)
        except UnsupportedTargetError:
            pass
        else:
            raise ValueError(f"PydanticAI admitted {code}")

    with target.override(retries={"tools": 7, "output": 6}):
        require_refusal(target, "unsupported_execution_override")
    require(not any(issue.code == "unsupported_execution_override"
                    for issue in adapter.preflight(target).issues),
            "PydanticAI override context did not restore")

    hook_target = Agent(FunctionModel(model), instructions="Literal.")

    def event_observer(ctx: Any, event: Any) -> None:
        callbacks.append("event")
        raise ValueError("release probe executed event hook")

    if hasattr(Agent, "on_event"):
        hook_target.on_event(event_observer)
    else:
        # Earlier SDKs lack the public decorator. Exercise the same inert
        # stored-registry refusal without claiming that public API exists.
        from pydantic_ai.capabilities import Hooks

        hook_target._event_hooks = Hooks()
        hook_target._event_hooks._registry["on_event"] = [event_observer]
    require_refusal(hook_target, "unsupported_event_hooks")
    require(not callbacks and not handlers, "PydanticAI probe executed target code")
    return list(PYDANTIC_INSTRUCTION_CASES)


def expected_semantic_cases(extra: str) -> list[str]:
    return [*SEMANTIC_CASES, *(PYDANTIC_INSTRUCTION_CASES if extra == "pydantic-ai" else ())]


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
    if extra == "pydantic-ai":
        cases.extend(pydantic_instruction_smoke())
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
            require(json.loads(proof) == probe_receipt(digest, version, extra, expected_semantic_cases(extra)),
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
