from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentcheck.config import AgentCheckConfig, LlmRealizationConfig
from agentcheck.domain import (
    ConversationRole,
    ConversationTurn,
    OracleProvenance,
    OracleStrength,
    OutputCriterion,
    OutputCriterionKind,
    PostconditionOperator,
    Scenario,
    StatePostcondition,
    ToolBehaviorConstraint,
    TrajectoryConstraint,
    TrajectoryConstraintKind,
)
from agentcheck.generate import build_account_support_suite
from agentcheck.runner import run_scenario_in_subprocess
from agentcheck.runner import orchestrator
from agentcheck.runner.worker_protocol import (
    WORKER_EXECUTION_INPUT_VERSION,
    WORKER_REQUEST_VERSION,
    WORKER_RUNTIME_CONFIG_VERSION,
    WorkerExecutionInput,
    WorkerRequest,
    WorkerRuntimeConfig,
)


def _oracle_only_scenario(sentinel: str) -> Scenario:
    oracle_id = "oracle-only"
    return Scenario(
        scenario_id=f"worker-oracle-separation-{sentinel}",
        title=f"Evaluator title {sentinel}",
        description=f"Evaluator description {sentinel}",
        conversation_turns=(
            ConversationTurn(
                turn_id="turn-1",
                role=ConversationRole.USER,
                content="Return a short acknowledgement.",
            ),
        ),
        expected_postconditions=(
            StatePostcondition(
                criterion_id="oracle-postcondition",
                path="oracle.only",
                operator=PostconditionOperator.EQUALS,
                expected=sentinel,
                oracle_ids=(oracle_id,),
            ),
        ),
        required_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id="oracle-required-tool",
                tool_name=f"required-{sentinel}",
                min_calls=1,
                max_calls=1,
                oracle_ids=(oracle_id,),
            ),
        ),
        allowed_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id="oracle-allowed-tool",
                tool_name=f"allowed-{sentinel}",
                max_calls=1,
                oracle_ids=(oracle_id,),
            ),
        ),
        forbidden_tool_behavior=(
            ToolBehaviorConstraint(
                criterion_id="oracle-forbidden-tool",
                tool_name=f"forbidden-{sentinel}",
                min_calls=0,
                max_calls=0,
                oracle_ids=(oracle_id,),
            ),
        ),
        trajectory_constraints=(
            TrajectoryConstraint(
                criterion_id="oracle-trajectory",
                kind=TrajectoryConstraintKind.OTHER,
                description=f"Evaluator trajectory {sentinel}",
                parameters={"oracle_sentinel": sentinel},
                oracle_ids=(oracle_id,),
            ),
        ),
        output_criteria=(
            OutputCriterion(
                criterion_id="oracle-output",
                kind=OutputCriterionKind.OTHER,
                description=f"Evaluator output {sentinel}",
                parameters={"oracle_sentinel": sentinel},
                oracle_ids=(oracle_id,),
            ),
        ),
        dimension_tags=(f"oracle:{sentinel}",),
        oracle_provenance=(
            OracleProvenance(
                oracle_id=oracle_id,
                strength=OracleStrength.EXPLICIT_INSTRUCTION,
                source=f"evaluator:{sentinel}",
                confidence=1.0,
                evidence_ids=(f"evidence:{sentinel}",),
            ),
        ),
        generation_seed=7,
    )


def _oracle_bearing_config(sentinel: str) -> AgentCheckConfig:
    return AgentCheckConfig(
        adapter="custom",
        entrypoint="agent.py:agent",
        seed=8_675_309,
        max_concurrency=7,
        environment_allowlist=(f"{sentinel}_ENV",),
        scenario_wall_clock_seconds=41.5,
        include_instructions_in_report=True,
        artifacts_directory=f"artifacts-{sentinel}",
        suite_path=f"frozen-{sentinel}.json",
        policy_packs=(f"policies/{sentinel}.json",),
        store_path=f"stores/{sentinel}.sqlite",
        max_cases=13,
        llm_realization=LlmRealizationConfig(enabled=True, model=sentinel),
    )


def test_run_request_has_only_runtime_and_scenario_execution_fields(
    tmp_path: Path,
) -> None:
    scenario_sentinel = "ORACLE_ONLY_SENTINEL_7F86A2"
    config_sentinel = "PARENT_CONFIG_SENTINEL_91C43D"
    scenario = _oracle_only_scenario(scenario_sentinel)
    config = _oracle_bearing_config(config_sentinel)

    request = orchestrator._worker_request(
        root=tmp_path,
        config=config,
        operation="run",
        scenario=scenario,
        run_id="worker-oracle-run",
    )
    payload = request.model_dump(mode="json")

    assert set(payload) == {
        "contract_version",
        "operation",
        "root",
        "runtime_config",
        "execution_input",
    }
    assert payload["contract_version"] == WORKER_REQUEST_VERSION
    runtime = payload["runtime_config"]
    assert isinstance(runtime, dict)
    assert set(runtime) == {
        "contract_version",
        "adapter",
        "entrypoint",
        "allow_network",
        "controlled_model",
        "network_allowlist",
        "tool_risk",
    }
    assert runtime["contract_version"] == WORKER_RUNTIME_CONFIG_VERSION
    assert runtime["adapter"] == "custom"
    assert runtime["entrypoint"] == "agent.py:agent"
    execution = payload["execution_input"]
    assert isinstance(execution, dict)
    assert set(execution) == {
        "contract_version",
        "run_id",
        "conversation_turns",
        "followup_turns",
        "initial_world_state",
        "tool_fixtures",
        "injected_faults",
        "resource_budgets",
    }
    assert execution["contract_version"] == WORKER_EXECUTION_INPUT_VERSION
    assert execution["run_id"] == "worker-oracle-run"
    serialized = json.dumps(payload, sort_keys=True)
    assert scenario_sentinel not in serialized
    assert config_sentinel not in serialized
    assert "8675309" not in serialized
    parent_only_config_fields = {
        "schema_version",
        "suite",
        "seed",
        "max_concurrency",
        "environment_allowlist",
        "scenario_wall_clock_seconds",
        "include_instructions_in_report",
        "artifacts_directory",
        "suite_path",
        "policy_packs",
        "store_path",
        "max_cases",
        "llm_realization",
        "python_executable",
    }
    assert parent_only_config_fields.isdisjoint(runtime)


def test_worker_execution_contract_rejects_oracle_fields_and_legacy_request() -> None:
    scenario = _oracle_only_scenario("ORACLE_ONLY_SENTINEL_7f86a2")
    execution = WorkerExecutionInput.from_scenario(
        scenario, run_id="worker-oracle-run"
    ).model_dump(mode="json")
    execution["expected_postconditions"] = []

    with pytest.raises(ValidationError, match="expected_postconditions"):
        WorkerExecutionInput.model_validate_json(json.dumps(execution))

    runtime = WorkerRuntimeConfig.from_config(AgentCheckConfig()).model_dump(
        mode="json"
    )
    runtime["suite_path"] = "answer-key.json"
    with pytest.raises(ValidationError, match="suite_path"):
        WorkerRuntimeConfig.model_validate_json(json.dumps(runtime))

    legacy = {
        "contract_version": "agentcheck.worker_request.v1",
        "operation": "run",
        "root": "/target",
        "config": AgentCheckConfig().model_dump(mode="json"),
        "scenario": scenario.model_dump(mode="json"),
        "run_id": "worker-oracle-run",
    }
    with pytest.raises(ValidationError):
        WorkerRequest.model_validate_json(json.dumps(legacy))


def test_execution_inputs_remain_matchable_to_importable_builtin_generator() -> None:
    """Record why request projection alone is not answer-key isolation."""

    suite = build_account_support_suite(seed=1729)
    scenario = suite[4]
    execution = WorkerExecutionInput.from_scenario(
        scenario, run_id="suite-run-case-4"
    )
    signature = execution.model_dump(
        mode="json", exclude={"contract_version", "run_id"}
    )

    matches = [
        candidate
        for candidate in suite
        if WorkerExecutionInput.from_scenario(
            candidate, run_id="candidate"
        ).model_dump(mode="json", exclude={"contract_version", "run_id"})
        == signature
    ]

    assert execution.run_id.endswith("-case-4")
    assert [candidate.scenario_id for candidate in matches] == [
        scenario.scenario_id
    ]
    assert matches[0].expected_postconditions == scenario.expected_postconditions
    assert matches[0].oracle_provenance == scenario.oracle_provenance


def test_ordinary_target_import_sees_no_request_file_or_parent_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "ORACLE_ONLY_SENTINEL_7F86A2"
    worker_temp = tmp_path / "worker-temp"
    worker_temp.mkdir()
    monkeypatch.setattr(orchestrator.tempfile, "tempdir", str(worker_temp))

    observed_path = tmp_path / "observed.json"
    (tmp_path / "agent.py").write_text(
        f"""
import json
import sys
from pathlib import Path

from agentcheck import TurnResult

OBSERVED = Path({str(observed_path)!r})
WORKER_TEMP = Path({str(worker_temp)!r})


def observe(stage):
    request_paths = sorted(WORKER_TEMP.glob("agentcheck-worker-*/request.json"))
    request_text = []
    for path in request_paths:
        try:
            request_text.append(path.read_text(encoding="utf-8"))
        except OSError:
            request_text.append("<unreadable>")
    existing = json.loads(OBSERVED.read_text(encoding="utf-8")) if OBSERVED.exists() else []
    existing.append({{
        "stage": stage,
        "argv": list(sys.argv),
        "request_paths": [str(path) for path in request_paths],
        "request_text": request_text,
    }})
    OBSERVED.write_text(json.dumps(existing), encoding="utf-8")


observe("import")


class ProbeAgent:
    name = "worker-oracle-probe"
    instructions = "Return an acknowledgement."
    tools = ()

    def start(self, message, tools):
        observe("start")
        return TurnResult(output="ack", state={{}})

    def resume(self, state, message, tools):
        return TurnResult(output="ack", state=state)


agent = ProbeAgent()
""".lstrip(),
        encoding="utf-8",
    )
    config = _oracle_bearing_config(sentinel)
    scenario = _oracle_only_scenario(sentinel)

    result = run_scenario_in_subprocess(
        tmp_path,
        config,
        scenario,
        "worker-oracle-run",
    )

    run = result.require_value()
    assert run.run_id == "worker-oracle-run"
    assert run.scenario_id == scenario.scenario_id
    observed = json.loads(observed_path.read_text(encoding="utf-8"))
    assert [item["stage"] for item in observed] == ["import", "start"]
    assert all(item["argv"] == ["-c"] for item in observed)
    assert all(item["request_paths"] == [] for item in observed)
    assert sentinel not in json.dumps(observed)
