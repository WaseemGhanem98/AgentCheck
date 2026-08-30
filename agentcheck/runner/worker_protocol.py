"""Strict parent/worker transport contracts.

The child receives explicit allowlists for runtime configuration and scenario
execution inputs instead of the parent evaluator's complete objects. This
prevents new parent-side fields from silently crossing the request boundary;
it is not a claim that same-process target code or target-root files are
isolated from a hostile target.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from agentcheck.config import AgentCheckConfig, ToolRiskDeclaration
from agentcheck.domain.base import ContractModel, JsonObject
from agentcheck.domain.scenario import (
    MAX_FOLLOWUP_TURNS,
    ConversationRole,
    ConversationTurn,
    InjectedFault,
    ResourceBudgets,
    Scenario,
    ToolFixture,
)


WORKER_REQUEST_VERSION: Literal["agentcheck.worker_request.v2"] = (
    "agentcheck.worker_request.v2"
)
WORKER_EXECUTION_INPUT_VERSION: Literal["agentcheck.worker_execution_input.v1"] = (
    "agentcheck.worker_execution_input.v1"
)
WORKER_RUNTIME_CONFIG_VERSION: Literal["agentcheck.worker_runtime_config.v1"] = (
    "agentcheck.worker_runtime_config.v1"
)
WORKER_RESPONSE_VERSION: Literal["agentcheck.worker_response.v1"] = (
    "agentcheck.worker_response.v1"
)


class WorkerExecutionInput(ContractModel):
    """Only scenario material required to execute one child-process run."""

    contract_version: Literal["agentcheck.worker_execution_input.v1"] = (
        WORKER_EXECUTION_INPUT_VERSION
    )
    run_id: str = Field(min_length=1, max_length=200)
    conversation_turns: tuple[ConversationTurn, ...] = Field(min_length=1)
    followup_turns: tuple[ConversationTurn, ...] = Field(
        default=(), max_length=MAX_FOLLOWUP_TURNS
    )
    initial_world_state: JsonObject = Field(default_factory=dict)
    tool_fixtures: tuple[ToolFixture, ...] = ()
    injected_faults: tuple[InjectedFault, ...] = ()
    resource_budgets: ResourceBudgets = Field(default_factory=ResourceBudgets)

    @classmethod
    def from_scenario(
        cls, scenario: Scenario, *, run_id: str
    ) -> "WorkerExecutionInput":
        """Project an evaluated Scenario onto its execution-only allowlist."""

        return cls(
            run_id=run_id,
            conversation_turns=scenario.conversation_turns,
            followup_turns=scenario.followup_turns,
            initial_world_state=scenario.initial_world_state,
            tool_fixtures=scenario.tool_fixtures,
            injected_faults=scenario.injected_faults,
            resource_budgets=scenario.resource_budgets,
        )

    @model_validator(mode="after")
    def validate_turn_sequence(self) -> "WorkerExecutionInput":
        non_user = sorted(
            {
                turn.role.value
                for turn in self.followup_turns
                if turn.role != ConversationRole.USER
            }
        )
        if non_user:
            raise ValueError(
                "worker follow-up turns must be user turns; found "
                f"{', '.join(non_user)}"
            )
        turn_ids = [
            turn.turn_id for turn in (*self.conversation_turns, *self.followup_turns)
        ]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError(
                "worker conversation and follow-up turn IDs must be unique"
            )
        return self


class WorkerRuntimeConfig(ContractModel):
    """Only configuration fields consumed by the child runtime.

    Selection, generation, suite/replay lookup, artifact, store, provider-
    realization, concurrency, timeout, environment construction, and Python-
    executable settings remain parent-side. This explicit projection keeps a
    future :class:`AgentCheckConfig` field from silently crossing into the
    target process.
    """

    contract_version: Literal["agentcheck.worker_runtime_config.v1"] = (
        WORKER_RUNTIME_CONFIG_VERSION
    )
    adapter: Literal["openai_agents", "pydantic_ai", "custom"]
    entrypoint: str = Field(min_length=1, max_length=16_000)
    allow_network: bool = False
    controlled_model: bool = False
    network_allowlist: tuple[str, ...] = ()
    tool_risk: dict[str, ToolRiskDeclaration] | None = Field(
        default=None, max_length=200
    )

    @classmethod
    def from_config(cls, config: AgentCheckConfig) -> "WorkerRuntimeConfig":
        """Project the validated application config onto child-only fields."""

        return cls(
            adapter=config.adapter,
            entrypoint=config.entrypoint,
            allow_network=config.allow_network,
            controlled_model=config.controlled_model,
            network_allowlist=config.network_allowlist,
            tool_risk=config.tool_risk,
        )


class WorkerRequest(ContractModel):
    """One complete, strict request accepted by the worker entrypoint."""

    contract_version: Literal["agentcheck.worker_request.v2"] = WORKER_REQUEST_VERSION
    operation: Literal["inspect", "run"]
    root: str = Field(min_length=1, max_length=16_000)
    runtime_config: WorkerRuntimeConfig
    execution_input: WorkerExecutionInput | None = None

    @model_validator(mode="after")
    def validate_operation_input(self) -> "WorkerRequest":
        if self.operation == "run" and self.execution_input is None:
            raise ValueError("run worker requests require execution_input")
        if self.operation == "inspect" and self.execution_input is not None:
            raise ValueError("inspect worker requests cannot include execution_input")
        return self


__all__ = [
    "WORKER_EXECUTION_INPUT_VERSION",
    "WORKER_REQUEST_VERSION",
    "WORKER_RUNTIME_CONFIG_VERSION",
    "WORKER_RESPONSE_VERSION",
    "WorkerExecutionInput",
    "WorkerRequest",
    "WorkerRuntimeConfig",
]
