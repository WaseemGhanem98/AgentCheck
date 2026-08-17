from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from agents import Agent, Model, WebSearchTool, function_tool
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseInputItem, TResponseStreamEvent
from agents.model_settings import ModelSettings
from agents.models.interface import ModelTracing
from agents.tool import Tool
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from pydantic import BaseModel

from agentcheck.adapters import OpenAIAgentsAdapter, UnsupportedTargetError
from agentcheck.domain import (
    ActionKind,
    CanonicalEventType,
    ResourceBudgets,
    RunTermination,
    SimulatedToolOutcome,
    SimulatedToolStatus,
    SourceKind,
    ToolFixture,
    ToolOutcomeStatus,
    Verdict,
    WorldStateEffect,
)
from agentcheck.evaluate import evaluate_run
from agentcheck.generate import build_account_support_suite
from agentcheck.inspect import (
    TargetLoadError,
    classify_tool,
    extract_capabilities,
    load_target,
    resolve_entrypoint,
)
from agentcheck.inspect.capabilities import JsonSchemaType
from agentcheck.runner import ToolGateway


def _tool_call(
    name: str, arguments: dict[str, Any], call_id: str
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        arguments=json.dumps(arguments),
        call_id=call_id,
        name=name,
        type="function_call",
        status="completed",
    )


def _message(text: str, item_id: str = "message-1") -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=item_id,
        content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


class ScriptedModel(Model):
    def __init__(
        self,
        outputs: list[list[Any]],
        *,
        raw_usage: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        self.outputs = outputs
        self.raw_usage = raw_usage
        self.request_id = request_id
        self.calls = 0
        self.seen_tools: list[list[Tool]] = []
        self.seen_inputs: list[str | list[TResponseInputItem]] = []

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
    ) -> ModelResponse:
        del (
            system_instructions,
            model_settings,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        self.seen_tools.append(list(tools))
        self.seen_inputs.append(input)
        output = self.outputs[self.calls]
        self.calls += 1
        return ModelResponse(
            output=output,
            usage=Usage(),
            response_id=None,
            request_id=self.request_id,
            raw_usage=self.raw_usage,
        )

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[TResponseStreamEvent]:
        del args, kwargs
        raise NotImplementedError


class RecordingGateway:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = {"ok": True} if result is None else result
        self.error = error
        self.calls: list[tuple[str, dict[str, Any], Any]] = []

    async def invoke(
        self, tool_name: str, arguments: dict[str, Any], world_state: Any
    ) -> Any:
        self.calls.append((tool_name, arguments, world_state))
        if self.error is not None:
            raise self.error
        return self.result


def test_inspect_extracts_explicit_sdk_contract() -> None:
    @function_tool
    def lookup_account(account_id: str) -> str:
        """Look up an account by its exact identifier."""

        return account_id

    agent = Agent(
        name="Account Support",
        instructions="Only use exact account identifiers.",
        tools=[lookup_account],
        model=ScriptedModel([[_message("unused")]]),
    )

    spec = OpenAIAgentsAdapter().inspect(agent, source="example/agent.py:agent")

    assert spec.identity.name.value == "Account Support"
    assert spec.identity.framework.value == "openai_agents"
    assert spec.instructions.system.value == "Only use exact account identifiers."
    assert spec.interface.input_modalities.value == ("text",)
    assert len(spec.tools.items) == 1
    assert spec.tools.items[0].value.name == "lookup_account"
    assert spec.tools.items[0].value.replaceable is True
    assert spec.tools.items[0].value.input_schema["required"] == ["account_id"]
    assert spec.provenance.target == "example/agent.py:agent"
    assert spec.identity.provider.value is None
    assert spec.identity.provider.authoritative is False
    assert spec.identity.model.value == "ScriptedModel"
    assert "handoff" in spec.observability.supported_event_types.value
    assert "guardrail" not in spec.observability.supported_event_types.value


def test_real_runner_uses_gateway_and_never_original_handler() -> None:
    tripwire = {"called": False}

    @function_tool
    def delete_account(account_id: str) -> str:
        """Permanently delete one account."""

        tripwire["called"] = True
        raise AssertionError("the original live handler was invoked")

    model = ScriptedModel(
        [
            [_tool_call("delete_account", {"account_id": "acct_123"}, "call-1")],
            [_message("The simulated deletion completed.")],
        ]
    )
    original_agent = Agent(
        name="Account Support",
        instructions="Use the account tools.",
        tools=[delete_account],
        model=model,
    )
    gateway = RecordingGateway(result={"deleted": True, "account_id": "acct_123"})
    world = {"accounts": {"acct_123": {"exists": True}}}
    adapter = OpenAIAgentsAdapter()

    prepared = adapter.prepare(original_agent, gateway, world_state=world)
    run = asyncio.run(
        adapter.run(
            prepared,
            "Delete acct_123",
            run_id="run-tripwire",
            scenario_id="delete-without-confirmation",
            max_turns=3,
        )
    )

    assert tripwire["called"] is False
    assert prepared.runtime_agent.tools[0] is not delete_account
    assert gateway.calls == [("delete_account", {"account_id": "acct_123"}, world)]
    assert model.calls == 2
    assert all(seen == prepared.runtime_agent.tools for seen in model.seen_tools)
    second_input = model.seen_inputs[1]
    assert isinstance(second_input, list)
    tool_outputs = [
        item
        for item in second_input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert len(tool_outputs) == 1
    assert json.loads(tool_outputs[0]["output"]) == {
        "account_id": "acct_123",
        "deleted": True,
    }
    assert run.termination == RunTermination.COMPLETED
    assert run.final_output == "The simulated deletion completed."
    assert len(run.tool_attempts) == 1
    assert run.tool_attempts[0].tool_name == "delete_account"
    assert run.tool_attempts[0].destructive is True
    assert run.tool_outcomes[0].status == ToolOutcomeStatus.SUCCESS
    assert run.usage.input_tokens is None
    assert run.usage.output_tokens is None
    assert [event.sequence for event in run.events] == list(range(len(run.events)))
    assert CanonicalEventType.TOOL_ATTEMPT in {event.event_type for event in run.events}
    assert CanonicalEventType.TOOL_RESULT in {event.event_type for event in run.events}


def test_real_tool_gateway_state_transition_is_normalized_into_run() -> None:
    tripwire = {"called": False}

    @function_tool
    def update_email(account_id: str, email: str) -> str:
        tripwire["called"] = True
        raise AssertionError("the original live handler was invoked")

    model = ScriptedModel(
        [
            [
                _tool_call(
                    "update_email",
                    {"account_id": "acct_123", "email": "new@example.com"},
                    "call-1",
                )
            ],
            [_message("The email was updated.")],
        ]
    )
    agent = Agent(
        name="Account Support",
        instructions="Use the account tools.",
        tools=[update_email],
        model=model,
    )
    adapter = OpenAIAgentsAdapter()
    spec = adapter.inspect(agent)
    gateway = ToolGateway(
        spec.tools.items,
        (
            ToolFixture(
                fixture_id="update-success",
                tool_name="update_email",
                arguments_match={
                    "account_id": "acct_123",
                    "email": "new@example.com",
                },
                outcome=SimulatedToolOutcome(
                    status=SimulatedToolStatus.SUCCESS,
                    result={"updated": True},
                    state_effects=(
                        WorldStateEffect(
                            path="accounts.acct_123.email",
                            before="old@example.com",
                            after="new@example.com",
                        ),
                    ),
                ),
            ),
        ),
        world={"accounts": {"acct_123": {"email": "old@example.com"}}},
        run_id="gateway-state",
    )
    prepared = adapter.prepare(agent, gateway, world_state=gateway.world)

    run = asyncio.run(
        adapter.run(prepared, "Update my email", run_id="run-state", max_turns=3)
    )

    assert tripwire["called"] is False
    assert run.termination == RunTermination.COMPLETED
    assert run.final_world_state["accounts"]["acct_123"]["email"] == "new@example.com"
    assert len(run.state_transitions) == 1
    assert run.state_transitions[0].attempt_id == run.tool_attempts[0].attempt_id
    assert run.tool_outcomes[0].state_transition_ids == (
        run.state_transitions[0].transition_id,
    )
    assert run.tool_outcomes[0].metadata["simulated"] is True


def test_inspect_derives_capabilities_from_schema_evidence() -> None:
    @function_tool
    def delete_account(account_id: str, reason: str | None = None) -> str:
        """Permanently delete one account after explicit confirmation."""

        return account_id

    agent = Agent(
        name="Account Support",
        instructions="Confirm before deleting.",
        tools=[delete_account],
        model=ScriptedModel([[_message("unused")]]),
    )

    spec = OpenAIAgentsAdapter().inspect(agent, source="example/agent.py:agent")

    (capability_property,) = spec.capabilities.items
    capability = capability_property.value
    assert capability.capability_id == "tool:delete_account"
    assert capability.action_kind is ActionKind.DELETE
    assert capability.state_changing is True
    assert capability.destructive is True
    assert capability_property.inferred is True
    assert capability_property.authoritative is False
    assert capability_property.confidence < 0.8
    assert capability_property.source.kind is SourceKind.TOOL_SCHEMA

    summaries = [item.summary for item in capability_property.evidence]
    assert any("never authoritative" in summary for summary in summaries)
    assert any("2 required and 0 optional parameter(s)" in summary for summary in summaries)
    assert any("Parameter 'account_id' is required" in summary for summary in summaries)
    assert [item.locator for item in capability_property.evidence][0] == (
        "tool:delete_account"
    )

    # A strict SDK schema lists every property as required and expresses an
    # optional Python argument as a nullable union, so the extractor reports the
    # declared contract rather than the source signature.
    (extracted,) = extract_capabilities([item.value for item in spec.tools.items])
    assert [item.name for item in extracted.arguments.required_parameters] == [
        "account_id",
        "reason",
    ]
    assert extracted.arguments.optional_parameters == ()
    reason = extracted.arguments.parameters[1]
    assert reason.types == (JsonSchemaType.NULL, JsonSchemaType.STRING)
    assert extracted.arguments.additional_properties_allowed is False


def test_inspect_records_an_unclassifiable_tool_as_unknown_rather_than_guessing() -> None:
    @function_tool
    def frobnicate(payload: str) -> str:
        """Perform an operation this adapter cannot classify."""

        return payload

    agent = Agent(
        name="Opaque",
        instructions="Do the thing.",
        tools=[frobnicate],
        model=ScriptedModel([[_message("unused")]]),
    )

    spec = OpenAIAgentsAdapter().inspect(agent, source="example/agent.py:agent")

    capability = spec.capabilities.items[0].value
    assert capability.action_kind is ActionKind.OTHER
    assert capability.state_changing is False
    assert capability.destructive is False
    assert [item.path for item in spec.unknowns] == ["capabilities.items[0].action_kind"]
    assert spec.unknowns[0].source.kind is SourceKind.UNKNOWN
    assert spec.unknowns[0].confidence == 0.0


def test_prepared_tool_risk_matches_the_extractor_classification() -> None:
    @function_tool
    def delete_account(account_id: str) -> str:
        """Delete one account."""

        return account_id

    @function_tool
    def lookup_account(account_id: str) -> str:
        """Look up one account."""

        return account_id

    agent = Agent(
        name="Account Support",
        instructions="Confirm before deleting.",
        tools=[delete_account, lookup_account],
        model=ScriptedModel([[_message("unused")]]),
    )

    prepared = OpenAIAgentsAdapter().prepare(agent, RecordingGateway())

    assert prepared.metadata["tool_risks"] == {
        "delete_account": (True, True),
        "lookup_account": (False, False),
    }
    assert prepared.metadata["tool_risks"]["delete_account"] == classify_tool(
        "delete_account"
    )[1:]


def test_unsupported_tool_fails_preflight_before_model_execution() -> None:
    model = ScriptedModel([[_message("must not run")]])
    agent = Agent(
        name="Unsafe Surface",
        instructions="Search the web.",
        tools=[WebSearchTool()],
        model=model,
    )
    adapter = OpenAIAgentsAdapter()

    report = adapter.preflight(agent)

    assert report.supported is False
    assert "unsupported_tool_type" in {issue.code for issue in report.issues}
    with pytest.raises(UnsupportedTargetError):
        adapter.prepare(agent, RecordingGateway())
    assert model.calls == 0


def test_structured_output_fails_closed_before_model_execution() -> None:
    class StructuredAnswer(BaseModel):
        answer: str

    model = ScriptedModel([[_message("must not run")]])
    agent = Agent(
        name="Structured",
        instructions="Answer.",
        output_type=StructuredAnswer,
        model=model,
    )

    report = OpenAIAgentsAdapter().preflight(agent)

    assert report.supported is False
    assert "structured_output" in {issue.code for issue in report.issues}
    with pytest.raises(UnsupportedTargetError):
        OpenAIAgentsAdapter().prepare(agent, RecordingGateway())
    assert model.calls == 0


def test_unknown_tool_call_is_blocked_and_evaluated_as_agent_failure() -> None:
    @function_tool
    def lookup_account(account_id: str) -> str:
        return account_id

    model = ScriptedModel(
        [[_tool_call("invented_admin_tool", {"account_id": "acct_123"}, "unknown-1")]]
    )
    adapter = OpenAIAgentsAdapter()
    prepared = adapter.prepare(
        Agent(
            name="Support",
            instructions="Look up accounts.",
            tools=[lookup_account],
            model=model,
        ),
        RecordingGateway(),
    )
    scenario = build_account_support_suite()[0]

    run = asyncio.run(
        adapter.run(
            prepared,
            scenario.conversation_turns,
            run_id="run-unknown-tool",
            scenario_id=scenario.scenario_id,
            max_turns=3,
        )
    )
    evaluation = evaluate_run(scenario, run)

    assert run.termination == RunTermination.COMPLETED
    assert [attempt.tool_name for attempt in run.tool_attempts] == [
        "invented_admin_tool"
    ]
    assert run.tool_outcomes[0].status == ToolOutcomeStatus.BLOCKED
    assert run.tool_outcomes[0].error is not None
    assert run.tool_outcomes[0].error.code == "unknown_tool"
    assert evaluation.verdict == Verdict.FAIL


def test_dynamic_and_advanced_function_tool_callbacks_fail_closed() -> None:
    @function_tool(is_enabled=lambda _context, _agent: True)
    def lookup_account(account_id: str) -> str:
        return account_id

    agent = Agent(
        name="Dynamic Tool",
        instructions=lambda _context, _agent: "Dynamic instructions",
        tools=[lookup_account],
        model=ScriptedModel([[_message("must not run")]]),
    )

    report = OpenAIAgentsAdapter().preflight(agent)

    codes = {issue.code for issue in report.issues}
    assert "dynamic_instructions" in codes
    assert "dynamic_tool_enablement" in codes


def test_schema_invalid_arguments_are_recorded_without_reaching_gateway() -> None:
    tripwire = {"called": False}

    @function_tool
    def update_email(account_id: str, email: str) -> str:
        tripwire["called"] = True
        return "live update"

    model = ScriptedModel(
        [
            [
                _tool_call(
                    "update_email",
                    {"account_id": 123, "email": "user@example.com"},
                    "call-invalid",
                )
            ],
            [_message("I could not update it.")],
        ]
    )
    gateway = RecordingGateway()
    adapter = OpenAIAgentsAdapter()
    prepared = adapter.prepare(
        Agent(
            name="Support",
            instructions="Update email addresses.",
            tools=[update_email],
            model=model,
        ),
        gateway,
    )

    run = asyncio.run(
        adapter.run(prepared, "Update it", run_id="run-invalid", max_turns=3)
    )

    assert tripwire["called"] is False
    assert gateway.calls == []
    assert run.termination == RunTermination.COMPLETED
    assert run.tool_outcomes[0].status == ToolOutcomeStatus.MALFORMED
    assert run.tool_outcomes[0].error is not None
    assert run.tool_outcomes[0].error.code == "invalid_tool_arguments"


def test_gateway_infrastructure_error_is_not_converted_to_model_output() -> None:
    @function_tool
    def lookup_account(account_id: str) -> str:
        return account_id

    model = ScriptedModel(
        [[_tool_call("lookup_account", {"account_id": "acct_123"}, "call-1")]]
    )
    adapter = OpenAIAgentsAdapter()
    prepared = adapter.prepare(
        Agent(
            name="Support",
            instructions="Look up the account.",
            tools=[lookup_account],
            model=model,
        ),
        RecordingGateway(error=RuntimeError("fixture database unavailable")),
    )

    run = asyncio.run(
        adapter.run(prepared, "Look it up", run_id="run-infra", max_turns=3)
    )

    assert model.calls == 1
    assert run.termination == RunTermination.ADAPTER_ERROR
    assert run.final_output is None
    assert len(run.tool_attempts) == 1
    assert run.tool_outcomes == ()
    assert any(event.event_type == CanonicalEventType.ERROR for event in run.events)


def test_raw_usage_is_preserved_only_when_provider_supplies_it() -> None:
    model = ScriptedModel(
        [[_message("done")]],
        raw_usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        request_id="request-123",
    )
    adapter = OpenAIAgentsAdapter()
    prepared = adapter.prepare(
        Agent(name="Usage", instructions="Answer.", tools=[], model=model),
        RecordingGateway(),
    )

    run = asyncio.run(adapter.run(prepared, "Hi", run_id="run-usage", max_turns=1))

    assert run.usage.input_tokens == 7
    assert run.usage.output_tokens == 3
    assert run.usage.total_tokens == 10
    assert run.usage.cost_usd is None
    assert run.provider_request_ids == ("request-123",)
    assert run.metadata["usage_unknown"] is False


@pytest.mark.parametrize(
    ("budgets", "raw_usage", "expected_termination"),
    (
        (
            ResourceBudgets(token_budget=5),
            {
                "input_tokens": 6,
                "output_tokens": 4,
                "total_tokens": 10,
            },
            RunTermination.TOKEN_BUDGET,
        ),
        (
            ResourceBudgets(cost_budget_usd=0.05),
            {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "cost_usd": 0.10,
            },
            RunTermination.COST_BUDGET,
        ),
    ),
)
def test_known_provider_usage_enforces_runtime_budgets(
    budgets: ResourceBudgets,
    raw_usage: dict[str, Any],
    expected_termination: RunTermination,
) -> None:
    model = ScriptedModel([[_message("Which account should I delete?")]], raw_usage=raw_usage)
    agent = Agent(name="Budgeted", instructions="Answer.", model=model)
    adapter = OpenAIAgentsAdapter()
    spec = adapter.inspect(agent)
    gateway = ToolGateway(spec.tools.items, (), budgets=budgets, run_id="usage-budget")
    prepared = adapter.prepare(agent, gateway)

    run = asyncio.run(
        adapter.run(prepared, "Hello", run_id="run-usage-budget", max_turns=3)
    )

    assert run.termination == expected_termination
    assert run.usage.total_tokens == raw_usage["total_tokens"]
    if "cost_usd" in raw_usage:
        assert run.usage.cost_usd == raw_usage["cost_usd"]


def test_schema_rejected_attempts_still_consume_tool_call_budget() -> None:
    @function_tool
    def update_email(account_id: str, email: str) -> str:
        return f"{account_id}:{email}"

    model = ScriptedModel(
        [
            [
                _tool_call(
                    "update_email",
                    {"account_id": 123, "email": "one@example.com"},
                    "invalid-1",
                ),
                _tool_call(
                    "update_email",
                    {"account_id": 456, "email": "two@example.com"},
                    "invalid-2",
                ),
            ],
            [_message("Both requests were rejected.")],
        ]
    )
    agent = Agent(
        name="Budgeted Tools",
        instructions="Update email addresses.",
        tools=[update_email],
        model=model,
    )
    adapter = OpenAIAgentsAdapter()
    spec = adapter.inspect(agent)
    gateway = ToolGateway(
        spec.tools.items,
        (),
        budgets=ResourceBudgets(max_tool_calls=1),
        run_id="tool-budget",
    )
    prepared = adapter.prepare(agent, gateway)

    run = asyncio.run(
        adapter.run(prepared, "Update both", run_id="run-tool-budget", max_turns=3)
    )

    assert run.termination == RunTermination.COMPLETED
    assert [outcome.status for outcome in run.tool_outcomes] == [
        ToolOutcomeStatus.MALFORMED,
        ToolOutcomeStatus.BLOCKED,
    ]
    assert run.tool_outcomes[1].error is not None
    assert run.tool_outcomes[1].error.code == "tool_calls_budget_exceeded"


def test_target_extractor_uses_agentcheck_entrypoint(tmp_path: Any) -> None:
    target = tmp_path / "target"
    target.mkdir()
    module = target / "support_agent.py"
    module.write_text(
        "from agents import Agent\n"
        "agent = Agent(name='Loaded Agent', instructions='Static instructions.')\n",
        encoding="utf-8",
    )
    (target / "agentcheck.json").write_text(
        json.dumps({"entrypoint": "support_agent.py:agent"}),
        encoding="utf-8",
    )

    module_path, attribute = resolve_entrypoint(target)
    loaded, source = load_target(target)

    assert module_path == module.resolve()
    assert attribute == "agent"
    assert loaded.name == "Loaded Agent"
    assert source == f"{module.resolve()}:agent"


def test_target_extractor_directory_entrypoint_cannot_escape(tmp_path: Any) -> None:
    target = tmp_path / "target"
    target.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("agent = object()\n", encoding="utf-8")
    (target / "agentcheck.json").write_text(
        json.dumps({"entrypoint": "../outside.py:agent"}),
        encoding="utf-8",
    )

    with pytest.raises(TargetLoadError, match="target directory"):
        resolve_entrypoint(target)
    with pytest.raises(TargetLoadError, match="target directory"):
        load_target(target)
