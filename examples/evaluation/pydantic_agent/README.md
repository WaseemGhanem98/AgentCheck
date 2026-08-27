# Offline PydanticAI order lookup

This worked example is a complete PydanticAI onboarding path. It uses
PydanticAI's local `FunctionModel`, needs no API key, makes zero provider
requests, and performs one real model-to-tool decision so the action-path result
is not vacuous.

The exported `agent` owns one declared function tool. Its original
`lookup_order` handler is a raising tripwire: if the real handler is ever
reached, the evaluation stops. During evaluation AgentCheck rebuilds the tool
with an AgentCheck-owned invoker, and `ToolGateway` supplies the simulated
result from the frozen scenario.

## Install

Install AgentCheck and the verified PydanticAI adapter from PyPI:

```bash
python -m pip install "agentcheck-ai[pydantic-ai]"
```

This installs the verified PydanticAI range:
`pydantic-ai-slim >=2.32,<2.36`. A version outside that range fails preflight
with `unsupported_sdk_version` instead of being inspected approximately.

## Run the example

From the AgentCheck repository root:

```bash
agentcheck inspect  examples/evaluation/pydantic_agent
agentcheck generate examples/evaluation/pydantic_agent
agentcheck test     examples/evaluation/pydantic_agent
agentcheck report   examples/evaluation/pydantic_agent --latest
```

Expected highlights are `Preflight: supported`, four generated schema/action
cases, and:

```text
Action paths exercised: 1/1 (a tool was actually called)
Passed:        4
Infra errors:  0
```

The exact report location is printed at the end of the test run.

## Why ControlledModel is false here

The example keeps `"controlled_model": false` because its own local
`FunctionModel` deliberately chooses `lookup_order`. There is still no provider
request.

For an ordinary provider-backed PydanticAI target, setting
`"controlled_model": true` asks AgentCheck to substitute its deterministic
offline model before the run. That model is intentionally neutral and never
chooses tools, so it is useful for credential-free reconstruction and
output-path checks but does not exercise a tool-action path. See
[the PydanticAI guide](../../../docs/pydantic-ai.md) for the complete tradeoff.

## Fixtures and prerequisites

`agentcheck-fixtures.json` supplies both the representative `order_id` and the
human-shaped `user_request` that makes calling `lookup_order` appropriate.
Fixture values are committed test data; use synthetic records and never put
credentials or real customer data in them.

This target has no gating lookup, so its `prerequisites` object is empty. When
one tool must run before the focal action, declare that gating tool under
`prerequisites` with the simulated result it should return. Missing or invalid
focal/prerequisite fixtures fail as `INFRA_ERROR`, never behavioral `FAIL`.
