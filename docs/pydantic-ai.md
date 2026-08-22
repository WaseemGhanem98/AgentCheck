# PydanticAI agents

AgentCheck supports an exact `pydantic_ai.Agent` target through the
`pydantic_ai` adapter. The adapter inspects the agent's declared function tools,
then builds a separate runtime agent whose tools invoke AgentCheck's
`ToolGateway`. The original target handlers remain on the source agent and are
never called during simulated evaluation.

## Install the verified version

AgentCheck is not published to PyPI yet. From an AgentCheck source checkout:

```bash
python -m pip install "agentcheck-ai[pydantic-ai] @ file://$PWD"
```

The extra installs `pydantic-ai-slim >=2.32,<2.33`. AgentCheck deliberately
supports only PydanticAI 2.32.x because inspection reads framework-private
attributes. A missing extra produces `framework_unavailable`; another minor
version produces `unsupported_sdk_version` with the expected and detected
versions. AgentCheck fails preflight instead of guessing at an unverified
object shape.

The distribution is `agentcheck-ai`, the Python import is `agentcheck`, and the
command is `agentcheck`. Do not install the unrelated PyPI distribution named
`agentcheck`.

## Export one target

Use the PydanticAI `Agent` you already have and export it from a Python file.
The smallest supported shape is:

```python
from pydantic_ai import Agent


def lookup_order(order_id: str) -> str:
    """Look up one order."""
    # Your production implementation stays here. AgentCheck replaces the
    # tool before a simulated run can reach this function.
    ...


agent = Agent(
    your_model,
    instructions="Help customers with their orders.",
    tools=[lookup_order],
    name="Order Support",
)
```

`your_model` is your existing PydanticAI model object. The worked example
uses a local `FunctionModel`, needs no API key, and is runnable without a
provider.

The V1 adapter requires:

- an exact `pydantic_ai.Agent`;
- static instructions;
- ordinary function tools with JSON Schema;
- no `RunContext` / `deps_type` dependency injection;
- no output validators, target capabilities, event-stream handler, or external
  toolsets.

Those unsupported surfaces are executable target behavior that cannot be
reconstructed safely. Preflight names each one and refuses the run.

## Configure and run

The target directory must exist before `agentcheck init`; the command writes
configuration but never creates a project directory or target source:

```bash
mkdir -p path/to/order-agent
agentcheck init path/to/order-agent \
  --adapter pydantic_ai \
  --entrypoint agent.py:agent
```

The relevant `agentcheck.json` fields are:

```json
{
  "schema_version": "agentcheck.config.v1",
  "adapter": "pydantic_ai",
  "entrypoint": "agent.py:agent",
  "suite": "account_support_v1",
  "allow_network": false,
  "controlled_model": false
}
```

Run the ordinary flow:

```bash
agentcheck inspect  path/to/order-agent
agentcheck fixtures init path/to/order-agent
# Replace REPLACE_ME values and add realistic user_request text.
agentcheck generate path/to/order-agent
agentcheck test     path/to/order-agent
agentcheck report   path/to/order-agent --latest
```

`inspect` imports trusted target code in a child process but does not run an
agent turn. `generate` freezes schema-boundary and compatible behavioral cases.
`test` runs each case in its own contained worker and prints the HTML report and
replay-manifest locations.

## Representative and prerequisite fixtures

`agentcheck-fixtures.json` is committed synthetic test data. Representative
arguments make generated calls realistic; `user_request` gives the model a
credible reason to act:

```json
{
  "schema_version": "agentcheck.fixtures.v1",
  "tools": {
    "cancel_order": {
      "arguments": {
        "order_id": "order_123",
        "reason": "duplicate order"
      },
      "user_request": "Cancel duplicate order order_123."
    }
  },
  "prerequisites": {
    "lookup_order": {
      "result": {
        "order_id": "order_123",
        "status": "pending"
      }
    }
  }
}
```

A prerequisite is a gating tool the agent may legitimately call before the
focal action. Its declared result is included as a single-use simulated fixture;
AgentCheck does not infer prerequisite relationships. If the agent calls a tool
without a matching focal or prerequisite fixture, calls an undeclared tool, or
supplies schema-invalid arguments, the scenario fails closed as `INFRA_ERROR`.
It is never reported as a behavioral `FAIL`.

Use synthetic records only. Credential-shaped fixture values are rejected, and
fixture packs are not a place for API keys or customer data.

## Offline ControlledModel

For an existing provider-backed PydanticAI target, enable:

```json
{
  "adapter": "pydantic_ai",
  "controlled_model": true
}
```

AgentCheck rebuilds the runtime agent with its deterministic local
`ControlledPydanticModel`. No provider request is made during the run, no
provider credential needs to be forwarded into the worker, and observable
model-request/model-response events still describe real calls to that local
replacement.

The replacement is intentionally neutral: it produces a deterministic text or
schema-shaped response and never chooses a tool. It can validate import,
inspection, reconstruction, output handling, reports, and non-action cases, but
a passing tool-action assertion may be vacuous because no tool was attempted.
The CLI explicitly reports action-path coverage.

To exercise model-to-tool behavior offline, keep `controlled_model` false and
make the target select its own local scripted model, as the
[credential-free worked example](../examples/evaluation/pydantic_agent/README.md)
does. Its run reports `Action paths exercised: 1/1`.

OpenAI Agents SDK follows the same configuration idea: set
`controlled_model` true to replace provider execution with AgentCheck's neutral
offline model. Install the separate verified extra
`agentcheck-ai[openai-agents]` (`openai-agents >=0.20,<0.21`). Custom agents are
different: AgentCheck cannot see or replace model calls owned by custom
orchestration, so `custom` plus `controlled_model` is explicitly rejected.

## Safety boundary

During simulated evaluation, accepted PydanticAI function tools are rebuilt
from their declarations with AgentCheck-owned invokers. Declared calls are
schema-checked, fixture-backed, and controlled by `ToolGateway`; unknown tools
and missing fixtures fail closed. Original target handlers do not execute.

The target is still trusted Python. Import-time code executes, and network
denial is not a general operating-system sandbox. Direct filesystem writes,
subprocess execution, or direct local-database access performed by imports or
other target orchestration are outside the declared-tool guarantee.

See the complete
[offline PydanticAI worked example](../examples/evaluation/pydantic_agent/README.md).
