# PydanticAI agents

AgentCheck supports an exact `pydantic_ai.Agent` target through the
`pydantic_ai` adapter. The adapter inspects the agent's declared function tools,
then builds a separate runtime agent whose tools invoke AgentCheck's
`ToolGateway`. The original target handlers remain on the source agent and are
never called during simulated evaluation.

## Install the verified version

Install the verified adapter extra from PyPI:

```bash
python -m pip install "agentcheck-ai[pydantic-ai]"
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

The adapter requires:

- an exact `pydantic_ai.Agent`;
- static instructions;
- ordinary function tools with JSON Schema;
- no output validators, a callable `validation_context`, target capabilities,
  event-stream handler, or external toolsets.

Those unsupported surfaces are executable target behavior that cannot be
reconstructed safely. Preflight names each one and refuses the run.

### Dependency injection / `RunContext[Deps]`

`deps_type` and tools that take `ctx: RunContext[Deps]` are supported. Real
public PydanticAI examples use this pattern heavily, so rejecting it outright
would leave most realistic agents unevaluable.

AgentCheck never constructs a real dependency object. `deps_type` is a static
type annotation the framework itself does not instantiate or validate at
runtime, so AgentCheck always passes its own fail-closed placeholder as
`deps=` when it drives a run. A tool's reconstructed AgentCheck-owned invoker
never reads `ctx.deps` — `ToolGateway` fixtures remain the only source of
simulated tool behavior — so the placeholder is never touched on the intended
path. If something unexpected ever reached it (an HTTP client, a database
session, credentials), attribute access raises immediately rather than
returning fabricated data or reaching a live object. `ctx` itself never
appears in the declared tool schema or a generated fixture: PydanticAI's own
schema generation already excludes it.

A callable `validation_context` is different and stays unsupported: unlike
`deps_type`, it is target code the framework itself invokes with a
`RunContext` to validate tool arguments and output, so it is rejected the
same way an output validator is. A non-callable `validation_context` value is
inert data and is accepted.

### Multi-agent patterns ("agent delegation" and "hand-off")

PydanticAI's own documentation names two multi-agent patterns, and neither is
a framework-level primitive the way the OpenAI Agents SDK's `Handoff` object
is — there is no distinct type, and no attribute on `Agent` that names either
pattern:

- **Agent delegation** is an ordinary `@agent.tool` function whose *body*
  happens to call `await other_agent.run(...)`. The sub-agent reference lives
  entirely inside source code AgentCheck already replaces and never executes
  — the delegating tool is indistinguishable from any other tool, and it is
  already safely unreachable under the same guarantee every other tool has.
  Nothing about it needs adapter support beyond what already exists, and
  building a "delegation topology" abstraction for it would be inventing
  structure the framework itself does not expose.
- **Programmatic hand-off** is application code — a loop, a
  human-in-the-loop prompt, arbitrary branching — calling one `Agent.run()`
  after another. There is no single `Agent` object representing the hand-off
  at all: it is two or more independent targets, each inspected and prepared
  on its own. AgentCheck evaluates one exported `pydantic_ai.Agent` per
  target; representing a hand-off would mean representing the *calling
  code's* control flow, which is exactly the kind of target-owned
  orchestration this adapter does not execute.

`describe_topology` returns `None` for every PydanticAI target for this
reason, not because topology detection is unfinished: there is no framework
structure to describe.

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
