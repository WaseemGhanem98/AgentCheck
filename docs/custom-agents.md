# Custom Python agents

AgentCheck supports OpenAI Agents SDK and PydanticAI natively, plus custom
Python agents through a lightweight integration contract. The custom path is
for orchestration code you own; it is not a generic adapter for arbitrary
framework objects and it is not a sandbox for arbitrary Python.

The contract is part of the base `agentcheck-ai` distribution and requires no
framework extra. A custom target supplies inert tool declarations and two
synchronous turn methods. AgentCheck supplies the only runtime for declared
tools during evaluation.

## Minimal contract

Run `agentcheck init TARGET --adapter custom` (TARGET must already
exist; `init` does not create it). If the configured entrypoint does not exist,
the command prints the following working skeleton without writing target source
for you. This block is tested against the CLI text so the two cannot drift.

<!-- custom-agent-cli-skeleton:start -->
```python
from agentcheck import ToolRuntime, TurnResult
from agentcheck.domain import ToolDefinition


class MyAgent:
    # Declarations only. AgentCheck is never handed a tool's implementation,
    # which is why it cannot run one.
    tools = (
        ToolDefinition(
            name="lookup_account",
            description="Read one account.",
            input_schema={
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
                "additionalProperties": False,
            },
        ),
    )

    def start(self, message: str, tools: ToolRuntime) -> TurnResult:
        outcome = tools.call("lookup_account", {"account_id": "A-1"})
        return TurnResult(output=f"Found {outcome.result}", state={})

    def resume(self, state, message: str, tools: ToolRuntime) -> TurnResult:
        return TurnResult(output="...", state=state)


agent = MyAgent()
```
<!-- custom-agent-cli-skeleton:end -->

The printed starter demonstrates the integration shape, not a finished agent
policy. Its `start()` method always calls `lookup_account` with `A-1`, so the
first generated evaluation can contain behavioral `FAIL` results until you
adapt the decision logic and add representative fixtures. Those results describe
the starter's behavior; they do not mean installation failed.

The configured object must expose:

- `tools`: a sequence of `ToolDefinition` values;
- `start(message, tools) -> TurnResult`: the opening user turn;
- `resume(state, message, tools) -> TurnResult`: each later scripted user turn.

Both methods are ordinarily synchronous, but may both instead be coroutine
functions -- `async def start(...)` / `async def resume(...)` -- if the loop
needs a real `await` (a genuine async provider client, for example). An async
pair is handed an `AsyncToolRuntime` whose `call()` is `async def` instead of
`def`; everything else about the contract is unchanged. `start` and `resume`
must agree: mixing one sync and one async method is refused at preflight as
`mismatched_turn_method_concurrency`, because AgentCheck picks one
`ToolRuntime` shape for the whole agent and cannot switch between turns.

`AsyncToolRuntime.call()` does not add support for genuinely concurrent tool
*dispatch*. It still runs the same synchronous gateway logic underneath, with
no `await` of its own, so one call always finishes before the next begins --
even if the agent issues them through `asyncio.gather()` or
`asyncio.create_task()`. That is a real, provable guarantee (the coroutine has
no internal yield point, so the event loop cannot pause it mid-call), but it
is not the same-stage "launch group" concurrency the OpenAI Agents and
PydanticAI adapters support: those derive decision order from an observed
model response listing several tool calls at once, and a custom agent's model
calls are never observed by AgentCheck at all, so there is no equivalent
evidence to derive that from. See `docs/concurrent-tool-decisions.md`. Real OS
threads calling into either `ToolRuntime` remain unsupported, exactly as
before.

`TurnResult.output` may be text or a JSON-compatible structured value.
`TurnResult.state` is opaque: AgentCheck passes the same object from `start()`
to `resume()` inside the worker and never serializes it. `metadata` is
normalized into JSON-safe data on the observed assistant-output event.

## Declare tools, never handlers

A `ToolDefinition` contains a name, description, JSON Schema, optional output
schema, and the `state_changing` / `destructive` risk flags. It contains no
callable field. Keep production implementations outside the object exported to
AgentCheck:

```python
DELETE_ACCOUNT = ToolDefinition(
    name="delete_account",
    description="Permanently delete one account after explicit confirmation.",
    input_schema={
        "type": "object",
        "properties": {"account_id": {"type": "string"}},
        "required": ["account_id"],
        "additionalProperties": False,
    },
    state_changing=True,
    destructive=True,
)
```

Do not attach a real function in another mapping or wrapper. AgentCheck accepts
only `ToolDefinition` values for this adapter, and `ToolGateway` independently
rejects live callables and handler-shaped fields.

During a turn, route the agent's requested action through the supplied runtime:

```python
outcome = tools.call("delete_account", {"account_id": "acct_123"})
if outcome.status.value == "success":
    return TurnResult(output="Deleted acct_123.")
return TurnResult(output="The deletion was not confirmed by the tool result.")
```

A simulated `success`, `error`, `timeout`, `empty`, `partial`, `stale`, or
`malformed` outcome is returned so the loop can react as it would to a real
tool. A refusal is raised instead: an unknown or undeclared tool, schema-invalid
arguments, a missing fixture, or an exhausted budget never receives an invented
result.

## Configuration and CLI flow

The minimal `agentcheck.json` is:

```json
{
  "schema_version": "agentcheck.config.v1",
  "adapter": "custom",
  "entrypoint": "agent.py:agent",
  "suite": "account_support_v1",
  "controlled_model": false
}
```

Then use the ordinary CLI path; there is no custom-only runner or fixture
format:

```bash
agentcheck inspect  path/to/target
agentcheck generate path/to/target
agentcheck test     path/to/target
agentcheck report   path/to/target --latest
```

Generation derives cases from the declared schemas and risk flags. Behavioural
rules are still expectations and are not invented merely because a tool looks
risky. To opt into rules scoped from declared `state_changing` and
`destructive` flags, add:

```json
{
  "policy_packs": ["derived_tool_risk_v1"]
}
```

The [worked custom example](../examples/evaluation/custom_agent/README.md) uses
that pack for a confirmation-gated `delete_account` action. It is deterministic,
offline, credential-free, and exercised by the test suite.

## Exact safety boundary

For the custom adapter, the declared-tool guarantee is exactly this:

- Declared real tool handlers are never supplied to AgentCheck, so AgentCheck
  cannot execute them.
- Every declared call made through `ToolRuntime` is schema-checked and simulated
  by the existing `ToolGateway`; only its simulated world changes.
- Unknown and undeclared tools fail closed. No plausible result is synthesized.
- Each scenario still runs in its own child process. The environment allowlist
  is empty and network egress is denied by default, with the guard installed
  before the target module is imported.
- Direct arbitrary Python side effects inside `start()`, `resume()`, their
  imports, or helpers are outside the declared-tool guarantee.

That last point is load-bearing. Custom orchestration is the target, so it has
to execute. A direct `os.remove(...)`, `subprocess.run(...)`, or local database
write in that code is not converted into a fixture-backed tool call. Process
isolation, credential removal, and socket-level egress denial contain common
escapes, but they do not prevent arbitrary local filesystem mutations. Use this
adapter only for trusted local code and keep side effects behind declared tools.

Network denial is not a general operating-system sandbox; it does not stop
those local effects.

## ControlledModel is unsupported

AgentCheck cannot replace a custom agent's model because the model call is
inside orchestration that AgentCheck only starts and resumes. Configuration
therefore fails before a worker starts instead of silently running without the
requested substitution. The actual configuration reason is:

```text
adapter 'custom' cannot substitute a controlled offline model, so
`controlled_model` must not be enabled for it: a custom agent owns its own
model calls and AgentCheck never sees them. Remove `controlled_model`, or have
the agent's own loop select a deterministic model for evaluation runs.
```

Keep `controlled_model` false or make the custom loop select its own local,
deterministic model for evaluation. The bundled example takes the latter idea
one step simpler: it uses deterministic Python control flow and makes no model
call at all.

## Agent turns are not model turns

AgentCheck observes calls to `start()` and `resume()` but cannot see how many
model requests happen inside either method. One agent turn may make zero, one,
or many model calls. Fabricating one `model_request` event per turn would turn
missing evidence into false telemetry, so the custom adapter records
`model_turns_observable: false` instead.

Consequences in evaluation are explicit:

- budget evidence records `model_turns: null`, never zero;
- the report adds a non-blocking `INCONCLUSIVE` model-turn-observability
  assertion while still evaluating measurable tool-call, wall-clock, token,
  and cost evidence as available;
- if a scenario explicitly requires a `MAX_MODEL_TURNS` trajectory constraint,
  that required assertion is `INCONCLUSIVE`, so the case cannot become `PASS`
  on an unobserved count;
- the runtime still limits the number of user/agent stages it drives, but that
  is a conversation bound, not evidence about provider model requests.

Replay has the same honest boundary as other adapters: it re-executes the pinned
source, scenario, fixtures, and harness behavior. It does not capture or make a
custom agent's model output deterministic.
