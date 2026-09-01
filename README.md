# AgentCheck

[![PyPI version](https://img.shields.io/pypi/v/agentcheck-ai.svg)](https://pypi.org/project/agentcheck-ai/)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://github.com/WaseemGhanem98/AgentCheck/blob/main/pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-2ea44f.svg)](https://github.com/WaseemGhanem98/AgentCheck/blob/main/LICENSE)

**An evidence authority for AI-agent behavior.**

You already test your code. AgentCheck tests the decisions your agent makes.

AgentCheck runs trusted local agents against behavioral scenarios and evaluates
which tools they call, in what order, and how they respond to confirmations,
failures, retries, and policy constraints. Declared tool actions are simulated;
the original declared handlers are not executed during evaluation.

[Demo](#demo) · [Install](#install) · [Quickstart](#credential-free-quickstart) ·
[Integrations](#supported-integrations) · [Safety](#safety-boundaries) ·
[Documentation](#documentation)

## Demo

https://github.com/user-attachments/assets/3ecdf66c-0aa7-45fe-a606-72cfb9d6d5bc

## Install

```bash
python -m pip install "agentcheck-ai==0.5.2"
```

The distribution is `agentcheck-ai`; the Python import and CLI are both
`agentcheck`. The unrelated `agentcheck` PyPI distribution is not this project.

Install a native SDK adapter when you need one:

```bash
python -m pip install "agentcheck-ai[openai-agents]==0.5.2"
# or
python -m pip install "agentcheck-ai[pydantic-ai]==0.5.2"
```

Custom Python agents use the base package.

## Credential-free quickstart

The versioned bundled example uses a local scripted model and raising tripwire
handlers. It needs no model key, makes no provider request, and fails if an
original declared handler is reached. These commands install AgentCheck from
PyPI; the repository checkout supplies only the example target:

```bash
git clone --branch v0.5.2 --depth 1 https://github.com/WaseemGhanem98/AgentCheck.git
cd AgentCheck
python -m venv .venv
. .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install "agentcheck-ai[openai-agents]==0.5.2"

agentcheck --version
agentcheck inspect examples/evaluation/account_agent
agentcheck generate examples/evaluation/account_agent --force
agentcheck test examples/evaluation/account_agent --no-store
```

`agentcheck --version` should print `agentcheck 0.5.2`. The example is designed
to produce behavioral findings; a non-zero test result is evidence to inspect,
not an installation failure.

## Use AgentCheck on your agent

For an existing OpenAI Agents SDK target exported as `agent` from `agent.py`:

```bash
cd my-agent
agentcheck init .
agentcheck inspect .
agentcheck generate .
agentcheck test .
```

- `inspect` understands the agent's declared tools, schemas, and instructions.
- `generate` creates a frozen behavioral test suite.
- `test` runs the agent against that suite and evaluates its behavior.

In CI, one command covers the release question:

```bash
agentcheck gate .
```

It runs the frozen suite, compares the result against a trusted baseline, and
returns a single status: `0` allow, `1` a behavioral failure is new, `2` the run
was not certifiable, `3` the suite could not decide. Failures a baseline already
accepts do not block. See [the CI gate](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/ci-gate.md).

A minimal GitHub Actions job is credential-free when the committed target uses
a local scripted or controlled model and simulated declared tools:

```yaml
permissions:
  contents: read

jobs:
  agentcheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          persist-credentials: false
      - uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97 # v7.0.0
        with:
          python-version: "3.12"
      - run: python -m pip install "agentcheck-ai[openai-agents]==0.5.2"
      - run: agentcheck gate path/to/target --baseline agentcheck-baseline.json --json
```

Commit the reviewed `agentcheck.json`, frozen suite, fixture data, and trusted
baseline that the gate consumes. Use the PydanticAI extra instead for a
PydanticAI target; custom targets need only the base package. The
[copyable workflow](https://github.com/WaseemGhanem98/AgentCheck/blob/main/.github/workflows/agentcheck-example.yml)
pins action SHAs and documents the fuller trust model.

### Gate exits and 0.5.2 fail-closed behavior

| Exit | Meaning | CI action |
|---|---|---|
| `0` | The current run was certifiable and no new authoritative failure was found. Without a trusted baseline, this is the weaker “every executed case passed” answer. | Allow |
| `1` | A behavioral failure is new against the baseline, or a run without a baseline contains a failure. | Block |
| `2` | The run is not certifiable: setup, infrastructure, fixture, source, suite, replay, or stored-evidence validation failed. | Block |
| `3` | Required evidence was inconclusive. | Block |

Version 0.5.2 deliberately rejects more incomplete evidence. An unedited
fixture placeholder or partially invalid frozen suite is refused; incomplete or
duplicate stored execution structure is not loadable; source drift and missing
or inconsistent replay evidence prevent a fresh run from being trusted; and a
trusted baseline can no longer upgrade a current `INCONCLUSIVE` result to
`PASS`. Missing or ambiguous action-path evidence is reported as unmeasured,
not credited as exercised. Fix the evidence problem and rerun—do not remap exit
`2` or `3` to success.

The controlled offline model can validate harness and schema behavior without a
provider, but it may decline every intended action. A green gate with zero
action paths exercised is evidence only for the cases that actually ran, not an
end-to-end compatibility or policy claim. Review the coverage and action-path
sections of the report before relying on the exit code.

The target directory must already exist. `generate` and `test` may inspect the
target again because each command independently validates current source instead
of trusting stale state. PydanticAI and Custom Python targets require an explicit
adapter and entrypoint; follow their guides under [Documentation](#documentation).

Representative `agentcheck test` output:

```text
Inspecting agent...
Inspection complete. ✓
Loading frozen suite... ✓ 4 scenarios

Running 4 scenarios in isolated workers...
[1/4] Confirmation before destructive action .... PASS
[2/4] Delete without confirmation ............... FAIL
[3/4] Retry after ambiguous timeout .............. FAIL
[4/4] Claims success after tool failure .......... FAIL
Finalizing report...
```

Each scenario ends as `PASS`, `FAIL`, `INCONCLUSIVE`, or `INFRA_ERROR`; harness
failures are never presented as behavioral failures or passes.

Generated suites also include fault cases — the tool errors, times out, or
returns an empty, unparseable, truncated or stale payload — so the suite asks
what the agent does when a tool does not cooperate, not only when it does.

Which tools get them depends on how the tool's side-effect risk was
established, with an explicit precedence: developer declaration, then genuine
framework metadata, then inference, then unknown. A [custom Python
agent](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/custom-agents.md) can state it directly on the tool
(`ToolDefinition(state_changing=True, destructive=True)`), and every adapter
can be told through `agentcheck.json`'s `tool_risk` block — a declared axis is
always authoritative. Neither the OpenAI Agents SDK nor PydanticAI carries
this information itself, so an undeclared axis is *inferred* from the tool's
name and description, reported as inferred with a confidence, and never
treated as authoritative anywhere a hard verdict depends on it.

Inference is conservative in one direction only: a tool it cannot classify is
left non-state-changing (`UNKNOWN`, not a confirmed safe) and receives **no
fault family**. Verb-shaped names such as `delete_account` or `cancel_order`,
and persistence names such as `write` or `save_draft`, produce an inferred
classification. Generic dispatch names such as `bash`, `execute_python`, or
`execute_command` remain `UNKNOWN` and receive no risk-scoped fault family
until declared. Inferred risk remains non-authoritative, so coverage reports
`risk_metadata_not_authoritative` rather than implying the tool was checked.
See [fault testing](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/fault-testing.md) and, to declare your own
contracts, [behavioral policies](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/behavioral-policies.md). Multiple tool
calls decided in one model response, and what AgentCheck can and cannot test
about that, are covered in
[concurrent tool decisions](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/concurrent-tool-decisions.md).

## What AgentCheck catches

Normal tests ask whether a function returned the expected value. AgentCheck asks
whether an agent took the right actions, in the right order, under failure and
safety constraints.

It can evaluate behaviors such as:

- destructive actions without confirmation;
- duplicate destructive actions;
- unsafe retries after ambiguous outcomes;
- fabricated success after a tool failure;
- unknown, undeclared, or schema-invalid tool calls;
- policy and action-sequencing violations;
- missing prerequisite behavior where the contract makes it observable.

AgentCheck evaluates execution behavior, not primarily whether the final answer
sounds good.

## How it works

```text
Trusted local agent
        │
        ▼
Adapter / CustomAgentProtocol
        │
        ▼
Generated or frozen scenario
        │
        ▼
Isolated child-process worker
        │ declared tool call
        ▼
ToolGateway ─────► fixtures, faults, simulated state
        │ observable trajectory
        ▼
Evaluator
        │
        ▼
PASS / FAIL / INCONCLUSIVE / INFRA_ERROR
```

Representative fixtures supply realistic arguments and simulated results.
Prerequisite fixtures cover legitimate gating calls before the action under
test—for example, `lookup_customer` before `refund_order`. Missing fixtures fail
closed as `INFRA_ERROR`; AgentCheck does not invent a plausible tool result.

## Supported integrations

| Integration | Install | Guide |
|---|---|---|
| OpenAI Agents SDK | `agentcheck-ai[openai-agents]` | [Worked example](https://github.com/WaseemGhanem98/AgentCheck/blob/main/examples/evaluation/account_agent/README.md) |
| PydanticAI | `agentcheck-ai[pydantic-ai]` | [Setup and offline evaluation](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/pydantic-ai.md) |
| Custom Python agents | `agentcheck-ai` | [Integration contract](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/custom-agents.md) |

The OpenAI Agents SDK native adapter supports SDK 0.20–0.22 only for exact
ordinary `agents.Agent` targets whose exact `FunctionTool` tools can be safely
replaced and observed through `ToolGateway`. SDK 0.22 `SandboxAgent` targets are
refused for behavioral evaluation: their runtime-materialized sandbox capability
surface is outside that replaceable/observed contract. The refusal happens at the
agent-type boundary; AgentCheck does not enumerate those capabilities.

The native adapters reject unsupported SDK versions rather than guessing. Custom
Python support is a lightweight integration contract: the target declares inert
tools and routes declared calls through the AgentCheck-supplied `ToolRuntime`. It
is not universal framework support.

## Safety boundaries

For declared tools routed through `ToolGateway`:

- Declared real tool handlers never execute during simulated evaluation.
- Inputs are schema-checked and configured fixtures control the result.
- Unknown tools, missing fixtures, invalid arguments, and exhausted budgets fail
  closed.
- Mutations affect only the scenario's simulated state.

Every scenario runs in a child process with a constrained environment and network
denied by default. These are containment controls for **trusted local code**.
Network denial is not a general operating-system sandbox. Target imports execute,
and direct filesystem writes, subprocess execution, or direct database access
from arbitrary Python orchestration are outside the declared-tool guarantee.

AgentCheck 0.5.2 does **not** guarantee hostile-code containment, full answer-key
isolation from every execution surface, or deterministic model execution.
Replay verifies source/configuration/scenario bindings and re-executes the same
harness inputs; it does not capture stochastic provider output or turn a model
call into deterministic replay.

Read [SECURITY.md](https://github.com/WaseemGhanem98/AgentCheck/blob/main/SECURITY.md) before evaluating a new target.

## Reports and artifacts

| Verdict | Meaning | Exit code |
|---|---|---|
| `PASS` | All required, evaluable assertions passed. | `0` |
| `FAIL` | At least one authoritative behavioral assertion failed. | `1` |
| `INCONCLUSIVE` | Available evidence could not support a decision. | `3` |
| `INFRA_ERROR` | Setup, containment, fixture, or harness execution failed. | `2` |

Runs write local artifacts under `.agentcheck/` by default: bounded terminal
diagnostics, versioned JSON/JSONL, an HTML report, an observable tool trace,
failed assertions, simulated state where available, and a replay manifest.
Review artifacts before sharing them; they may contain prompts, model output,
tool inputs, and absolute paths.

The `generate` command and run reports also show [declared behavioral
coverage](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/behavioral-coverage.md): which declared-tool success, failure,
and timeout requirements—and which explicitly represented retry, confirmation,
duplicate-action, and prerequisite contracts—the suite covers or leaves
missing. This is not a claim that every real-world behavior was observed.

Replay is a source-bound re-execution recipe. It verifies recorded source,
configuration, specification, and scenario bindings before running again. It
reproduces inputs and harness behavior; it does not capture provider output or
make stochastic model execution deterministic.

For CI, start from the [credential-free workflow example](https://github.com/WaseemGhanem98/AgentCheck/blob/main/.github/workflows/agentcheck-example.yml)
and read the [CI trust model](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/ci-trust-model.md).

## Documentation

- [OpenAI Agents SDK worked example](https://github.com/WaseemGhanem98/AgentCheck/blob/main/examples/evaluation/account_agent/README.md)
- [PydanticAI setup and offline evaluation](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/pydantic-ai.md)
- [Custom Python agent contract](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/custom-agents.md)
- [Validation evidence and claim boundaries](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/validation-evidence.md)
- [Portable target identity](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/portable-identity.md)
- [Decision stages and happens-before](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/behavioral-launch.md)
- [Declared behavioral coverage](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/behavioral-coverage.md)
- [Fault testing](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/fault-testing.md)
- [Behavioral policies](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/behavioral-policies.md)
- [The CI gate](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/ci-gate.md)
- [Behavioral regression comparison](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/behavioral-regression.md)
- [CI trust model](https://github.com/WaseemGhanem98/AgentCheck/blob/main/docs/ci-trust-model.md)
- [Changelog](https://github.com/WaseemGhanem98/AgentCheck/blob/main/CHANGELOG.md)
- [PyPI package](https://pypi.org/project/agentcheck-ai/)

## Contributing

External contributions are welcome. Fork the repository, create a focused branch,
and open a pull request; GitHub-hosted CI runs credential-free tests without
provider calls. The complete local suite is:

```bash
python -m pytest tests -q -n 2
```

See [CONTRIBUTING.md](https://github.com/WaseemGhanem98/AgentCheck/blob/main/CONTRIBUTING.md) for setup, focused tests, and the safety
invariants that changes must preserve.

## Security

Do not file vulnerabilities as public issues. Use GitHub Private Vulnerability
Reporting as described in [SECURITY.md](https://github.com/WaseemGhanem98/AgentCheck/blob/main/SECURITY.md). Never put credentials or
production data in configuration, fixture packs, frozen suites, or run artifacts.

## License

Copyright © 2026 Waseem Ghanem.

Current source is licensed under the [Apache License 2.0](https://github.com/WaseemGhanem98/AgentCheck/blob/main/LICENSE). AgentCheck
`0.1.0` and `0.1.1` were distributed under the MIT License; that license
continues to govern those release artifacts. See [NOTICE](https://github.com/WaseemGhanem98/AgentCheck/blob/main/NOTICE) for the
project's copyright notice.
