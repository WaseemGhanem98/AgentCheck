# AgentCheck

[![PyPI version](https://img.shields.io/pypi/v/agentcheck-ai.svg)](https://pypi.org/project/agentcheck-ai/)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-2ea44f.svg)](LICENSE)

**Behavioral testing for AI agents.**

You already test your code. AgentCheck tests the decisions your agent makes.

AgentCheck runs trusted local agents against behavioral scenarios and evaluates
which tools they call, in what order, and how they respond to confirmations,
failures, retries, and policy constraints. Declared tool actions are simulated;
the original declared handlers are not executed during evaluation.

[Demo](#demo) · [Install](#install) · [Quickstart](#quickstart) ·
[Integrations](#supported-integrations) · [Safety](#safety-boundaries) ·
[Documentation](#documentation)

## Demo

https://github.com/user-attachments/assets/3ecdf66c-0aa7-45fe-a606-72cfb9d6d5bc

## Install

```bash
pip install agentcheck-ai
```

The distribution is `agentcheck-ai`; the Python import and CLI are both
`agentcheck`. The unrelated `agentcheck` PyPI distribution is not this project.

Install a native SDK adapter when you need one:

```bash
pip install "agentcheck-ai[openai-agents]"
# or
pip install "agentcheck-ai[pydantic-ai]"
```

Custom Python agents use the base package.

## Quickstart

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
| OpenAI Agents SDK | `agentcheck-ai[openai-agents]` | [Worked example](examples/evaluation/account_agent/README.md) |
| PydanticAI | `agentcheck-ai[pydantic-ai]` | [Setup and offline evaluation](docs/pydantic-ai.md) |
| Custom Python agents | `agentcheck-ai` | [Integration contract](docs/custom-agents.md) |

The native adapters are pinned to verified SDK minor versions and reject
unsupported versions rather than guessing. Custom Python support is a lightweight
integration contract: the target declares inert tools and routes declared calls
through the AgentCheck-supplied `ToolRuntime`. It is not universal framework
support.

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

Read [SECURITY.md](SECURITY.md) before evaluating a new target.

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
coverage](docs/behavioral-coverage.md): which declared-tool success, failure,
and timeout requirements—and which explicitly represented retry, confirmation,
duplicate-action, and prerequisite contracts—the suite covers or leaves
missing. This is not a claim that every real-world behavior was observed.

Replay is a source-bound re-execution recipe. It verifies recorded source,
configuration, specification, and scenario bindings before running again. It
reproduces inputs and harness behavior; it does not capture provider output or
make stochastic model execution deterministic.

For CI, start from the [credential-free workflow example](.github/workflows/agentcheck-example.yml)
and read the [CI trust model](docs/ci-trust-model.md).

## Documentation

- [OpenAI Agents SDK worked example](examples/evaluation/account_agent/README.md)
- [PydanticAI setup and offline evaluation](docs/pydantic-ai.md)
- [Custom Python agent contract](docs/custom-agents.md)
- [Validation evidence and claim boundaries](docs/validation-evidence.md)
- [Declared behavioral coverage](docs/behavioral-coverage.md)
- [Behavioral regression comparison](docs/behavioral-regression.md)
- [CI trust model](docs/ci-trust-model.md)
- [PyPI package](https://pypi.org/project/agentcheck-ai/)

## Contributing

External contributions are welcome. Fork the repository, create a focused branch,
and open a pull request; GitHub-hosted CI runs credential-free tests without
provider calls. The complete local suite is:

```bash
python -m pytest tests -q -n 2
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, focused tests, and the safety
invariants that changes must preserve.

## Security

Do not file vulnerabilities as public issues. Use GitHub Private Vulnerability
Reporting as described in [SECURITY.md](SECURITY.md). Never put credentials or
production data in configuration, fixture packs, frozen suites, or run artifacts.

## License

Current source is licensed under the [Apache License 2.0](LICENSE). AgentCheck
`0.1.0` and `0.1.1` were distributed under the MIT License; that license
continues to govern those release artifacts.
