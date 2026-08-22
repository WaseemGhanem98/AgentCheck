"""Credential-free end-to-end coverage for the PydanticAI worked example."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agentcheck.adapters import PydanticAIAdapter
from agentcheck.cli import main
from agentcheck.config import load_config
from agentcheck.domain import CanonicalRun, CaseEvaluation, Verdict
from agentcheck.fixtures import load_fixture_pack
from agentcheck.inspect import load_target


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "pydantic_agent"
PYDANTIC_DOC = REPOSITORY_ROOT / "docs" / "pydantic-ai.md"
README = REPOSITORY_ROOT / "README.md"


def test_example_contract_fixtures_and_docs_are_one_onboarding_path() -> None:
    root, config = load_config(EXAMPLE)
    target, source = load_target(root)
    spec = PydanticAIAdapter().inspect(target, source=source)
    fixtures = load_fixture_pack(root)

    assert config.adapter == "pydantic_ai"
    assert config.entrypoint == "agent.py:agent"
    assert config.controlled_model is False
    assert config.allow_network is False
    assert spec.identity.name.value == "Offline Order Support"
    assert [item.value.name for item in spec.tools.items] == ["lookup_order"]
    assert fixtures is not None
    assert fixtures.tools["lookup_order"].arguments == {"order_id": "order_123"}
    assert fixtures.tools["lookup_order"].user_request is not None
    assert fixtures.prerequisites == {}

    source_text = (EXAMPLE / "agent.py").read_text(encoding="utf-8")
    assert "FunctionModel(_script)" in source_text
    assert "original lookup_order handler must never run" in source_text

    docs = PYDANTIC_DOC.read_text(encoding="utf-8")
    example_docs = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    for text in (docs, example_docs):
        assert "pydantic-ai-slim >=2.32,<2.33" in text
        assert "agentcheck inspect" in text
        assert "agentcheck generate" in text
        assert "agentcheck test" in text
        assert "controlled_model" in text
        assert "prerequisite" in text.casefold()
        assert "no API key" in text
    assert "docs/pydantic-ai.md" in readme


def test_example_cli_flow_exercises_a_simulated_tool_without_provider_or_handler(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "pydantic-agent"
    shutil.copytree(EXAMPLE, target)

    source = target / "agent.py"
    instrumented = source.read_text(encoding="utf-8").replace(
        "from typing import Any\n",
        "from pathlib import Path\nfrom typing import Any\n",
    )
    handler_line = '    ORIGINAL_HANDLER_CALLS.append("lookup_order")'
    instrumented = instrumented.replace(
        handler_line,
        (
            '    Path(__file__).with_name("original-handler-executed").write_text(\n'
            '        "executed\\n", encoding="utf-8"\n'
            "    )\n"
            f"{handler_line}"
        ),
    )
    assert instrumented != source.read_text(encoding="utf-8")
    source.write_text(instrumented, encoding="utf-8")
    handler_probe = target / "original-handler-executed"

    assert main(["inspect", str(target)]) == 0
    assert main(["generate", str(target)]) == 0
    assert (
        main(
            [
                "test",
                str(target),
                "--run-id",
                "pydantic-example-test",
                "--no-store",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Preflight: supported" in output
    assert "Action paths exercised: 1/1" in output
    assert "Infra errors:  0" in output
    assert not handler_probe.exists()

    run_root = target / ".agentcheck" / "runs" / "pydantic-example-test"
    runs = tuple(
        CanonicalRun.model_validate_json(line)
        for line in (run_root / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    )
    evaluations = tuple(
        CaseEvaluation.model_validate_json(line)
        for line in (run_root / "evaluations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    )

    assert len(runs) == 4
    assert len(evaluations) == 4
    assert any(run.tool_attempts for run in runs)
    assert all(run.provider_request_ids == () for run in runs)
    assert all(
        outcome.metadata.get("simulated") is True
        for run in runs
        for outcome in run.tool_outcomes
    )
    assert all(item.verdict == Verdict.PASS for item in evaluations)
    assert all(item.infrastructure_error is None for item in evaluations)
    assert (run_root / "report.html").is_file()
    assert not handler_probe.exists()


def test_pydantic_guide_explains_controlled_model_limits_and_fail_closed_setup() -> None:
    docs = " ".join(PYDANTIC_DOC.read_text(encoding="utf-8").split())

    assert "pydantic-ai-slim >=2.32,<2.33" in docs
    assert "unsupported_sdk_version" in docs
    assert "ControlledPydanticModel" in docs
    assert "never chooses a tool" in docs
    assert "passing tool-action assertion may be vacuous" in docs
    assert "Action paths exercised: 1/1" in docs
    assert "matching focal or prerequisite fixture" in docs
    assert "fails closed" in docs
    assert "`INFRA_ERROR`" in docs
    assert "Original target handlers do not execute" in docs
    assert "outside the declared-tool guarantee" in docs
