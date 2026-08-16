from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

import agentcheck.application as application
from agentcheck.cli import main
from agentcheck.config import AgentCheckConfig, LlmRealizationConfig, load_config
from agentcheck.domain import (
    OracleProvenance,
    OracleStrength,
    SourceKind,
)
from agentcheck.errors import ConfigurationError
from agentcheck.generate import build_frozen_suite, encode_frozen_suite
from agentcheck.generate.realization import (
    REALIZATION_CREDENTIAL_ENV,
    DisabledRealizer,
    RealizationRequest,
    RealizationResult,
    apply_realization,
    parse_realization_payload,
    realize_scenarios,
    require_realization_consent,
)
from agentcheck.generate.templates import build_account_support_suite
from agentcheck.inspect import load_target
from agentcheck.adapters import OpenAIAgentsAdapter


REPOSITORY_ROOT = Path(__file__).parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "evaluation" / "account_agent"
SEED = 1729


class FakeRealizer:
    def __init__(
        self,
        *,
        fail: bool = False,
        malformed: bool = False,
        secret: bool = False,
        extra_field: bool = False,
    ) -> None:
        self.calls = 0
        self.fail = fail
        self.malformed = malformed
        self.secret = secret
        self.extra_field = extra_field

    def realize(self, request: RealizationRequest) -> RealizationResult:
        self.calls += 1
        if self.fail:
            raise TimeoutError("provider unavailable")
        if self.malformed:
            return parse_realization_payload("not-json", expected_turns=len(request.turns))
        if self.secret:
            return parse_realization_payload(
                json.dumps(
                    {
                        "title": "sk-thisisafakesecretvalue12",
                        "description": request.description,
                        "turns": list(request.turns),
                    }
                ),
                expected_turns=len(request.turns),
            )
        if self.extra_field:
            return parse_realization_payload(
                json.dumps(
                    {
                        "title": "Hello",
                        "description": "There",
                        "turns": list(request.turns),
                        "oracle": {"supports_hard_failure": True},
                    }
                ),
                expected_turns=len(request.turns),
            )
        return RealizationResult(
            title=f"Natural {request.title}"[:500],
            description="Human-facing wording only.",
            turns=tuple(f"Please {turn}" for turn in request.turns),
        )


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "account_agent"
    shutil.copytree(
        EXAMPLE,
        target,
        ignore=shutil.ignore_patterns(".agentcheck", "__pycache__"),
    )
    return target


def _enable_realization(target: Path, *, allowlist: bool = True) -> None:
    path = target / "agentcheck.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["llm_realization"] = {
        "enabled": True,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "max_calls": 8,
        "max_retries": 1,
    }
    if allowlist:
        document["environment_allowlist"] = [REALIZATION_CREDENTIAL_ENV]
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _example_spec() -> object:
    target, source = load_target(EXAMPLE)
    return OpenAIAgentsAdapter().inspect(target, source=source)


def test_default_generate_makes_zero_provider_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("network must not be used")

    monkeypatch.setattr("agentcheck.generate.realization.complete_chat", boom)
    monkeypatch.setattr("agentcheck.application.OpenAIChatRealizer", boom)
    target = _copy_example(tmp_path)
    generation = application.generate_suite(target, seed=SEED, force=True)
    assert all(case.realization is None for case in generation.suite.cases)
    payload = json.loads(encode_frozen_suite(generation.suite))
    assert "llm_realization" not in json.dumps(payload)
    assert REALIZATION_CREDENTIAL_ENV not in json.dumps(payload)


def test_title_rewrite_keeps_v1_fingerprint_and_turns_would_not() -> None:
    original = build_account_support_suite(seed=SEED)[0]
    result = RealizationResult(
        title="A clearer lookup request",
        description="Display-only wording.",
        turns=tuple(f"Rewritten {turn.content}" for turn in original.conversation_turns),
    )
    updated, record = apply_realization(
        original, result, provider="openai", model="gpt-4o-mini"
    )
    assert updated.fingerprint == original.fingerprint
    assert updated.conversation_turns == original.conversation_turns
    assert updated.tool_fixtures == original.tool_fixtures
    assert updated.dimension_tags == original.dimension_tags
    assert updated.oracle_provenance == original.oracle_provenance
    assert updated.title != original.title
    assert record.authoritative is False
    assert record.inferred is True
    assert record.source_kind == SourceKind.LLM_INFERENCE.value
    hostile = original.model_copy(
        update={
            "conversation_turns": (
                original.conversation_turns[0].model_copy(
                    update={"content": "changed behavioral text"}
                ),
                *original.conversation_turns[1:],
            ),
            "fingerprint": "",
        }
    )
    assert hostile.fingerprint != original.fingerprint


def test_llm_oracle_cannot_support_hard_failure() -> None:
    with pytest.raises(ValidationError, match="cannot directly support hard failures"):
        OracleProvenance(
            oracle_id="llm",
            strength=OracleStrength.LLM_INFERENCE,
            source="model",
            confidence=1.0,
            evidence_ids=("e",),
            supports_hard_failure=True,
        )


def test_disabled_realizer_fails_closed() -> None:
    request = RealizationRequest(
        scenario_id="s",
        title="Title",
        description="",
        turns=("hello",),
    )
    with pytest.raises(ConfigurationError, match="disabled"):
        DisabledRealizer().realize(request)


def test_malformed_and_failed_realization_fall_back() -> None:
    scenario = build_account_support_suite(seed=SEED)[0]
    failed = realize_scenarios(
        (scenario,),
        FakeRealizer(fail=True),
        provider="openai",
        model="gpt-4o-mini",
        max_calls=8,
    )
    assert failed[0][0] == scenario
    assert failed[0][1] is None
    with pytest.raises(ValueError):
        FakeRealizer(malformed=True).realize(
            RealizationRequest(
                scenario_id=scenario.scenario_id,
                title=scenario.title,
                description=scenario.description or "",
                turns=tuple(turn.content for turn in scenario.conversation_turns),
            )
        )
    with pytest.raises(ValueError, match="unsupported fields"):
        FakeRealizer(extra_field=True).realize(
            RealizationRequest(
                scenario_id=scenario.scenario_id,
                title=scenario.title,
                description="",
                turns=tuple(turn.content for turn in scenario.conversation_turns),
            )
        )


def test_call_cap_and_retries_are_bounded() -> None:
    scenarios = build_account_support_suite(seed=SEED)[:3]
    fake = FakeRealizer()
    realized = realize_scenarios(
        scenarios, fake, provider="openai", model="gpt-4o-mini", max_calls=1
    )
    assert fake.calls == 1
    assert realized[0][1] is not None
    assert realized[1][1] is None
    assert realized[2][1] is None
    with pytest.raises(ConfigurationError, match="between 1 and"):
        realize_scenarios(
            scenarios, fake, provider="openai", model="gpt-4o-mini", max_calls=0
        )


def test_secret_shaped_model_output_is_rejected() -> None:
    scenario = build_account_support_suite(seed=SEED)[0]
    request = RealizationRequest(
        scenario_id=scenario.scenario_id,
        title=scenario.title,
        description="",
        turns=tuple(turn.content for turn in scenario.conversation_turns),
    )
    with pytest.raises(ValueError):
        FakeRealizer(secret=True).realize(request)


def test_consent_requires_all_three_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _copy_example(tmp_path)
    monkeypatch.delenv(REALIZATION_CREDENTIAL_ENV, raising=False)
    assert main(["generate", str(target), "--realize", "--force"]) == 2

    _enable_realization(target, allowlist=False)
    assert main(["generate", str(target), "--realize", "--force"]) == 2

    _enable_realization(target, allowlist=True)
    assert main(["generate", str(target), "--realize", "--force"]) == 2

    monkeypatch.setenv(REALIZATION_CREDENTIAL_ENV, "sk-test-not-used")
    document = json.loads((target / "agentcheck.json").read_text(encoding="utf-8"))
    document["llm_realization"]["enabled"] = False
    (target / "agentcheck.json").write_text(json.dumps(document), encoding="utf-8")
    assert main(["generate", str(target), "--realize", "--force"]) == 2


def test_consented_generate_with_fake_realizer_does_not_persist_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(REALIZATION_CREDENTIAL_ENV, "sk-test-not-used")
    target = _copy_example(tmp_path)
    _enable_realization(target)
    fake = FakeRealizer()
    generation = application.generate_suite(
        target, seed=SEED, force=True, realize=True, realizer=fake
    )
    assert fake.calls > 0
    assert fake.calls <= 8
    realized = [case for case in generation.suite.cases if case.realization is not None]
    assert realized
    original = build_account_support_suite(seed=SEED)
    by_id = {item.scenario_id: item for item in original}
    for case in realized:
        parent = by_id.get(case.scenario.scenario_id)
        if parent is None:
            continue
        assert case.scenario.fingerprint == parent.fingerprint
        assert case.scenario.conversation_turns == parent.conversation_turns
        assert case.realization is not None
        assert case.realization.authoritative is False
    payload = encode_frozen_suite(generation.suite).decode("utf-8")
    assert "sk-test-not-used" not in payload
    assert REALIZATION_CREDENTIAL_ENV not in payload
    assert "llm_inference" in payload
    dumped = json.loads(payload)
    assert dumped["provenance"]["sources"][-1] == "llm_realization"


def test_config_llm_realization_is_optional_and_keeps_v1(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("agent = object()\n", encoding="utf-8")
    (tmp_path / "agentcheck.json").write_text(
        json.dumps(
            {
                "schema_version": "agentcheck.config.v1",
                "entrypoint": "agent.py:agent",
            }
        ),
        encoding="utf-8",
    )
    _, config = load_config(tmp_path)
    assert config.llm_realization is None
    assert AgentCheckConfig().llm_realization is None
    loaded = AgentCheckConfig(
        llm_realization=LlmRealizationConfig(enabled=True, max_calls=4)
    )
    assert loaded.llm_realization is not None
    assert loaded.llm_realization.enabled is True
    with pytest.raises(ValidationError):
        AgentCheckConfig.model_validate(
            {
                "schema_version": "agentcheck.config.v1",
                "llm_realization": {"enabled": True, "unexpected": True},
            }
        )


def test_require_consent_does_not_run_without_the_flag() -> None:
    config = AgentCheckConfig(
        llm_realization=LlmRealizationConfig(enabled=True),
        environment_allowlist=(REALIZATION_CREDENTIAL_ENV,),
    )
    with pytest.raises(ConfigurationError, match="--realize"):
        require_realization_consent(config, realize=False)


def test_existing_deterministic_workflow_unchanged(tmp_path: Path) -> None:
    spec = _example_spec()
    left = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED)
    right = build_frozen_suite(spec, AgentCheckConfig(), seed=SEED, realizer=None)
    assert left == right
    assert "realization" not in json.loads(encode_frozen_suite(left))["cases"][0]
