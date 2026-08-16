"""Resolve, attach, and apply declared policy packs without a second evaluator."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from agentcheck.config import AgentCheckConfig, contained_path
from agentcheck.domain import (
    AgentProperty,
    AgentSpec,
    OracleProvenance,
    OracleStrength,
    PoliciesSpec,
    Policy,
    Scenario,
    SourceKind,
    SourceReference,
    SpecEvidence,
)
from agentcheck.errors import ConfigurationError

from .builtins import builtin_policy_pack_index
from .pack import PolicyPack, PolicyRule, PolicyRuleKind


_MAX_PACK_BYTES = 64 * 1024


class PolicyPackRegistry:
    """Built-in names plus contained in-target JSON documents."""

    def __init__(self, builtins: Mapping[str, PolicyPack] | None = None) -> None:
        self._builtins = dict(builtins or builtin_policy_pack_index())

    def get_builtin(self, pack_id: str) -> PolicyPack | None:
        return self._builtins.get(pack_id)

    def resolve(self, name: str, *, root: Path) -> PolicyPack:
        builtin = self._builtins.get(name)
        if builtin is not None:
            return builtin
        if "/" not in name.replace("\\", "/"):
            known = ", ".join(sorted(self._builtins))
            raise ConfigurationError(
                f"unknown policy pack {name!r}; known built-ins: {known}"
            )
        path = contained_path(root, name)
        return _load_pack_file(path)


def _load_pack_file(path: Path) -> PolicyPack:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfigurationError(f"unable to read policy pack {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read(_MAX_PACK_BYTES + 1)
    except OSError as exc:
        raise ConfigurationError(f"unable to read policy pack {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > _MAX_PACK_BYTES:
        raise ConfigurationError(
            f"{path.name} exceeds the {_MAX_PACK_BYTES} byte policy-pack limit"
        )
    try:
        return PolicyPack.model_validate_json(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ConfigurationError(f"invalid policy pack {path.name}: {exc}") from exc


def merge_policy_pack_names(
    config: AgentCheckConfig, extra: Sequence[str] | None = None
) -> tuple[str, ...]:
    names: list[str] = []
    for name in (*(config.policy_packs or ()), *(extra or ())):
        stripped = name.strip()
        if stripped and stripped not in names:
            names.append(stripped)
    return tuple(names)


def resolve_policy_packs(
    root: Path,
    config: AgentCheckConfig,
    extra: Sequence[str] | None = None,
    *,
    registry: PolicyPackRegistry | None = None,
) -> tuple[PolicyPack, ...]:
    resolver = registry or PolicyPackRegistry()
    packs: list[PolicyPack] = []
    seen: set[str] = set()
    for name in merge_policy_pack_names(config, extra):
        pack = resolver.resolve(name, root=root)
        if pack.pack_id in seen:
            continue
        seen.add(pack.pack_id)
        packs.append(pack)
    return tuple(packs)


def attach_declared_policies(spec: AgentSpec, packs: Sequence[PolicyPack]) -> AgentSpec:
    """Record declared packs on the spec without changing ``spec_id``."""

    if not packs:
        return spec
    items = list(spec.policies.items)
    existing = {item.value.policy_id for item in items}
    for pack in packs:
        if pack.pack_id in existing:
            continue
        items.append(
            AgentProperty(
                value=Policy(
                    policy_id=pack.pack_id,
                    description=pack.description,
                    version=pack.version,
                ),
                source=SourceReference(
                    kind=SourceKind.DECLARED_POLICY,
                    locator=f"policy_pack:{pack.pack_id}",
                    description=pack.title,
                ),
                confidence=1.0,
                evidence=(
                    SpecEvidence(
                        evidence_id=f"policy-pack:{pack.pack_id}",
                        summary=pack.title,
                        locator=f"policy_pack:{pack.pack_id}",
                    ),
                ),
                inferred=False,
                authoritative=True,
            )
        )
        existing.add(pack.pack_id)
    return spec.model_copy(update={"policies": PoliciesSpec(items=tuple(items))})


def policy_oracle(
    pack: PolicyPack, rule: PolicyRule, *, declared: bool
) -> OracleProvenance:
    if declared:
        return OracleProvenance(
            oracle_id=f"policy:{pack.pack_id}:{rule.rule_id}",
            strength=OracleStrength.VERSIONED_POLICY,
            source=f"declared policy pack {pack.pack_id}",
            confidence=1.0,
            evidence_ids=(f"policy-pack:{pack.pack_id}:{rule.rule_id}",),
            supports_hard_failure=True,
        )
    return OracleProvenance(
        oracle_id=f"policy:{pack.pack_id}:{rule.rule_id}:undeclared",
        strength=OracleStrength.VERSIONED_POLICY,
        source=f"undeclared policy pack {pack.pack_id}",
        confidence=0.5,
        evidence_ids=(f"policy-pack:{pack.pack_id}:{rule.rule_id}:undeclared",),
        supports_hard_failure=False,
    )


def _referenced_tools(data: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for group in (
        "tool_fixtures",
        "required_tool_behavior",
        "allowed_tool_behavior",
        "forbidden_tool_behavior",
    ):
        for item in data.get(group) or []:
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str):
                names.add(item["tool_name"])
    for item in data.get("trajectory_constraints") or []:
        if not isinstance(item, dict):
            continue
        tool_name = (item.get("parameters") or {}).get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            names.add(tool_name)
    return names


def _ensure_oracle(data: dict[str, Any], oracle: OracleProvenance) -> str:
    payload = json.loads(oracle.model_dump_json())
    for existing in data.get("oracle_provenance") or []:
        if isinstance(existing, dict) and existing.get("oracle_id") == oracle.oracle_id:
            return oracle.oracle_id
    data.setdefault("oracle_provenance", []).append(payload)
    return oracle.oracle_id


def _unique_criterion_id(data: Mapping[str, Any], candidate: str) -> str:
    existing: set[str] = set()
    for group in (
        "expected_postconditions",
        "required_tool_behavior",
        "allowed_tool_behavior",
        "forbidden_tool_behavior",
        "trajectory_constraints",
        "output_criteria",
    ):
        for item in data.get(group) or []:
            if isinstance(item, dict) and isinstance(item.get("criterion_id"), str):
                existing.add(item["criterion_id"])
    identifier = candidate
    suffix = 2
    while identifier in existing:
        identifier = f"{candidate}:{suffix}"
        suffix += 1
    return identifier


def _add_oracle_to_criterion(item: dict[str, Any], oracle_id: str) -> bool:
    ids = list(item.get("oracle_ids") or [])
    if oracle_id in ids:
        return False
    ids.append(oracle_id)
    item["oracle_ids"] = ids
    return True


def _apply_trajectory_rule(
    data: dict[str, Any],
    pack: PolicyPack,
    rule: PolicyRule,
    *,
    declared: bool,
) -> bool:
    tool_name = rule.tool_name
    if not tool_name or tool_name not in _referenced_tools(data):
        return False
    oracle_id = _ensure_oracle(data, policy_oracle(pack, rule, declared=declared))
    kind = rule.kind.value
    for item in data.get("trajectory_constraints") or []:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != kind:
            continue
        if (item.get("parameters") or {}).get("tool_name") != tool_name:
            continue
        return _add_oracle_to_criterion(item, oracle_id)
    data.setdefault("trajectory_constraints", []).append(
        {
            "criterion_id": _unique_criterion_id(
                data, f"{data['scenario_id']}:policy:{rule.rule_id}"
            ),
            "kind": kind,
            "description": rule.description,
            "parameters": {"tool_name": tool_name},
            "required": True,
            "oracle_ids": [oracle_id],
        }
    )
    return True


def _apply_output_rule(
    data: dict[str, Any],
    pack: PolicyPack,
    rule: PolicyRule,
    *,
    declared: bool,
) -> bool:
    oracle_id = _ensure_oracle(data, policy_oracle(pack, rule, declared=declared))
    kind = rule.kind.value
    for item in data.get("output_criteria") or []:
        if isinstance(item, dict) and item.get("kind") == kind:
            return _add_oracle_to_criterion(item, oracle_id)
    data.setdefault("output_criteria", []).append(
        {
            "criterion_id": _unique_criterion_id(
                data, f"{data['scenario_id']}:policy:{rule.rule_id}"
            ),
            "kind": kind,
            "description": rule.description,
            "parameters": {},
            "required": True,
            "oracle_ids": [oracle_id],
        }
    )
    return True


def apply_policy_packs(
    scenario: Scenario,
    packs: Sequence[PolicyPack],
    *,
    declared: bool,
) -> Scenario:
    """Attach pack rules to a scenario using existing evaluator kinds only."""

    if not packs:
        return scenario
    data = json.loads(scenario.model_dump_json())
    changed = False
    for pack in packs:
        for rule in pack.rules:
            if rule.kind is PolicyRuleKind.NO_FABRICATED_SUCCESS:
                changed = (
                    _apply_output_rule(data, pack, rule, declared=declared) or changed
                )
            else:
                changed = (
                    _apply_trajectory_rule(data, pack, rule, declared=declared)
                    or changed
                )
    if not changed:
        return scenario
    data["fingerprint"] = ""
    return Scenario.model_validate_json(json.dumps(data, ensure_ascii=False))


def apply_policy_packs_to_many(
    scenarios: Sequence[Scenario],
    packs: Sequence[PolicyPack],
    *,
    declared: bool,
) -> tuple[Scenario, ...]:
    if not packs:
        return tuple(scenarios)
    return tuple(
        apply_policy_packs(scenario, packs, declared=declared) for scenario in scenarios
    )


__all__ = [
    "PolicyPackRegistry",
    "apply_policy_packs",
    "apply_policy_packs_to_many",
    "attach_declared_policies",
    "merge_policy_pack_names",
    "policy_oracle",
    "resolve_policy_packs",
]
