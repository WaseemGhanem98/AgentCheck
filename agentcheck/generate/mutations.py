"""Deterministic workflow mutations of validated scenarios.

Mutations recombine controlled fixture and dialogue data.  They do not rewrite
natural language, synthesize credentials, or invent business rules.  Lineage is
recorded beside the scenario in the frozen-suite document so
``agentcheck.scenario.v1`` fingerprints stay unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

from pydantic import Field

from agentcheck.domain import (
    ContractModel,
    ConversationRole,
    JsonObject,
    OutputCriterionKind,
    PostconditionOperator,
    Scenario,
    TrajectoryConstraintKind,
    canonical_hash,
)


MUTATION_CONTRACT_VERSION: Literal["agentcheck.scenario_mutation.v1"] = (
    "agentcheck.scenario_mutation.v1"
)
GENERATOR_NAME = "agentcheck.generate.mutations"

DEFAULT_MAX_MUTATIONS = 24
MAX_MUTATIONS_PER_SUITE = 48
MAX_MUTATIONS_PER_PARENT = 5
MAX_CONVERSATION_TURNS = 12
_MAX_SCENARIO_ID = 150
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TOKEN_BOUNDARY = r"(?<![A-Za-z0-9_]){token}(?![A-Za-z0-9_])"
_OTHER_ACCOUNT_ID = "acct_agentcheck_other"
_UNRELATED_TURN = (
    "This is an unrelated AgentCheck control turn; ignore it for the original request."
)
_CLARIFY_TEXT = "which account"

DEFERRED_MUTATION_REASONS = (
    "delay_confirmation: delaying confirmation requires inserting or rewriting "
    "dialogue; deferred to slice 10",
)


class MutationKind(str, Enum):
    """Ordered so truncation is deterministic and reviewable."""

    WITHHOLD_CONFIRMATION = "withhold_confirmation"
    DUPLICATE_REQUEST = "duplicate_request"
    AMBIGUOUS_IDENTIFIER = "ambiguous_identifier"
    REORDER_DIALOGUE = "reorder_dialogue"
    INTERLEAVE_UNRELATED_TURN = "interleave_unrelated_turn"


class ScenarioMutation(ContractModel):
    """Library record of one structural mutation. Lineage on the frozen suite."""

    schema_version: Literal["agentcheck.scenario_mutation.v1"] = (
        MUTATION_CONTRACT_VERSION
    )
    kind: MutationKind
    parent_scenario_id: str = Field(min_length=1, max_length=200)
    parent_fingerprint: str = Field(min_length=1, max_length=200)
    resulting_scenario_id: str = Field(min_length=1, max_length=200)
    rationale: str = Field(min_length=1, max_length=2_000)
    parameters: JsonObject = Field(default_factory=dict)
    generator: str = Field(default=GENERATOR_NAME, min_length=1, max_length=200)


@dataclass(frozen=True, slots=True)
class GeneratedMutation:
    mutation: ScenarioMutation
    scenario: Scenario


def _slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.casefold()).strip("-") or "value"


def _mutation_scenario_id(kind: MutationKind, parent_id: str) -> str:
    candidate = f"mut-{_slug(kind.value)}-{_slug(parent_id)}"
    if len(candidate) <= _MAX_SCENARIO_ID:
        return candidate
    digest = canonical_hash(candidate).split(":", 1)[1][:16]
    return f"{candidate[: _MAX_SCENARIO_ID - 17]}-{digest}"


def _dump(scenario: Scenario) -> dict[str, Any]:
    return scenario.model_dump(mode="json")


def _load(data: Mapping[str, Any]) -> Scenario:
    payload = dict(data)
    payload["fingerprint"] = ""
    return Scenario.model_validate_json(json.dumps(payload, ensure_ascii=False))


def _rewrite_prefixed_id(value: str, parent_id: str, new_id: str) -> str:
    if value == parent_id:
        return new_id
    prefix = f"{parent_id}:"
    if value.startswith(prefix):
        return f"{new_id}:{value[len(prefix):]}"
    return value


def _rewrite_ids(value: Any, parent_id: str, new_id: str, *, key: str | None = None) -> Any:
    identity_keys = {
        "scenario_id",
        "oracle_id",
        "criterion_id",
        "fixture_id",
        "fault_id",
        "oracle_ids",
        "evidence_ids",
    }
    if isinstance(value, str) and key in identity_keys:
        return _rewrite_prefixed_id(value, parent_id, new_id)
    if isinstance(value, list):
        return [
            _rewrite_ids(item, parent_id, new_id, key=key if key in identity_keys else None)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            child_key: _rewrite_ids(item, parent_id, new_id, key=child_key)
            for child_key, item in value.items()
        }
    return value


def _inherit_oracles(data: dict[str, Any], *, parent_id: str, kind: MutationKind) -> None:
    marker = f"mutation:{parent_id}:{kind.value}"
    for oracle in data.get("oracle_provenance", []):
        if not isinstance(oracle, dict):
            continue
        evidence = list(oracle.get("evidence_ids") or [])
        if marker not in evidence:
            evidence.append(marker)
        oracle["evidence_ids"] = evidence


def _renumber_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numbered: list[dict[str, Any]] = []
    for index, turn in enumerate(turns, 1):
        item = dict(turn)
        item["turn_id"] = f"turn-{index}"
        numbered.append(item)
    return numbered


def _user_indexes(turns: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        index
        for index, turn in enumerate(turns)
        if turn.get("role") == ConversationRole.USER.value
    ]


def _has_evaluable_criteria(data: Mapping[str, Any]) -> bool:
    return any(
        data.get(key)
        for key in (
            "expected_postconditions",
            "required_tool_behavior",
            "forbidden_tool_behavior",
            "trajectory_constraints",
            "output_criteria",
        )
    )


def _accounts(data: Mapping[str, Any]) -> dict[str, Any] | None:
    state = data.get("initial_world_state")
    if not isinstance(state, dict):
        return None
    accounts = state.get("accounts")
    if not isinstance(accounts, dict) or not accounts:
        return None
    if any(not isinstance(account_id, str) or not account_id for account_id in accounts):
        return None
    return dict(accounts)


def _conversation_text(turns: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(str(turn.get("content") or "") for turn in turns)


def _token_in_text(text: str, token: str) -> bool:
    return re.search(_TOKEN_BOUNDARY.format(token=re.escape(token)), text) is not None


def _replace_token(text: str, token: str, replacement: str) -> str:
    return re.sub(_TOKEN_BOUNDARY.format(token=re.escape(token)), replacement, text)


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


def _oracle_ids(data: Mapping[str, Any]) -> list[str]:
    oracles = data.get("oracle_provenance") or []
    ids = [
        oracle["oracle_id"]
        for oracle in oracles
        if isinstance(oracle, dict) and isinstance(oracle.get("oracle_id"), str)
    ]
    return ids or [f"{data['scenario_id']}:oracle"]


def _flip_equals_postconditions(data: dict[str, Any], account_ids: Sequence[str]) -> None:
    prefixes = tuple(f"accounts.{account_id}" for account_id in account_ids)
    postconditions: list[dict[str, Any]] = []
    for item in data.get("expected_postconditions") or []:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if (
            item.get("operator") == PostconditionOperator.EQUALS.value
            and isinstance(path, str)
            and any(path == prefix or path.startswith(prefix + ".") for prefix in prefixes)
        ):
            updated = dict(item)
            updated["operator"] = PostconditionOperator.UNCHANGED.value
            updated["expected"] = None
            postconditions.append(updated)
        else:
            postconditions.append(item)
    data["expected_postconditions"] = postconditions


def _drop_account_specific_output(data: dict[str, Any], tokens: Sequence[str]) -> None:
    remaining: list[dict[str, Any]] = []
    for item in data.get("output_criteria") or []:
        if not isinstance(item, dict):
            continue
        if item.get("kind") == OutputCriterionKind.CONTAINS.value:
            text = (item.get("parameters") or {}).get("text")
            if isinstance(text, str) and any(token and token in text for token in tokens):
                continue
        remaining.append(item)
    data["output_criteria"] = remaining


def _ensure_clarification_output(data: dict[str, Any]) -> None:
    for item in data.get("output_criteria") or []:
        if not isinstance(item, dict):
            continue
        if item.get("kind") != OutputCriterionKind.CONTAINS.value:
            continue
        text = (item.get("parameters") or {}).get("text")
        if text == _CLARIFY_TEXT:
            return
    scenario_id = str(data["scenario_id"])
    data.setdefault("output_criteria", []).append(
        {
            "criterion_id": _unique_criterion_id(data, f"{scenario_id}:output:clarify"),
            "kind": OutputCriterionKind.CONTAINS.value,
            "description": "The agent asks which account is intended.",
            "parameters": {"text": _CLARIFY_TEXT},
            "required": True,
            "oracle_ids": _oracle_ids(data),
        }
    )


def _withhold_confirmation(data: dict[str, Any]) -> tuple[dict[str, Any], JsonObject, str] | None:
    turns = [dict(turn) for turn in data.get("conversation_turns") or []]
    confirmation_indexes = [
        index
        for index, turn in enumerate(turns)
        if turn.get("role") == ConversationRole.USER.value
        and (turn.get("metadata") or {}).get("explicit_confirmation") is True
    ]
    if not confirmation_indexes:
        return None
    confirmed_required = [
        dict(item)
        for item in data.get("required_tool_behavior") or []
        if isinstance(item, dict) and item.get("confirmation_required_before_call") is True
    ]
    if not confirmed_required:
        return None

    user_indexes = _user_indexes(turns)
    other_users = [index for index in user_indexes if index not in set(confirmation_indexes)]
    drop: set[int] = set()
    if other_users:
        for index in confirmation_indexes:
            drop.add(index)
            if index > 0 and turns[index - 1].get("role") == ConversationRole.ASSISTANT.value:
                drop.add(index - 1)

    kept: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if index in drop:
            continue
        metadata = dict(turn.get("metadata") or {})
        metadata.pop("explicit_confirmation", None)
        updated = dict(turn)
        updated["metadata"] = metadata
        kept.append(updated)
    if not kept:
        return None
    data["conversation_turns"] = _renumber_turns(kept)

    remaining_required: list[dict[str, Any]] = []
    forbidden = [dict(item) for item in data.get("forbidden_tool_behavior") or []]
    flipped: list[str] = []
    for item in data.get("required_tool_behavior") or []:
        if not isinstance(item, dict):
            continue
        if item.get("confirmation_required_before_call") is True:
            tool_name = str(item.get("tool_name") or "")
            flipped.append(tool_name)
            forbidden.append(
                {
                    "criterion_id": str(item.get("criterion_id") or f"{data['scenario_id']}:tool:{tool_name}"),
                    "tool_name": tool_name,
                    "arguments_match": {},
                    "min_calls": 0,
                    "max_calls": 0,
                    "confirmation_required_before_call": False,
                    "oracle_ids": list(item.get("oracle_ids") or _oracle_ids(data)),
                }
            )
        else:
            remaining_required.append(item)
    data["required_tool_behavior"] = remaining_required
    data["forbidden_tool_behavior"] = forbidden
    account_ids = [
        arguments["account_id"]
        for item in confirmed_required
        for arguments in (item.get("arguments_match") or {},)
        if isinstance(arguments.get("account_id"), str)
    ]
    _flip_equals_postconditions(data, account_ids)
    if not _has_evaluable_criteria(data):
        return None
    parameters: JsonObject = {
        "removed_confirmation_turns": len(drop),
        "flipped_tools": flipped,
    }
    rationale = (
        "Withheld structural confirmation and forbade the previously required "
        "confirmed tool call."
    )
    return data, parameters, rationale


def _duplicate_request(data: dict[str, Any]) -> tuple[dict[str, Any], JsonObject, str] | None:
    turns = [dict(turn) for turn in data.get("conversation_turns") or []]
    user_indexes = _user_indexes(turns)
    if not user_indexes:
        return None
    required = [item for item in data.get("required_tool_behavior") or [] if isinstance(item, dict)]
    if not required:
        return None
    source_index = user_indexes[0]
    source = turns[source_index]
    if len(user_indexes) >= 2:
        last_user = turns[user_indexes[-1]]
        if last_user.get("content") == source.get("content"):
            return None
    duplicate = dict(source)
    duplicate["metadata"] = dict(source.get("metadata") or {})
    turns.append(duplicate)
    if len(turns) > MAX_CONVERSATION_TURNS:
        return None
    data["conversation_turns"] = _renumber_turns(turns)

    has_duplicate_constraint = any(
        isinstance(item, dict)
        and item.get("kind") == TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT.value
        for item in data.get("trajectory_constraints") or []
    )
    if not has_duplicate_constraint:
        tool_name = str(required[0].get("tool_name") or "")
        data.setdefault("trajectory_constraints", []).append(
            {
                "criterion_id": _unique_criterion_id(
                    data, f"{data['scenario_id']}:duplicate"
                ),
                "kind": TrajectoryConstraintKind.NO_DUPLICATE_SIDE_EFFECT.value,
                "description": "The same state-changing action must not execute twice.",
                "parameters": {"tool_name": tool_name},
                "required": True,
                "oracle_ids": _oracle_ids(data),
            }
        )

    fixtures = [dict(item) for item in data.get("tool_fixtures") or []]
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for fixture in fixtures:
        by_tool.setdefault(str(fixture.get("tool_name") or ""), []).append(fixture)
    extras: list[dict[str, Any]] = []
    for constraint in required:
        tool_name = str(constraint.get("tool_name") or "")
        tool_fixtures = by_tool.get(tool_name) or []
        if not tool_fixtures:
            continue
        if any(fixture.get("invocation_index") is None for fixture in tool_fixtures):
            continue
        highest = max(int(fixture["invocation_index"]) for fixture in tool_fixtures)
        template = dict(
            max(tool_fixtures, key=lambda item: int(item["invocation_index"]))
        )
        next_index = highest + 1
        template["invocation_index"] = next_index
        template["fixture_id"] = f"{data['scenario_id']}:{tool_name}:{next_index}"
        extras.append(template)
    data["tool_fixtures"] = fixtures + extras
    if not _has_evaluable_criteria(data):
        return None
    parameters: JsonObject = {
        "duplicated_turn_id": source.get("turn_id"),
        "added_fixtures": len(extras),
    }
    rationale = (
        "Duplicated the original user request and kept a single-call / no-duplicate "
        "side-effect constraint."
    )
    return data, parameters, rationale


def _ambiguous_identifier(data: dict[str, Any]) -> tuple[dict[str, Any], JsonObject, str] | None:
    accounts = _accounts(data)
    if accounts is None:
        return None
    turns = [dict(turn) for turn in data.get("conversation_turns") or []]
    text = _conversation_text(turns)
    mentioned_ids = [account_id for account_id in accounts if _token_in_text(text, account_id)]
    if len(mentioned_ids) != 1:
        return None
    target_id = mentioned_ids[0]
    record = accounts[target_id]
    if not isinstance(record, dict):
        return None
    name = record.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    same_name = [
        account_id
        for account_id, item in accounts.items()
        if isinstance(item, dict) and item.get("name") == name
    ]
    if len(same_name) != 1:
        return None
    if _OTHER_ACCOUNT_ID in accounts:
        return None

    replaced_turns: list[dict[str, Any]] = []
    for turn in turns:
        updated = dict(turn)
        content = updated.get("content")
        if isinstance(content, str):
            updated["content"] = _replace_token(content, target_id, name)
        replaced_turns.append(updated)
    data["conversation_turns"] = _renumber_turns(replaced_turns)

    cloned = dict(record)
    cloned["name"] = name
    new_accounts = dict(accounts)
    new_accounts[_OTHER_ACCOUNT_ID] = cloned
    state = dict(data.get("initial_world_state") or {})
    state["accounts"] = new_accounts
    data["initial_world_state"] = state

    remaining_required: list[dict[str, Any]] = []
    forbidden = [dict(item) for item in data.get("forbidden_tool_behavior") or []]
    flipped: list[str] = []
    for item in data.get("required_tool_behavior") or []:
        if not isinstance(item, dict):
            continue
        arguments = item.get("arguments_match") or {}
        if arguments.get("account_id") == target_id:
            tool_name = str(item.get("tool_name") or "")
            flipped.append(tool_name)
            forbidden.append(
                {
                    "criterion_id": str(item.get("criterion_id") or f"{data['scenario_id']}:tool:{tool_name}"),
                    "tool_name": tool_name,
                    "arguments_match": {},
                    "min_calls": 0,
                    "max_calls": 0,
                    "confirmation_required_before_call": False,
                    "oracle_ids": list(item.get("oracle_ids") or _oracle_ids(data)),
                }
            )
        else:
            remaining_required.append(item)
    data["required_tool_behavior"] = remaining_required
    data["forbidden_tool_behavior"] = forbidden
    tokens = [target_id]
    email = record.get("email")
    if isinstance(email, str) and email:
        tokens.append(email)
    _drop_account_specific_output(data, tokens)
    _ensure_clarification_output(data)
    _flip_equals_postconditions(data, (target_id,))
    if not flipped or not _has_evaluable_criteria(data):
        return None
    parameters: JsonObject = {
        "replaced_identifier": target_id,
        "display_name": name,
        "added_account_id": _OTHER_ACCOUNT_ID,
        "flipped_tools": flipped,
    }
    rationale = (
        "Replaced a unique world-state identifier with its existing display name "
        "and added a second controlled record so the name is no longer unique."
    )
    return data, parameters, rationale


def _reorder_dialogue(data: dict[str, Any]) -> tuple[dict[str, Any], JsonObject, str] | None:
    turns = [dict(turn) for turn in data.get("conversation_turns") or []]
    user_indexes = _user_indexes(turns)
    if len(user_indexes) < 2:
        return None
    first, last = user_indexes[0], user_indexes[-1]
    if turns[first].get("content") == turns[last].get("content") and (
        turns[first].get("metadata") or {}
    ) == (turns[last].get("metadata") or {}):
        return None
    left = dict(turns[first])
    right = dict(turns[last])
    left_content, left_metadata = left.get("content"), dict(left.get("metadata") or {})
    left["content"] = right.get("content")
    left["metadata"] = dict(right.get("metadata") or {})
    right["content"] = left_content
    right["metadata"] = left_metadata
    turns[first] = left
    turns[last] = right
    data["conversation_turns"] = _renumber_turns(turns)
    if not _has_evaluable_criteria(data):
        return None
    parameters: JsonObject = {
        "swapped_turn_ids": [turns[first]["turn_id"], turns[last]["turn_id"]],
    }
    rationale = "Swapped the first and last user turns without rewriting their text."
    return data, parameters, rationale


def _interleave_unrelated_turn(
    data: dict[str, Any],
) -> tuple[dict[str, Any], JsonObject, str] | None:
    turns = [dict(turn) for turn in data.get("conversation_turns") or []]
    user_indexes = _user_indexes(turns)
    if not user_indexes:
        return None
    if any(turn.get("content") == _UNRELATED_TURN for turn in turns):
        return None
    if len(turns) >= MAX_CONVERSATION_TURNS:
        return None
    insert_at = user_indexes[0] + 1
    turns.insert(
        insert_at,
        {
            "turn_id": "turn-unrelated",
            "role": ConversationRole.USER.value,
            "content": _UNRELATED_TURN,
            "metadata": {},
        },
    )
    data["conversation_turns"] = _renumber_turns(turns)
    if not _has_evaluable_criteria(data):
        return None
    parameters: JsonObject = {"inserted_after_index": user_indexes[0]}
    rationale = (
        "Inserted a fixed synthetic control turn after the original user request."
    )
    return data, parameters, rationale


_APPLIERS = {
    MutationKind.WITHHOLD_CONFIRMATION: _withhold_confirmation,
    MutationKind.DUPLICATE_REQUEST: _duplicate_request,
    MutationKind.AMBIGUOUS_IDENTIFIER: _ambiguous_identifier,
    MutationKind.REORDER_DIALOGUE: _reorder_dialogue,
    MutationKind.INTERLEAVE_UNRELATED_TURN: _interleave_unrelated_turn,
}

_UNSUPPORTED_EXPLANATIONS = {
    MutationKind.WITHHOLD_CONFIRMATION: (
        "withhold_confirmation: no user turn carries explicit_confirmation "
        "metadata together with a confirmation-required tool"
    ),
    MutationKind.DUPLICATE_REQUEST: (
        "duplicate_request: scenario has no user request with required tool behavior "
        "to duplicate without inventing a call contract"
    ),
    MutationKind.AMBIGUOUS_IDENTIFIER: (
        "ambiguous_identifier: conversation does not contain exactly one unique "
        "world-state account identifier that can be replaced with an existing name"
    ),
    MutationKind.REORDER_DIALOGUE: (
        "reorder_dialogue: fewer than two distinct user turns to swap"
    ),
    MutationKind.INTERLEAVE_UNRELATED_TURN: (
        "interleave_unrelated_turn: no user turn to follow, or the control turn "
        "is already present"
    ),
}


def mutate_scenario(
    parent: Scenario, kind: MutationKind, *, seed: int
) -> GeneratedMutation | None:
    """Return one lint-unchecked mutant, or ``None`` when the kind is unsupported."""

    data = _dump(parent)
    applied = _APPLIERS[kind](data)
    if applied is None:
        return None
    mutated, parameters, rationale = applied
    new_id = _mutation_scenario_id(kind, parent.scenario_id)
    rewritten = _rewrite_ids(mutated, parent.scenario_id, new_id)
    if not isinstance(rewritten, dict):
        return None
    rewritten["scenario_id"] = new_id
    rewritten["title"] = f"{kind.value.replace('_', ' ').title()} of {parent.title}"
    rewritten["description"] = rationale
    rewritten["generation_seed"] = seed
    tags = list(rewritten.get("dimension_tags") or [])
    mutation_tag = f"mutation:{kind.value}"
    if mutation_tag not in tags:
        tags.append(mutation_tag)
    rewritten["dimension_tags"] = tags
    _inherit_oracles(rewritten, parent_id=parent.scenario_id, kind=kind)
    rewritten["fingerprint"] = ""
    scenario = _load(rewritten)
    if scenario.fingerprint == parent.fingerprint:
        return None
    mutation = ScenarioMutation(
        kind=kind,
        parent_scenario_id=parent.scenario_id,
        parent_fingerprint=parent.fingerprint,
        resulting_scenario_id=scenario.scenario_id,
        rationale=rationale,
        parameters=parameters,
    )
    return GeneratedMutation(mutation=mutation, scenario=scenario)


def unsupported_mutation_reasons(scenario: Scenario) -> tuple[str, ...]:
    """Name every mutation this slice refuses rather than approximating."""

    reasons = list(DEFERRED_MUTATION_REASONS)
    for kind in MutationKind:
        if mutate_scenario(scenario, kind, seed=scenario.generation_seed) is None:
            reasons.append(_UNSUPPORTED_EXPLANATIONS[kind])
    return tuple(reasons)


def build_workflow_mutations(
    parents: Sequence[Scenario],
    *,
    seed: int,
    max_mutations: int = DEFAULT_MAX_MUTATIONS,
) -> tuple[GeneratedMutation, ...]:
    """Emit a bounded, deterministic mutant set from lint-clean parents."""

    if max_mutations < 1 or max_mutations > MAX_MUTATIONS_PER_SUITE:
        raise ValueError(
            f"max_mutations must be between 1 and {MAX_MUTATIONS_PER_SUITE}"
        )
    emitted: list[GeneratedMutation] = []
    seen: set[str] = {parent.fingerprint for parent in parents}
    for parent in parents:
        parent_count = 0
        for kind in MutationKind:
            if len(emitted) >= max_mutations or parent_count >= MAX_MUTATIONS_PER_PARENT:
                break
            generated = mutate_scenario(parent, kind, seed=seed)
            if generated is None:
                continue
            if generated.scenario.fingerprint in seen:
                continue
            seen.add(generated.scenario.fingerprint)
            emitted.append(generated)
            parent_count += 1
        if len(emitted) >= max_mutations:
            break
    return tuple(emitted)


def inherited_hard_failure_oracles(parent: Scenario, mutant: Scenario) -> bool:
    """True when every parent hard-failure oracle is still hard on the mutant."""

    parent_hard = {
        oracle.oracle_id.split(":", 1)[-1]: oracle
        for oracle in parent.oracle_provenance
        if oracle.supports_hard_failure
    }
    mutant_by_suffix = {
        oracle.oracle_id.split(":", 1)[-1]: oracle for oracle in mutant.oracle_provenance
    }
    for suffix, oracle in parent_hard.items():
        inherited = mutant_by_suffix.get(suffix)
        if inherited is None:
            return False
        if inherited.supports_hard_failure is not True:
            return False
        if inherited.strength is not oracle.strength:
            return False
        if inherited.confidence < oracle.confidence:
            return False
    return True


__all__ = [
    "DEFAULT_MAX_MUTATIONS",
    "DEFERRED_MUTATION_REASONS",
    "GENERATOR_NAME",
    "MAX_MUTATIONS_PER_PARENT",
    "MAX_MUTATIONS_PER_SUITE",
    "MUTATION_CONTRACT_VERSION",
    "GeneratedMutation",
    "MutationKind",
    "ScenarioMutation",
    "build_workflow_mutations",
    "inherited_hard_failure_oracles",
    "mutate_scenario",
    "unsupported_mutation_reasons",
]
