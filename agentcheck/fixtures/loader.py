"""Load and validate representative input values from inside the target.

Mirrors the policy-pack loader deliberately: contained path, no symlink
following, bounded size, versioned contract, and a ``ConfigurationError`` on
anything malformed. A fixture that cannot be trusted is refused rather than
partially applied, because a silently ignored fixture would look like coverage
the suite does not have.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from agentcheck.config import contained_path
from agentcheck.domain import AgentSpec, JsonObject
from agentcheck.errors import ConfigurationError
from agentcheck.privacy import redact_log_text
from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator

from .pack import DEFAULT_FIXTURES_FILENAME, FixturePack


_MAX_FIXTURE_BYTES = 64 * 1024
_ENV_KEY = "$env"


def load_fixture_pack(root: Path, *, filename: str | None = None) -> FixturePack | None:
    """Read the target's fixture file, or ``None`` when it declares none.

    Absence is the normal case and is not an error: fixtures are optional and
    a target without them still gets structural coverage.
    """

    name = filename or DEFAULT_FIXTURES_FILENAME
    path = contained_path(root, name)
    if not path.exists():
        return None
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConfigurationError(f"unable to read {name}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read(_MAX_FIXTURE_BYTES + 1)
    except OSError as exc:
        raise ConfigurationError(f"unable to read {name}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > _MAX_FIXTURE_BYTES:
        raise ConfigurationError(
            f"{name} exceeds the {_MAX_FIXTURE_BYTES} byte fixture limit"
        )
    try:
        return FixturePack.model_validate_json(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ConfigurationError(f"invalid {name}: {exc}") from exc


def _reject_environment_reference(tool: str, parameter: str, value: Any) -> None:
    """Refuse ``{"$env": ...}`` rather than half-supporting it.

    A model-supplied argument is not where a secret belongs: it would have to
    reach the request text, and therefore the frozen suite, to mean anything.
    Rather than serialise a credential or pretend a placeholder is a usable
    value, this is unsupported and says so.
    """

    if isinstance(value, Mapping) and _ENV_KEY in value:
        raise ConfigurationError(
            f"{tool}.{parameter} uses {_ENV_KEY}, which is not supported for "
            "representative input values: an argument the model supplies would "
            "have to be written into the generated request and the frozen "
            "suite. Use a committed non-secret test value instead, and keep "
            "credentials in environment_allowlist for the provider."
        )


def validate_fixture_pack(pack: FixturePack, spec: AgentSpec) -> dict[str, JsonObject]:
    """Check every value against the tool contract as it exists right now.

    Fails closed on an unknown tool, an unknown parameter, or a value the
    declared schema rejects, so a fixture that drifted out of date is reported
    instead of quietly shaping a scenario that no longer matches the target.
    """

    definitions = {item.value.name: item.value for item in spec.tools.items}
    resolved: dict[str, JsonObject] = {}
    for tool_name in sorted(pack.tools):
        values = pack.tools[tool_name]
        definition = definitions.get(tool_name)
        if definition is None:
            known = ", ".join(sorted(definitions)) or "(none)"
            raise ConfigurationError(
                f"fixture names unknown tool {tool_name!r}; this target declares: {known}"
            )
        schema = definition.input_schema or {}
        declared = set((schema.get("properties") or {}))
        for parameter in sorted(values.arguments):
            value = values.arguments[parameter]
            _reject_environment_reference(tool_name, parameter, value)
            if declared and parameter not in declared:
                known = ", ".join(sorted(declared)) or "(none)"
                raise ConfigurationError(
                    f"fixture for {tool_name} names unknown parameter "
                    f"{parameter!r}; the tool declares: {known}. The tool schema "
                    "may have changed since the fixture was written."
                )
            if isinstance(value, str) and redact_log_text(value) != value:
                raise ConfigurationError(
                    f"{tool_name}.{parameter} looks like a credential; fixtures are "
                    "committed test data and must not contain secrets"
                )
        if values.arguments:
            try:
                validator = offline_validator(dict(schema))
            except (UnsafeSchemaReference, ValueError) as exc:
                raise ConfigurationError(
                    f"cannot validate fixtures for {tool_name}: {exc}"
                ) from exc
            errors = sorted(
                validator.iter_errors(dict(values.arguments)),
                key=lambda item: list(item.path),
            )
            # Partial fixtures are allowed, so a missing required argument is
            # filled from the schema baseline rather than rejected here.
            fatal = [e for e in errors if e.validator != "required"]
            if fatal:
                first = fatal[0]
                where = ".".join(str(part) for part in first.path) or "(root)"
                raise ConfigurationError(
                    f"fixture value for {tool_name}.{where} does not satisfy the "
                    f"declared schema: {first.message}"
                )
        if values.user_request is not None:
            # Already stripped and length-checked by the contract model, so a
            # blank request fails at parse time and never reaches here.
            request = values.user_request
            if redact_log_text(request) != request:
                raise ConfigurationError(
                    f"{tool_name}.user_request looks like it contains a credential; "
                    "fixtures are committed test data and must not contain secrets"
                )
        resolved[tool_name] = dict(values.arguments)
    return resolved


def load_representative_inputs(
    root: Path, spec: AgentSpec, *, filename: str | None = None
) -> dict[str, JsonObject]:
    """Loaded, validated representative inputs, or an empty mapping."""

    pack = load_fixture_pack(root, filename=filename)
    if pack is None:
        return {}
    return validate_fixture_pack(pack, spec)


def load_scenario_requests(
    root: Path, spec: AgentSpec, *, filename: str | None = None
) -> dict[str, str]:
    """Developer-authored request text per tool, or an empty mapping.

    Kept separate from :func:`load_representative_inputs` because the two
    answer different questions: that one supplies the values an action needs,
    this one supplies the situation that makes acting the right response.
    Validation is shared, so an authored request is checked against the same
    tool contract before it can shape a scenario.
    """

    pack = load_fixture_pack(root, filename=filename)
    if pack is None:
        return {}
    validate_fixture_pack(pack, spec)
    return {
        name: values.user_request
        for name, values in sorted(pack.tools.items())
        if values.user_request is not None
    }


__all__ = [
    "load_fixture_pack",
    "load_representative_inputs",
    "load_scenario_requests",
    "validate_fixture_pack",
]
