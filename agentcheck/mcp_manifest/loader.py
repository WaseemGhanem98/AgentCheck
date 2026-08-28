"""Load and validate a developer-declared external-toolset manifest.

Mirrors ``agentcheck/fixtures/loader.py`` deliberately: contained path, no
symlink following, bounded size, versioned contract, offline-only schema
validation, and a ``ConfigurationError`` on anything malformed. A manifest
that cannot be trusted is refused rather than partially applied -- the same
reasoning as fixtures: a silently ignored or partially-read declaration would
look like coverage the target does not actually have.
"""

from __future__ import annotations

import os
from pathlib import Path

from agentcheck.config import contained_path
from agentcheck.errors import ConfigurationError
from agentcheck.schema_safety import UnsafeSchemaReference, offline_validator

from .pack import DEFAULT_MCP_MANIFEST_FILENAME, McpManifest


_MAX_MANIFEST_BYTES = 256 * 1024


def load_mcp_manifest(root: Path, *, filename: str | None = None) -> McpManifest | None:
    """Read the target's MCP manifest file, or ``None`` when the file is absent.

    Absence is the normal case, not an error: a target with no external
    toolset needs no manifest, and one that has one but declares no manifest
    keeps failing closed exactly as before this existed.
    """

    name = filename or DEFAULT_MCP_MANIFEST_FILENAME
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
            raw = handle.read(_MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise ConfigurationError(f"unable to read {name}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ConfigurationError(
            f"{name} exceeds the {_MAX_MANIFEST_BYTES} byte manifest limit"
        )
    try:
        manifest = McpManifest.model_validate_json(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ConfigurationError(f"invalid {name}: {exc}") from exc
    # Refused rather than treated as absent: a file that declares nothing
    # would otherwise clear the external-toolset refusal while describing no
    # tools at all, which reads as coverage the target never had.
    if not manifest.tools:
        raise ConfigurationError(
            f"{name} declares no tools. Remove the file if the target has no "
            "external toolset, or declare the toolset's tool schemas in it."
        )
    for tool_name, declared in manifest.tools.items():
        try:
            offline_validator(declared.input_schema)
        except UnsafeSchemaReference as exc:
            raise ConfigurationError(
                f"{name}: tool {tool_name!r} input_schema {exc}"
            ) from exc
        except Exception as exc:  # jsonschema's own schema-validity errors
            raise ConfigurationError(
                f"{name}: tool {tool_name!r} input_schema is not a valid JSON Schema: {exc}"
            ) from exc
    return manifest


__all__ = ["load_mcp_manifest"]
