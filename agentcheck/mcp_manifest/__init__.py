"""Developer-declared schemas for a target's external (MCP) toolsets."""

from .loader import load_mcp_manifest
from .pack import (
    DEFAULT_MCP_MANIFEST_FILENAME,
    MCP_MANIFEST_CONTRACT_VERSION,
    DeclaredMcpTool,
    McpManifest,
)

__all__ = [
    "DEFAULT_MCP_MANIFEST_FILENAME",
    "MCP_MANIFEST_CONTRACT_VERSION",
    "DeclaredMcpTool",
    "McpManifest",
    "load_mcp_manifest",
]
