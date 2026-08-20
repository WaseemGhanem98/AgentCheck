"""Framework adapters supported by AgentCheck."""

from .base import (
    AdapterDependencyError,
    AdapterError,
    AdapterRuntimeError,
    EventSinkProtocol,
    FrameworkAdapter,
    GatewayRequest,
    PreflightReport,
    PreparedTarget,
    SupportIssue,
    ToolGatewayProtocol,
    UnsupportedTargetError,
    decode_preflight_report,
    decode_topology,
    encode_preflight_report,
    format_support_issues,
)
from .openai_agents import OpenAIAgentsAdapter
from .pydantic_ai import PydanticAIAdapter

__all__ = [
    "AdapterDependencyError",
    "AdapterError",
    "AdapterRuntimeError",
    "EventSinkProtocol",
    "FrameworkAdapter",
    "GatewayRequest",
    "OpenAIAgentsAdapter",
    "PreflightReport",
    "PydanticAIAdapter",
    "PreparedTarget",
    "SupportIssue",
    "ToolGatewayProtocol",
    "UnsupportedTargetError",
    "decode_preflight_report",
    "decode_topology",
    "encode_preflight_report",
    "format_support_issues",
]
