"""Prove a narrow OpenAI Agents ``on_handoff`` subset without executing it.

The official customer-service callback assigns ``context.context.flight_number``
from an f-string that embeds ``random.randint``.  AgentCheck reconstructs that
as a declarative context assignment and applies it to an AgentCheck-owned bag.
The original callable is never invoked, compiled, or stored in artifacts.

Anything that cannot be shown to be that form — including nested state, control
flow, awaits, helpers, wrappers, and unavailable source — stays
``handoff_callback``.  ``random.randint(lo, hi)`` is a recognized constructor
replaced by the deterministic lower bound; the original RNG is not executed.
"""

from __future__ import annotations

import ast
import functools
import inspect
import random as stdlib_random
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentcheck.privacy import redact_log_text

from .base import AdapterRuntimeError, SupportIssue

_MAX_SOURCE_CHARS = 8192
_MAX_AST_NODES = 200
_MAX_BODY_STATEMENTS = 16
_MAX_ASSIGNMENTS = 4
_MAX_STRING_CHARS = 200
_MAX_RANDINT_SPAN = 1_000_000
_FIELD_NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")

_SHAPE_MESSAGE = (
    "The on_handoff callback is not a supported context-assignment form; "
    "AgentCheck never executes target callbacks."
)
_SIDE_EFFECT_MESSAGE = (
    "The on_handoff callback uses operations AgentCheck will not reconstruct "
    "without executing target code."
)
_SOURCE_MESSAGE = (
    "The on_handoff callback source is unavailable; AgentCheck cannot prove "
    "it is a supported context assignment."
)


class _CallbackReject(Exception):
    """Internal fail-closed signal; never raised to callers."""

    def __init__(self, kind: str, detail: str) -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class ContextAssignment:
    """One proven setattr onto the AgentCheck-owned run context bag."""

    field: str
    value: bool | int | str | None


@dataclass(frozen=True)
class HandoffCallbackAnalysis:
    """Result of static callback analysis.  ``issue`` set means unsupported."""

    assignments: tuple[ContextAssignment, ...]
    issue: SupportIssue | None


class AgentCheckRunContext:
    """Mutable bag for proven ``on_handoff`` assignments.

    Original target Pydantic context models are never instantiated.  setattr
    here cannot reach objects the target created.
    """


def encode_context_assignments(
    assignments: Sequence[ContextAssignment],
) -> list[dict[str, bool | int | str | None]]:
    """JSON-safe evidence records; field/value only, never callback source."""

    return [{"field": item.field, "value": item.value} for item in assignments]


def apply_context_assignments(
    bag: Any, assignments: Sequence[ContextAssignment]
) -> None:
    """Apply proven assignments onto the AgentCheck-owned context bag."""

    if not isinstance(bag, AgentCheckRunContext):
        raise AdapterRuntimeError("handoff context bag is not AgentCheck-owned")
    for item in assignments:
        setattr(bag, item.field, item.value)


def analyze_on_handoff_callback(
    callback: Any, *, location: str
) -> HandoffCallbackAnalysis:
    """Return proven assignments or a ``handoff_callback`` issue.

    This function must not call ``callback``.
    """

    try:
        assignments = _analyze_callback(callback)
    except _CallbackReject as exc:
        message = {
            "source": _SOURCE_MESSAGE,
            "shape": _SHAPE_MESSAGE,
            "side_effect": _SIDE_EFFECT_MESSAGE,
        }.get(exc.kind, _SHAPE_MESSAGE)
        return HandoffCallbackAnalysis(
            assignments=(),
            issue=SupportIssue(
                code="handoff_callback",
                message=message,
                location=location,
            ),
        )
    return HandoffCallbackAnalysis(assignments=assignments, issue=None)


def _analyze_callback(callback: Any) -> tuple[ContextAssignment, ...]:
    if isinstance(callback, functools.partial) or inspect.ismethod(callback):
        raise _CallbackReject("shape", "callable is wrapped or bound")
    if getattr(callback, "__wrapped__", None) is not None:
        raise _CallbackReject("shape", "callable is decorator-wrapped")
    if not inspect.isfunction(callback) or callback.__name__ == "<lambda>":
        raise _CallbackReject("shape", "callable is not a plain function")
    if callback.__code__.co_freevars:
        raise _CallbackReject("shape", "callback closes over free variables")
    try:
        closure = inspect.getclosurevars(callback)
    except (TypeError, ValueError) as exc:
        raise _CallbackReject("shape", "closure cannot be inspected") from exc
    if closure.nonlocals:
        raise _CallbackReject("shape", "callback closes over nonlocals")

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError) as exc:
        raise _CallbackReject("shape", "signature cannot be inspected") from exc
    parameters = list(signature.parameters.values())
    if len(parameters) != 1:
        raise _CallbackReject("shape", "callback does not take one argument")
    kind = parameters[0].kind
    if kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise _CallbackReject("shape", "callback parameter is not positional")

    source = _callback_source(callback)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise _CallbackReject("shape", "callback source is not valid Python") from exc
    if len(tree.body) != 1 or not isinstance(
        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        raise _CallbackReject("shape", "source is not a single function")
    function = tree.body[0]
    if function.name != callback.__name__:
        raise _CallbackReject("shape", "source function name mismatch")
    if function.decorator_list:
        raise _CallbackReject("shape", "callback is decorated")
    if function.args.defaults or function.args.kw_defaults:
        raise _CallbackReject("shape", "callback has default arguments")
    _assert_single_positional_arg(function)
    _assert_node_budget(function)
    return _assignments_from_body(
        function,
        param_name=_positional_arg_name(function),
        globals_map=callback.__globals__,
    )


def _callback_source(callback: Any) -> str:
    try:
        source = inspect.getsource(callback)
    except (OSError, TypeError) as exc:
        raise _CallbackReject("source", "getsource failed") from exc
    sourcefile = inspect.getsourcefile(callback)
    if sourcefile is None or sourcefile.startswith("<"):
        raise _CallbackReject("source", "source file is unavailable")
    source = textwrap.dedent(source)
    if not source or len(source) > _MAX_SOURCE_CHARS:
        raise _CallbackReject("source", "source is empty or oversized")
    return source


def _positional_arg_name(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    positional = [*function.args.posonlyargs, *function.args.args]
    return positional[0].arg


def _assert_single_positional_arg(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    args = function.args
    positional = [*args.posonlyargs, *args.args]
    if len(positional) != 1 or args.vararg is not None or args.kwarg is not None:
        raise _CallbackReject("shape", "function does not take one positional argument")
    if args.kwonlyargs:
        raise _CallbackReject("shape", "function has keyword-only parameters")


def _assert_node_budget(function: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    """Walk the body only.  Annotations may legally contain Subscript nodes."""

    nodes = 0
    for statement in function.body:
        for node in ast.walk(statement):
            nodes += 1
            if nodes > _MAX_AST_NODES:
                raise _CallbackReject("shape", "callback AST exceeds the analysis bound")
            if isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom,
                    ast.Global,
                    ast.Nonlocal,
                    ast.With,
                    ast.AsyncWith,
                    ast.Try,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.If,
                    ast.Match,
                    ast.Raise,
                    ast.Assert,
                    ast.Delete,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.Await,
                    ast.Lambda,
                    ast.ClassDef,
                    ast.NamedExpr,
                    ast.Starred,
                    ast.AugAssign,
                    ast.AnnAssign,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                raise _CallbackReject("side_effect", type(node).__name__)


def _assignments_from_body(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    param_name: str,
    globals_map: Mapping[str, Any],
) -> tuple[ContextAssignment, ...]:
    statements = list(function.body)
    if statements and _is_docstring(statements[0]):
        statements = statements[1:]
    if statements and isinstance(statements[-1], ast.Return):
        value = statements[-1].value
        if value is not None and not (
            isinstance(value, ast.Constant) and value.value is None
        ):
            raise _CallbackReject("shape", "callback returns a value")
        statements = statements[:-1]
    if len(statements) > _MAX_BODY_STATEMENTS:
        raise _CallbackReject("shape", "callback body is too large")

    locals_map: dict[str, bool | int | str | None] = {}
    assignments: list[ContextAssignment] = []
    seen_fields: set[str] = set()
    budget = [_MAX_AST_NODES]
    for statement in statements:
        if isinstance(statement, ast.Pass):
            continue
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise _CallbackReject("shape", "unsupported statement")
        target = statement.targets[0]
        assigned = _eval_expr(
            statement.value,
            locals_map=locals_map,
            globals_map=globals_map,
            budget=budget,
        )
        if isinstance(target, ast.Name):
            if target.id == param_name or not _FIELD_NAME.match(target.id):
                raise _CallbackReject("shape", "invalid local name")
            locals_map[target.id] = assigned
            continue
        field = _context_field_name(target, param_name)
        if field in seen_fields:
            raise _CallbackReject("shape", "duplicate context field")
        if len(assignments) >= _MAX_ASSIGNMENTS:
            raise _CallbackReject("shape", "too many context assignments")
        if isinstance(assigned, str) and redact_log_text(assigned) != assigned:
            raise _CallbackReject("side_effect", "assignment value is secret-shaped")
        seen_fields.add(field)
        assignments.append(ContextAssignment(field=field, value=assigned))
    return tuple(assignments)


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and type(statement.value.value) is str
    )


def _context_field_name(target: ast.expr, param_name: str) -> str:
    if not isinstance(target, ast.Attribute):
        raise _CallbackReject("shape", "assignment target is not an attribute")
    inner = target.value
    if not isinstance(inner, ast.Attribute):
        raise _CallbackReject("shape", "assignment is not context.context.<field>")
    if not isinstance(inner.value, ast.Name) or inner.value.id != param_name:
        raise _CallbackReject("shape", "assignment does not use the callback parameter")
    if inner.attr != "context":
        raise _CallbackReject("shape", "assignment does not target wrapper.context")
    if not _FIELD_NAME.match(target.attr):
        raise _CallbackReject("shape", "context field name is not a safe identifier")
    return target.attr


def _eval_expr(
    node: ast.expr,
    *,
    locals_map: Mapping[str, bool | int | str | None],
    globals_map: Mapping[str, Any],
    budget: list[int],
) -> bool | int | str | None:
    budget[0] -= 1
    if budget[0] < 0:
        raise _CallbackReject("shape", "expression exceeds the analysis bound")
    if isinstance(node, ast.Constant):
        return _safe_constant(node.value)
    if isinstance(node, ast.Name):
        if node.id not in locals_map:
            raise _CallbackReject("side_effect", "unknown name")
        return locals_map[node.id]
    if isinstance(node, ast.JoinedStr):
        return _eval_joined_str(
            node, locals_map=locals_map, globals_map=globals_map, budget=budget
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_expr(
            node.left, locals_map=locals_map, globals_map=globals_map, budget=budget
        )
        right = _eval_expr(
            node.right, locals_map=locals_map, globals_map=globals_map, budget=budget
        )
        if type(left) is not str or type(right) is not str:
            raise _CallbackReject("shape", "addition is not string concatenation")
        return _bounded_string(left + right)
    if isinstance(node, ast.Call):
        return _eval_randint(
            node, locals_map=locals_map, globals_map=globals_map, budget=budget
        )
    raise _CallbackReject("side_effect", type(node).__name__)


def _eval_joined_str(
    node: ast.JoinedStr,
    *,
    locals_map: Mapping[str, bool | int | str | None],
    globals_map: Mapping[str, Any],
    budget: list[int],
) -> str:
    parts: list[str] = []
    for value in node.values:
        budget[0] -= 1
        if budget[0] < 0:
            raise _CallbackReject("shape", "f-string exceeds the analysis bound")
        if isinstance(value, ast.Constant) and type(value.value) is str:
            parts.append(value.value)
            continue
        if not isinstance(value, ast.FormattedValue):
            raise _CallbackReject("shape", "unsupported f-string part")
        if value.conversion not in (-1, 115) or value.format_spec is not None:
            raise _CallbackReject("shape", "unsupported f-string conversion")
        rendered = _eval_expr(
            value.value, locals_map=locals_map, globals_map=globals_map, budget=budget
        )
        if type(rendered) not in (int, str) or type(rendered) is bool:
            raise _CallbackReject("shape", "f-string value is not int or str")
        parts.append(str(rendered))
    return _bounded_string("".join(parts))


def _eval_randint(
    node: ast.Call,
    *,
    locals_map: Mapping[str, bool | int | str | None],
    globals_map: Mapping[str, Any],
    budget: list[int],
) -> int:
    if node.keywords or len(node.args) != 2:
        raise _CallbackReject("side_effect", "unsupported call")
    if not _is_proven_randint(node.func, locals_map=locals_map, globals_map=globals_map):
        raise _CallbackReject("side_effect", "unsupported call")
    low = _eval_expr(
        node.args[0], locals_map=locals_map, globals_map=globals_map, budget=budget
    )
    high = _eval_expr(
        node.args[1], locals_map=locals_map, globals_map=globals_map, budget=budget
    )
    if type(low) is not int or type(high) is not int:
        raise _CallbackReject("shape", "randint bounds are not integers")
    if low > high or (high - low) > _MAX_RANDINT_SPAN:
        raise _CallbackReject("shape", "randint range is invalid")
    # Deterministic AgentCheck-owned stand-in for the original RNG.
    return low


def _is_proven_randint(
    func: ast.expr,
    *,
    locals_map: Mapping[str, bool | int | str | None],
    globals_map: Mapping[str, Any],
) -> bool:
    if isinstance(func, ast.Attribute):
        if not isinstance(func.value, ast.Name) or func.attr != "randint":
            return False
        if func.value.id in locals_map:
            return False
        return globals_map.get(func.value.id) is stdlib_random
    if isinstance(func, ast.Name):
        if func.id in locals_map:
            return False
        return globals_map.get(func.id) is stdlib_random.randint
    return False


def _safe_constant(value: Any) -> bool | int | str | None:
    if value is None or type(value) is bool or type(value) is int or type(value) is str:
        if type(value) is str:
            return _bounded_string(value)
        return value
    raise _CallbackReject("shape", "unsupported constant")


def _bounded_string(value: str) -> str:
    if len(value) > _MAX_STRING_CHARS:
        raise _CallbackReject("shape", "string exceeds the analysis bound")
    return value
