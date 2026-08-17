"""Load the explicit local AgentCheck target used by the Phase 1 CLI.

The MVP intentionally avoids repository-wide code discovery.  A directory target
uses ``agentcheck.json`` when present and otherwise defaults to
``agent.py:agent``.  Direct ``file.py`` and ``file.py:attribute`` references are
also accepted.

Nested entrypoints are imported as packages under the configured target root so
``from .tools import ...`` and ``from examples.auto_mode import ...`` resolve
with normal Python semantics.  Only that root is added to ``sys.path``.  Files
whose resolved path leaves the root, including outbound symlinks, are refused.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import importlib.util
import inspect
import json
import keyword
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from agentcheck.config import ParsedEntrypoint, parse_entrypoint


class TargetLoadError(RuntimeError):
    """Raised when a configured local target cannot be imported unambiguously."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "target_import_failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _split_entrypoint(value: str, *, base: Path) -> tuple[Path, str, bool]:
    try:
        parsed: ParsedEntrypoint = parse_entrypoint(value)
    except ValueError as exc:
        raise TargetLoadError(str(exc), code="target_entrypoint_invalid") from exc
    module_path = Path(parsed.path)
    if not module_path.is_absolute():
        module_path = base / module_path
    try:
        module_path = module_path.resolve()
    except (OSError, RuntimeError) as exc:
        raise TargetLoadError(
            f"target path cannot be resolved: {exc}", code="target_path_escape"
        ) from exc
    if module_path.suffix != ".py":
        raise TargetLoadError(
            "Phase 1 entrypoints must reference a .py file",
            code="target_entrypoint_invalid",
        )
    if not module_path.is_file():
        raise TargetLoadError(
            f"entrypoint file does not exist: {module_path}",
            code="target_entrypoint_invalid",
        )
    return module_path, parsed.attribute, parsed.factory


def _require_contained(path: Path, root: Path) -> Path:
    """Resolve ``path`` and refuse anything that leaves ``root``."""

    try:
        resolved_root = root.resolve()
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise TargetLoadError(
            f"target path cannot be resolved: {exc}", code="target_path_escape"
        ) from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise TargetLoadError(
            "refusing to follow a path that escapes the target directory",
            code="target_path_escape",
        ) from exc
    return resolved


def resolve_entrypoint(target: str | Path) -> tuple[Path, str]:
    """Resolve a directory/config/direct-file target without importing it."""

    module_path, attribute, _root, _factory = _resolve_target(target)
    return module_path, attribute


def _resolve_target(target: str | Path) -> tuple[Path, str, Path, bool]:
    """Return ``(module_path, attribute, root, factory)`` contained in root."""

    raw_target = str(target)
    if ":" in raw_target:
        possible_file, _ = raw_target.rsplit(":", 1)
        if possible_file.endswith(".py"):
            module_path, attribute, factory = _split_entrypoint(
                raw_target, base=Path.cwd()
            )
            root = module_path.parent
            return module_path, attribute, root, factory

    path = Path(raw_target).resolve()
    if path.is_file():
        if path.suffix != ".py":
            raise TargetLoadError(
                f"target file must be Python: {path}",
                code="target_entrypoint_invalid",
            )
        return path, "agent", path.parent, False
    if not path.is_dir():
        raise TargetLoadError(
            f"target path does not exist: {path}", code="target_entrypoint_invalid"
        )

    config_path = path / "agentcheck.json"
    entrypoint = "agent.py:agent"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TargetLoadError(
                f"invalid {config_path.name}: {exc}",
                code="target_entrypoint_invalid",
            ) from exc
        if not isinstance(config, dict):
            raise TargetLoadError(
                f"{config_path.name} must contain a JSON object",
                code="target_entrypoint_invalid",
            )
        configured = config.get("entrypoint", entrypoint)
        if not isinstance(configured, str) or not configured.strip():
            raise TargetLoadError(
                "agentcheck.json entrypoint must be a non-empty string",
                code="target_entrypoint_invalid",
            )
        entrypoint = configured.strip()
    module_path, attribute, factory = _split_entrypoint(entrypoint, base=path)
    _require_contained(module_path, path)
    return module_path, attribute, path, factory


def _path_exists(path: Path) -> bool:
    try:
        return path.is_symlink() or path.exists()
    except OSError:
        return False


def _dotted_module_name(relative: Path) -> str:
    if relative.suffix != ".py":
        raise TargetLoadError("Phase 1 entrypoints must reference a .py file")
    if relative.name == "__init__.py":
        parts = relative.parent.parts
        if not parts:
            raise TargetLoadError(
                "the target root __init__.py cannot be used as a package entrypoint"
            )
    else:
        parts = relative.with_suffix("").parts
    if not parts:
        raise TargetLoadError("entrypoint path is not a valid Python module")
    for part in parts:
        if not part.isidentifier() or keyword.iskeyword(part):
            raise TargetLoadError(
                f"entrypoint path component {part!r} is not a valid Python module name"
            )
    return ".".join(parts)


class _ContainedTargetFinder:
    """Load target-root modules only; outbound resolved paths fail closed."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        parts = fullname.split(".")
        if not parts or not all(part.isidentifier() for part in parts):
            return None
        current = self.root
        for index, part in enumerate(parts):
            if keyword.iskeyword(part):
                return None
            module_file = current / f"{part}.py"
            package_dir = current / part
            file_hit = _path_exists(module_file)
            dir_hit = _path_exists(package_dir)
            if file_hit and dir_hit:
                raise TargetLoadError(
                    f"ambiguous import {fullname!r}: both {module_file.name} "
                    f"and {package_dir.name}/ exist under the target",
                    code="target_import_failed",
                )
            last = index == len(parts) - 1
            if not file_hit and not dir_hit:
                if index == 0:
                    return None
                raise TargetLoadError(
                    f"no module named {fullname!r} inside the target directory",
                    code="target_import_failed",
                )
            if file_hit:
                resolved = _require_contained(module_file, self.root)
                if not resolved.is_file():
                    raise TargetLoadError(
                        "refusing to follow a path that escapes the target directory"
                    )
                if not last:
                    raise TargetLoadError(
                        f"cannot import {fullname!r}: {part!r} is a module, not a package"
                    )
                return importlib.util.spec_from_file_location(fullname, resolved)
            resolved_dir = _require_contained(package_dir, self.root)
            if not resolved_dir.is_dir():
                raise TargetLoadError(
                    "refusing to follow a path that escapes the target directory"
                )
            init_file = package_dir / "__init__.py"
            if _path_exists(init_file):
                resolved_init = _require_contained(init_file, self.root)
                if not resolved_init.is_file():
                    raise TargetLoadError(
                        "refusing to follow a path that escapes the target directory"
                    )
                if last:
                    return importlib.util.spec_from_file_location(
                        fullname,
                        resolved_init,
                        submodule_search_locations=[str(resolved_dir)],
                    )
                current = resolved_dir
                continue
            if last:
                spec = importlib.machinery.ModuleSpec(
                    fullname, None, is_package=True
                )
                spec.submodule_search_locations = [str(resolved_dir)]
                return spec
            current = resolved_dir
        return None


def enable_contained_target_imports(root: Path) -> None:
    """Keep the target root importable for the rest of this process.

    The worker installs this once after AgentCheck itself has been imported so
    target code cannot shadow the worker at startup, and so later scenario
    imports still resolve under the same containment rules.
    """

    resolved = root.resolve()
    root_str = str(resolved)
    if not any(
        isinstance(item, _ContainedTargetFinder) and item.root == resolved
        for item in sys.meta_path
    ):
        sys.meta_path.insert(0, _ContainedTargetFinder(resolved))
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


@contextmanager
def _target_import_context(root: Path) -> Iterator[None]:
    resolved = root.resolve()
    root_str = str(resolved)
    finder = _ContainedTargetFinder(resolved)
    sys.meta_path.insert(0, finder)
    inserted_path = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        inserted_path = True
    try:
        yield
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        if inserted_path:
            try:
                sys.path.remove(root_str)
            except ValueError:
                pass


def _evict_stale_modules(module_name: str, *, root: Path) -> None:
    """Drop cached packages that would shadow this target's import name."""

    resolved_root = root.resolve()
    parts = module_name.split(".")
    for length in range(1, len(parts) + 1):
        name = ".".join(parts[:length])
        existing = sys.modules.get(name)
        if existing is None:
            continue
        origin = getattr(existing, "__file__", None)
        under_root = False
        if isinstance(origin, str) and origin:
            try:
                Path(origin).resolve().relative_to(resolved_root)
            except (ValueError, OSError, RuntimeError):
                under_root = False
            else:
                under_root = True
        if under_root:
            continue
        stale = [
            cached
            for cached in list(sys.modules)
            if cached == name or cached.startswith(f"{name}.")
        ]
        for cached in stale:
            sys.modules.pop(cached, None)


def _provider_required_during_import(exc: BaseException) -> bool:
    if "OPENAI_API_KEY" in os.environ:
        return False
    try:
        text = str(exc)
    except BaseException:
        return False
    markers = ("OPENAI_API_KEY", "api_key client option must be set")
    return any(marker in text for marker in markers)


def _import_failure(module_path: Path, exc: BaseException) -> TargetLoadError:
    if isinstance(exc, ModuleNotFoundError):
        missing = exc.name or "unknown"
        return TargetLoadError(
            (
                f"Target import failed: no module named {missing!r} in the worker "
                "Python environment. AgentCheck does not install target dependencies "
                "automatically and does not download packages from the network. "
                "Prepare a Python environment that contains the target's packages, "
                "then run AgentCheck from that environment or set python_executable "
                f"/ --python to that interpreter. Worker Python: {sys.executable}"
            ),
            code="target_dependency_missing",
            details={
                "missing_module": missing,
                "python_executable": sys.executable,
            },
        )
    if _provider_required_during_import(exc):
        return TargetLoadError(
            (
                "Target import failed because discovery required a provider "
                "credential. AgentCheck does not pass OPENAI_API_KEY unless it is "
                "listed in environment_allowlist. This is not a passing inspection."
            ),
            code="provider_required_during_import",
            details={"error_type": type(exc).__name__},
        )
    return TargetLoadError(
        f"failed to import {module_path}: {exc}",
        code="target_import_failed",
        details={"error_type": type(exc).__name__},
    )


def _factory_required_parameters(func: Any) -> tuple[str, ...]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return ()
    required: list[str] = []
    skip_kinds = {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }
    for name, parameter in signature.parameters.items():
        if parameter.kind in skip_kinds:
            continue
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return tuple(required)


def _invoke_factory(value: Any, *, module_path: Path, attribute: str) -> Any:
    if inspect.isclass(value):
        raise TargetLoadError(
            (
                f"{module_path}:{attribute}() names a class. AgentCheck only calls "
                "explicit zero-argument factory functions, not classes."
            ),
            code="target_factory_unsupported",
            details={"attribute": attribute},
        )
    if not callable(value):
        raise TargetLoadError(
            (
                f"{module_path}:{attribute}() is not callable. The factory form "
                "requires a zero-argument function."
            ),
            code="target_factory_unsupported",
            details={"attribute": attribute},
        )
    if inspect.iscoroutinefunction(value) or inspect.isasyncgenfunction(value):
        raise TargetLoadError(
            (
                f"{module_path}:{attribute}() is async. AgentCheck does not run "
                "async factories during discovery."
            ),
            code="target_factory_unsupported",
            details={"attribute": attribute},
        )
    required = _factory_required_parameters(value)
    if required:
        names = ", ".join(required)
        raise TargetLoadError(
            (
                f"{module_path}:{attribute}() requires arguments ({names}). "
                "AgentCheck only calls zero-argument factories declared as "
                "path.py:name()."
            ),
            code="target_factory_unsupported",
            details={"attribute": attribute, "required_parameters": list(required)},
        )
    try:
        result = value()
    except TargetLoadError:
        raise
    except Exception as exc:
        if _provider_required_during_import(exc):
            raise TargetLoadError(
                (
                    "Factory execution failed because discovery required a "
                    "provider credential. AgentCheck does not pass OPENAI_API_KEY "
                    "unless it is listed in environment_allowlist. This is not a "
                    "passing inspection."
                ),
                code="provider_required_during_import",
                details={"attribute": attribute, "error_type": type(exc).__name__},
            ) from exc
        raise TargetLoadError(
            f"factory {attribute}() failed: {exc}",
            code="target_factory_failed",
            details={"attribute": attribute, "error_type": type(exc).__name__},
        ) from exc
    if inspect.iscoroutine(result) or inspect.isasyncgen(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        raise TargetLoadError(
            (
                f"{module_path}:{attribute}() returned an awaitable. AgentCheck "
                "does not run async factories during discovery."
            ),
            code="target_factory_unsupported",
            details={"attribute": attribute},
        )
    return result


def _entrypoint_label(module_path: Path, root: Path, attribute: str, *, factory: bool) -> str:
    try:
        relative = module_path.relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = module_path.name
    suffix = "()" if factory else ""
    return f"{relative}:{attribute}{suffix}"


def _load_standalone_module(module_path: Path, root: Path) -> ModuleType:
    digest = hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_agentcheck_target_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise TargetLoadError(
            f"could not create an import spec for {module_path}",
            code="target_import_failed",
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with _target_import_context(root):
            spec.loader.exec_module(module)
    except TargetLoadError:
        sys.modules.pop(module_name, None)
        raise
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise _import_failure(module_path, exc) from exc
    return module


def _load_package_module(module_path: Path, root: Path, relative: Path) -> ModuleType:
    module_name = _dotted_module_name(relative)
    _evict_stale_modules(module_name, root=root)
    try:
        with _target_import_context(root):
            module = importlib.import_module(module_name)
    except TargetLoadError:
        raise
    except Exception as exc:
        raise _import_failure(module_path, exc) from exc
    origin = getattr(module, "__file__", None)
    if origin is None:
        raise TargetLoadError(
            f"import {module_name!r} did not produce a source file for {module_path}",
            code="target_import_failed",
        )
    loaded = Path(origin).resolve()
    if loaded != module_path.resolve():
        raise TargetLoadError(
            f"import {module_name!r} resolved to {loaded}, not the configured entrypoint",
            code="target_import_failed",
        )
    return module


def _load_module(module_path: Path, *, root: Path) -> ModuleType:
    root = root.resolve()
    module_path = _require_contained(module_path, root)
    relative = module_path.relative_to(root)
    if len(relative.parts) == 1 and relative.name != "__init__.py":
        return _load_standalone_module(module_path, root)
    return _load_package_module(module_path, root, relative)


def load_target(target: str | Path) -> tuple[Any, str]:
    """Import the configured object and return it with canonical provenance."""

    module_path, attribute, root, factory = _resolve_target(target)
    module = _load_module(module_path, root=root)
    if not hasattr(module, attribute):
        raise TargetLoadError(
            f"{module_path} does not export {attribute!r}",
            code="target_attribute_missing",
            details={"attribute": attribute},
        )
    value = getattr(module, attribute)
    if factory:
        value = _invoke_factory(value, module_path=module_path, attribute=attribute)
        return value, f"{module_path}:{attribute}()"
    if inspect.isfunction(value) or inspect.ismethod(value):
        label = _entrypoint_label(module_path, root, attribute, factory=True)
        raise TargetLoadError(
            (
                f"{module_path} exports {attribute!r} as a function, not an agent "
                "object. AgentCheck does not auto-call factory functions. To opt in, "
                f"set entrypoint to {label!r}."
            ),
            code="target_factory_not_declared",
            details={"attribute": attribute},
        )
    if inspect.isclass(value):
        label = _entrypoint_label(module_path, root, attribute, factory=True)
        raise TargetLoadError(
            (
                f"{module_path} exports {attribute!r} as a class, not an agent "
                "instance. AgentCheck requires an Agent instance or an explicit "
                f"factory entrypoint {label!r}."
            ),
            code="unsupported_agent_shape",
            details={"attribute": attribute},
        )
    return value, f"{module_path}:{attribute}"


def inspect_target(target: str | Path) -> Any:
    """Load and inspect a supported local agent through the OpenAI adapter."""

    from agentcheck.adapters.openai_agents import OpenAIAgentsAdapter

    agent, source = load_target(target)
    return OpenAIAgentsAdapter().inspect(agent, source=source)


__all__ = [
    "TargetLoadError",
    "enable_contained_target_imports",
    "inspect_target",
    "load_target",
    "resolve_entrypoint",
]
