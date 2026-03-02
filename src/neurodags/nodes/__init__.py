from __future__ import annotations

import pkgutil
import sys
from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any

from neurodags.loggers import get_logger

NodeCallable = Callable[..., Any]

log = get_logger(__name__)

_NODE_REGISTRY: dict[str, NodeCallable] = {}


def register_node(
    func: NodeCallable | None = None,
    *,
    name: str | None = None,
    override: bool = False,
) -> NodeCallable:
    """Register a callable node factory.

    Can be used as ``@register_node`` or ``@register_node(name="alias")``.
    """

    def decorator(target: NodeCallable) -> NodeCallable:
        key = name or target.__name__
        if not override and key in _NODE_REGISTRY and _NODE_REGISTRY[key] is not target:
            raise ValueError(f"Node '{key}' is already registered")
        _NODE_REGISTRY[key] = target
        return target

    if func is not None:
        return decorator(func)

    return decorator


def get_node(name: str) -> NodeCallable:
    """Return a registered node by name."""
    try:
        return _NODE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_NODE_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown node '{name}'. Available nodes: {available}") from exc


def iter_nodes() -> Iterable[tuple[str, NodeCallable]]:
    """Yield registered nodes as ``(name, callable)`` pairs."""
    return _NODE_REGISTRY.items()


def list_nodes() -> tuple[str, ...]:
    """Return the registered node names sorted alphabetically."""
    return tuple(sorted(_NODE_REGISTRY))


def clear_node_registry() -> None:
    """Remove all registered nodes (primarily useful for tests)."""
    _NODE_REGISTRY.clear()


def unregister_node(name: str) -> None:
    """Remove a node from the registry if present."""
    _NODE_REGISTRY.pop(name, None)


def discover(package: str | None = None) -> None:
    """Import all modules in the package to trigger registrations."""
    package_name = package or __name__
    module = sys.modules[package_name]
    if not hasattr(module, "__path__"):
        return

    for mod_info in pkgutil.iter_modules(module.__path__):
        if mod_info.name.startswith("_"):
            continue
        module_name = f"{package_name}.{mod_info.name}"
        try:
            import_module(module_name)
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency failure
            missing_name = getattr(exc, "name", None)
            log.warning(
                "Skipping node module due to missing dependency",
                module=module_name,
                missing=missing_name,
            )
        except ImportError as exc:  # pragma: no cover - misconfigured module
            log.warning(
                "Skipping node module because it failed to import",
                module=module_name,
                error=str(exc),
            )


def register_node_with_name(name: str, func: Callable) -> None:
    func.__name__ = name
    register_node(func)  # reuse existing path


# Eagerly discover nodes so registry is populated on package import.
discover()

__all__ = [
    "clear_node_registry",
    "discover",
    "get_node",
    "iter_nodes",
    "list_nodes",
    "register_node",
    "register_node_with_name",
    "unregister_node",
]
