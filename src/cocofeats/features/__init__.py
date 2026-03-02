from __future__ import annotations

import pkgutil
import sys
from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any

from cocofeats.loggers import get_logger

DerivativeCallable = Callable[..., Any]


class DerivativeEntry:
    """Bundle a derivative callable with its definition."""

    def __init__(
        self,
        name: str,
        func: DerivativeCallable,
        definition: dict[str, Any] | None = None,
    ):
        self.name = name
        self.func = func
        self.definition = definition or {}

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<DerivativeEntry name={self.name!r} func={self.func!r}>"


# Global registry
_DERIVATIVE_REGISTRY: dict[str, DerivativeEntry] = {}

log = get_logger(__name__)


# -------------------------
# Registration API
# -------------------------


def register_derivative(
    func: DerivativeCallable | None = None,
    *,
    name: str | None = None,
    override: bool = False,
    definition: dict[str, Any] | None = None,
) -> DerivativeCallable:
    """Register a derivative callable with optional definition.

    Can be used as:

        @register_derivative
        def my_derivative(...): ...

    or:

        @register_derivative(name="alias", definition=defn)

    or programmatically with `register_derivative_with_name`.
    """

    def decorator(target: DerivativeCallable) -> DerivativeCallable:
        key = name or target.__name__
        if not override and key in _DERIVATIVE_REGISTRY and _DERIVATIVE_REGISTRY[key].func is not target:
            raise ValueError(f"Derivative '{key}' is already registered")
        elif override and key in _DERIVATIVE_REGISTRY:
            log.info(f"Overriding existing derivative registration for '{key}'")

        _DERIVATIVE_REGISTRY[key] = DerivativeEntry(key, target, definition)
        return target

    if func is not None:
        return decorator(func)

    return decorator


def register_derivative_with_name(
    name: str,
    func: DerivativeCallable,
    definition: dict[str, Any] | None = None,
    override: bool = False,
) -> None:
    """Register a derivative under a specific name."""
    func.__name__ = name
    register_derivative(func, name=name, definition=definition, override=override)


# -------------------------
# Lookup API
# -------------------------


def get_derivative(name: str) -> DerivativeEntry:
    """Return a registered derivative entry by name."""
    try:
        return _DERIVATIVE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_DERIVATIVE_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown derivative '{name}'. Available derivatives: {available}") from exc


def iter_derivatives() -> Iterable[tuple[str, DerivativeEntry]]:
    """Yield registered derivatives as (name, DerivativeEntry) pairs."""
    return _DERIVATIVE_REGISTRY.items()


def list_derivatives() -> tuple[str, ...]:
    """Return the registered derivative names sorted alphabetically."""
    return tuple(sorted(_DERIVATIVE_REGISTRY))


def clear_derivative_registry() -> None:
    """Remove all registered derivatives (useful for tests)."""
    _DERIVATIVE_REGISTRY.clear()


def unregister_derivative(name: str) -> None:
    """Remove a derivative from the registry if present."""
    _DERIVATIVE_REGISTRY.pop(name, None)


# -------------------------
# Convenience
# -------------------------

__all__ = [
    "DerivativeEntry",
    "clear_derivative_registry",
    "get_derivative",
    "iter_derivatives",
    "list_derivatives",
    "register_derivative",
    "register_derivative_with_name",
    "unregister_derivative",
]


def discover(package: str | None = None) -> None:
    """Import all modules in the package to trigger registrations."""
    package_name = package or __name__
    module = sys.modules[package_name]
    if not hasattr(module, "__path__"):
        return

    for mod_info in pkgutil.iter_modules(module.__path__):
        if mod_info.name.startswith("_"):
            continue
        import_module(f"{package_name}.{mod_info.name}")


discover()
