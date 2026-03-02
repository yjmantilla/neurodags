"""neurodags: A modern Python package template."""

from .example import add

try:
    # provided by hatch-vcs at build / editable install
    from ._version import __version__
except Exception:
    # last-resort fallback when importing directly from a bare git checkout
    __version__ = "0+unknown"

__all__ = ["__version__", "add"]
