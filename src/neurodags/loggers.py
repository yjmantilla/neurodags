# loggers.py
from __future__ import annotations

import logging
import os
import sys

import structlog

_CONFIGURED = False


def _coerce_level(level: str | int) -> int:
    """
    Convert a textual or numeric level to a stdlib logging level.

    Parameters
    ----------
    level : str or int
        Either a string such as ``"INFO"``/``"DEBUG"`` or a numeric level.

    Returns
    -------
    int
        A valid stdlib logging level (e.g., ``logging.INFO``).

    Raises
    ------
    ValueError
        If ``level`` is an unknown string.
    """
    if isinstance(level, int):
        return level
    lvl = logging.getLevelName(level.upper())
    # logging.getLevelName returns int for known names, str otherwise
    if isinstance(lvl, int):
        return lvl
    raise ValueError(f"Unknown log level: {level!r}")


def configure_logging(
    *,
    json: bool | None = None,
    level: str | int | None = None,
    route_stdlib: bool = False,
) -> None:
    """
    Configure structlog and stdlib logging once.

    This sets up a consistent logging pipeline for your app. By default,
    structlog logs are rendered either as pretty console output (TTY) or JSON
    (non-TTY / env setting). If ``route_stdlib`` is enabled, third-party logs
    from the stdlib logging system are also rendered through the same formatter.

    Parameters
    ----------
    json : bool, optional
        Force JSON output. If ``None`` (default), JSON is used when
        standard output is not a TTY or when the environment variable
        ``LOG_FMT=json`` is set.
    level : str or int, optional
        Global log level, e.g., ``"INFO"`` or ``logging.INFO``.
        Defaults to the value of ``$LOG_LEVEL`` or ``"INFO"``.
    route_stdlib : bool, optional
        If ``True``, route stdlib logs (including third-party libraries) through
        structlog's renderer for uniform formatting. Default is ``False``.

    Returns
    -------
    None

    Notes
    -----
    - This function is idempotent and returns immediately on subsequent calls.
    - When ``route_stdlib=True``, a ``ProcessorFormatter`` is used so that
      stdlib logs flow through structlog's renderer.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # ---- defaults from environment / context ----
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    level = _coerce_level(level)

    if json is None:
        # Prefer JSON in non-TTY (batch/HPC/CI) or when explicitly requested
        json = (os.getenv("LOG_FMT", "json").lower() == "json") or (not sys.stdout.isatty())

    # ---- stdlib logging baseline ----
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(level)

    # Remove pre-existing handlers to avoid duplicates (e.g., notebooks or reloads)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if route_stdlib:
        # Route stdlib logs via structlog's renderer
        from structlog.stdlib import ProcessorFormatter

        render = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()

        # stdlib -> ProcessorFormatter -> structlog renderer
        pf = ProcessorFormatter(
            foreign_pre_chain=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.format_exc_info,
            ],
            processors=[render],  # final rendering happens here
        )
        handler.setFormatter(pf)

        # structlog pipeline hands off to the stdlib ProcessorFormatter above
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.filter_by_level,
                ProcessorFormatter.remove_processors_meta,  # hand off to stdlib formatter
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Simpler path: structlog renders structlog logs; stdlib logs use basic formatter
        handler.setFormatter(logging.Formatter("%(message)s"))

        processors = [
            structlog.contextvars.merge_contextvars,  # include contextvars if used
            structlog.stdlib.filter_by_level,  # drop events below level early
            structlog.processors.add_log_level,  # add 'level' field
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,  # clean tracebacks
            structlog.stdlib.add_logger_name,  # logger name field
            structlog.stdlib.PositionalArgumentsFormatter(),
        ]
        render = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()

        structlog.configure(
            processors=[*processors, render],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    root.addHandler(handler)
    _CONFIGURED = True

    # --- Optional tips (uncomment as needed) ---------------------------------
    # # Reduce noisy libraries when routing stdlib logs:
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    # logging.getLogger("asyncio").setLevel(logging.ERROR)
    #
    # # Toggle routing via env (call with route_stdlib=(os.getenv("LOG_ROUTE_STDLIB") == "true")):
    # # configure_logging(route_stdlib=(os.getenv("LOG_ROUTE_STDLIB", "false").lower() == "true"))
    #
    # # Bind common context once (service name, version, etc.):
    # # import structlog
    # # structlog.contextvars.bind_contextvars(service="neurodags", version="1.2.3")
    #
    # # Emit JSON in local dev too:
    # # configure_logging(json=True)
    #
    # # Use ultra-compact console rendering:
    # # render = structlog.dev.ConsoleRenderer(colors=True)
    # # (swap into the configuration above)


def get_logger(name: str | None = None, **bind):
    """
    Get a bound structlog logger.

    Parameters
    ----------
    name : str or None, optional
        Logger name. If ``None``, uses the current module name.
    **bind
        Key/value pairs to bind immediately to the logger's context.

    Returns
    -------
    structlog.stdlib.BoundLogger
        A logger that renders via the configured structlog pipeline.
    """
    log = structlog.get_logger(name or __name__)
    return log.bind(**bind) if bind else log
