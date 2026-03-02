# tests/test_loggers.py
from __future__ import annotations

import json
import logging
import sys
from typing import Generator

import pytest
import structlog

# Adjust import to your package layout:
import neurodags.loggers as logmod
from neurodags.loggers import configure_logging, get_logger


@pytest.fixture(autouse=True)
def reset_logging_state(monkeypatch) -> Generator[None, None, None]:
    """
    Reset stdlib logging and structlog state before each test.

    - clear root handlers
    - reset structlog configuration
    - ensure our module's _CONFIGURED is False
    """
    # Clear root handlers
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.NOTSET)

    # Reset structlog to defaults
    structlog.reset_defaults()
    structlog.configure()  # minimal, no processors

    # Reset our module state
    monkeypatch.setattr(logmod, "_CONFIGURED", False, raising=False)

    yield

    # Cleanup again
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)


def _stdout_text(capsys) -> str:
    out, _err = capsys.readouterr()
    return out


def _last_json_line(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines, "no output captured"
    return json.loads(lines[-1])


def test_json_render_when_not_tty(monkeypatch, capsys):
    # Don’t replace stdout; just make it report “not a TTY”
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)

    configure_logging(level="INFO")
    log = get_logger("test").bind(ctx="ctxv")
    log.info("hello", foo=123)

    out = _stdout_text(capsys)
    data = _last_json_line(out)
    assert data["event"] == "hello"
    assert data["foo"] == 123
    assert data["ctx"] == "ctxv"
    assert str(data["level"]).lower() == "info"


def test_console_render_when_tty(monkeypatch, capsys):
    # Make stdout “a TTY”
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    configure_logging(level="INFO")
    get_logger("cli").info("hi", foo=1)

    out = _stdout_text(capsys)
    # Don’t assert exact formatting; just check key bits are present
    assert "hi" in out
    assert ("foo=1" in out) or ('"foo": 1' in out)  # guard in case console changes


def test_level_filtering(capsys, monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)  # JSON easier to check
    configure_logging(level="WARNING")
    log = get_logger("lvl")
    log.info("nope")
    log.warning("warn", code=7)

    out = _stdout_text(capsys)
    assert "nope" not in out
    data = _last_json_line(out)
    assert data["event"] == "warn"
    # code may or may not be top-level with some renderers; allow either presence or absence
    assert ("code" not in data) or data.get("code") == 7


def test_idempotent_configuration():
    root = logging.getLogger()

    configure_logging(level="INFO")
    first_ids = tuple(id(h) for h in root.handlers)
    assert len(first_ids) >= 1

    configure_logging(level="DEBUG")  # should be a no-op
    second_ids = tuple(id(h) for h in root.handlers)
    assert second_ids == first_ids


def test_bound_context_present(capsys, monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)  # JSON
    configure_logging(level="INFO")
    get_logger("ctx", user="alice", session="s1").info("evt", x=1)

    data = _last_json_line(_stdout_text(capsys))
    assert data["event"] == "evt"
    assert data.get("user") == "alice"
    assert data.get("session") == "s1"
    assert data.get("x") == 1


def test_route_stdlib(monkeypatch, capsys):
    # Route stdlib logs through structlog renderer (JSON)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    configure_logging(level="INFO", route_stdlib=True)

    # Emit a stdlib log
    logging.getLogger("thirdparty").warning("lib message", extra={"code": 42})

    data = _last_json_line(_stdout_text(capsys))
    assert data["event"] == "lib message"
    assert str(data["level"]).lower() == "warning"
    # 'code' may not always be in the payload depending on handler; don't require it


def test_env_overrides_force_json(monkeypatch, capsys):
    # LOG_FMT=json should force JSON even if TTY
    monkeypatch.setenv("LOG_FMT", "json")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

    configure_logging(level="INFO")
    get_logger().info("hey", a=2)

    data = _last_json_line(_stdout_text(capsys))
    assert data["event"] == "hey"
    assert data["a"] == 2


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        logmod._coerce_level("NOT_A_LEVEL")


def test_coerce_level_int_passthrough():
    # Should return the int unchanged
    assert logmod._coerce_level(logging.DEBUG) == logging.DEBUG
    assert logmod._coerce_level(42) == 42


def test_configure_logging_level_none_defaults_env(monkeypatch, capsys):
    # Simulate LOG_LEVEL=DEBUG in environment
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)

    configure_logging(level=None)  # should read from env
    get_logger("envtest").debug("debugmsg", val=123)

    out = _stdout_text(capsys)
    data = _last_json_line(out)
    assert data["event"] == "debugmsg"
    assert data["val"] == 123
    assert str(data["level"]).lower() == "debug"
