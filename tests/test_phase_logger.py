import io
import logging

from app.core.phase_logger import Phase, PhaseLogger


def _logger(monkeypatch, level):
    if level:
        monkeypatch.setenv("LOG_LEVEL", level)
    else:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
    pl = PhaseLogger(f"test-{level}")
    pl.logger.propagate = False
    pl.logger.handlers[0].stream = io.StringIO()
    return pl, pl.logger.handlers[0].stream


def test_default_shows_only_summaries_warnings_and_errors(monkeypatch):
    pl, out = _logger(monkeypatch, None)
    pl.log_phase(Phase.TOOL_EXECUTION, "Tool gestartet")
    pl.log_phase(Phase.AGENT_RESPONSE, "Chat fertig", level=logging.INFO)
    pl.log_phase(Phase.TOOL_EXECUTION, "Timeout", level=logging.WARNING)
    text = out.getvalue()
    assert "Tool gestartet" not in text
    assert "Chat fertig" in text and "Timeout" in text


def test_log_level_debug_shows_the_flow(monkeypatch):
    pl, out = _logger(monkeypatch, "DEBUG")
    pl.log_phase(Phase.TOOL_EXECUTION, "Tool gestartet")
    assert "Tool gestartet" in out.getvalue()
