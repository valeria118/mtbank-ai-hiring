"""JSON-логирование входа/выхода агентов — отдельный пункт технического стека в ТЗ, и до этого он был единственным требованием без единого теста."""
from __future__ import annotations

import json
import logging

import pytest

import logging_utils
from logging_utils import JsonFormatter, log_agent_io, timed


def _record(message: str, **extra_data) -> logging.LogRecord:
    record = logging.LogRecord(
        name="agent.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=message, args=(), exc_info=None,
    )
    if extra_data:
        record.extra_data = extra_data
    return record


def test_json_formatter_emits_parseable_json_with_extra_fields():
    payload = json.loads(JsonFormatter().format(_record("agent.input", agent="quality", input="текст")))
    assert payload["message"] == "agent.input"
    assert payload["logger"] == "agent.test"
    assert payload["level"] == "INFO"
    assert payload["agent"] == "quality"
    assert payload["input"] == "текст"


def test_json_formatter_keeps_cyrillic_readable():
    r"""ensure_ascii=False: иначе русский текст в логах превращается в \uXXXX и лог перестаёт быть читаемым глазами при разборе инцидента."""
    line = JsonFormatter().format(_record("agent.output", output="Клиент недоволен"))
    assert "Клиент недоволен" in line


@pytest.fixture
def agent_logs(caplog):
    """caplog ловит записи через корневой логгер, а get_logger выставляет propagate=False (чтобы в проде запись не дублировалась своим хендлером и корневым)."""
    logger = logging_utils.get_logger("agent.tester")
    original = logger.propagate
    logger.propagate = True
    try:
        yield caplog
    finally:
        logger.propagate = original


def _extra(caplog, message: str) -> dict:
    return next(r.extra_data for r in caplog.records if r.message == message)


class _Agent:
    @log_agent_io("tester")
    async def run(self, transcript):
        return {"verdict": "ok", "transcript_len": len(transcript)}


class _FailingAgent:
    @log_agent_io("tester")
    async def run(self, transcript):
        raise RuntimeError("LLM недоступна")


@pytest.mark.asyncio
async def test_agent_io_logs_full_input_by_default(monkeypatch, agent_logs):
    """Требование ТЗ — логировать вход и выход каждого агента."""
    monkeypatch.setattr(logging_utils.settings, "log_truncate_chars", 0)
    long_transcript = [{"speaker": "Клиент", "start": 0.0, "end": 1.0, "text": "а" * 2000}]

    with agent_logs.at_level(logging.INFO, logger="agent.tester"):
        await _Agent().run(long_transcript)

    logged_input = _extra(agent_logs, "agent.input")
    assert "…" not in logged_input["input"]
    assert logged_input["input"].count("а") == 2000
    assert "input_truncated" not in logged_input

    logged_output = _extra(agent_logs, "agent.output")
    assert logged_output["agent"] == "tester"
    assert "duration_ms" in logged_output
    assert "ok" in logged_output["output"]


@pytest.mark.asyncio
async def test_agent_io_marks_truncation_explicitly(monkeypatch, agent_logs):
    """Если обрезка всё-таки включена, она не прячется: по логу должно быть видно, что текст неполный, и какой была полная длина."""
    monkeypatch.setattr(logging_utils.settings, "log_truncate_chars", 50)

    with agent_logs.at_level(logging.INFO, logger="agent.tester"):
        await _Agent().run([{"speaker": "Клиент", "start": 0.0, "end": 1.0, "text": "б" * 500}])

    entry = _extra(agent_logs, "agent.input")
    assert entry["input"].endswith("…")
    assert entry["input_truncated"] is True
    assert entry["input_chars"] > 50


@pytest.mark.asyncio
async def test_agent_io_logs_and_reraises_failure(agent_logs):
    with agent_logs.at_level(logging.ERROR, logger="agent.tester"):
        with pytest.raises(RuntimeError, match="LLM недоступна"):
            await _FailingAgent().run([])

    assert _extra(agent_logs, "agent.error") == {"agent": "tester", "error": "LLM недоступна"}


def test_timed_logs_duration_and_reraises(caplog):
    logger = logging.getLogger("step.tester")
    with caplog.at_level(logging.INFO, logger="step.tester"):
        with timed(logger, "analyze_request", filename="call.wav"):
            pass
    done = next(r.extra_data for r in caplog.records if r.message == "analyze_request.done")
    assert done["filename"] == "call.wav"
    assert done["duration_ms"] >= 0

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="step.tester"):
        with pytest.raises(ValueError):
            with timed(logger, "analyze_request", filename="broken.wav"):
                raise ValueError("битый файл")
    failed = next(r.extra_data for r in caplog.records if r.message == "analyze_request.error")
    assert failed["error"] == "битый файл"
