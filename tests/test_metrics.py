"""Тесты Prometheus-метрик."""
from __future__ import annotations

import pytest

SAMPLE_RESULT = {
    "classification": {"topic": "кредиты", "priority": "medium"},
    "quality_score": {"total": 78, "checklist": {}},
    "compliance": {"passed": True, "issues": []},
    "summary": "…",
    "action_items": [],
}


def _counter_value(counter, **labels):
    if labels:
        return counter.labels(**labels)._value.get()
    return counter._value.get()


def test_record_analysis_increments_total():
    import metrics

    before = _counter_value(metrics.CALLS_TOTAL)
    metrics.record_analysis(SAMPLE_RESULT)
    assert _counter_value(metrics.CALLS_TOTAL) == before + 1


def test_record_analysis_increments_topic_counter():
    import metrics

    before = _counter_value(metrics.CALLS_BY_TOPIC, topic="карты")
    metrics.record_analysis(dict(SAMPLE_RESULT, classification={"topic": "карты", "priority": "low"}))
    assert _counter_value(metrics.CALLS_BY_TOPIC, topic="карты") == before + 1


def test_record_analysis_counts_compliance_failures():
    import metrics

    before = _counter_value(metrics.COMPLIANCE_FAILED)
    metrics.record_analysis(dict(SAMPLE_RESULT, compliance={"passed": False, "issues": ["x"]}))
    assert _counter_value(metrics.COMPLIANCE_FAILED) == before + 1


def test_record_analysis_survives_failed_agent():
    """Оркестратор кладёт {"error": ...} вместо результата — метрики не должны падать и не должны считать это за валидные значения."""
    import metrics

    before = _counter_value(metrics.CALLS_TOTAL)
    metrics.record_analysis({"classification": {"error": "LLM down"},
                             "quality_score": {"error": "LLM down"},
                             "compliance": {"error": "LLM down"}})
    assert _counter_value(metrics.CALLS_TOTAL) == before + 1


def test_record_degraded_counts_each_failed_agent():
    """Деградированный анализ идёт в отдельный счётчик, а не в общие метрики: иначе отсутствующая оценка уезжает в гистограмму нулём, а «комплаенс не отработал» — в счётчик найденных нарушений."""
    import metrics

    before_compliance = _counter_value(metrics.ANALYSES_DEGRADED, agent="compliance")
    before_calls = _counter_value(metrics.CALLS_TOTAL)
    before_failed = _counter_value(metrics.COMPLIANCE_FAILED)

    metrics.record_degraded({"compliance": "LLM недоступна", "summary": "таймаут"})

    assert _counter_value(metrics.ANALYSES_DEGRADED, agent="compliance") == before_compliance + 1
    assert _counter_value(metrics.ANALYSES_DEGRADED, agent="summary") >= 1
    assert _counter_value(metrics.CALLS_TOTAL) == before_calls
    assert _counter_value(metrics.COMPLIANCE_FAILED) == before_failed


def test_start_exporter_is_idempotent_and_survives_busy_port(monkeypatch):
    """Контейнеру pipelines нужен свой /metrics — роут в чужой образ сервера Pipelines не добавить."""
    import metrics

    started = []
    monkeypatch.setattr(metrics, "_exporter_started", False)
    monkeypatch.setattr(metrics, "start_http_server", lambda port: started.append(port))

    metrics.start_exporter(9100)
    metrics.start_exporter(9100)
    assert started == [9100], "второй вызов не должен поднимать сервер повторно"

    def busy(port):
        raise OSError("address already in use")

    monkeypatch.setattr(metrics, "_exporter_started", False)
    monkeypatch.setattr(metrics, "start_http_server", busy)
    metrics.start_exporter(9100)  # не должно бросить


def test_metrics_endpoint_exposes_prometheus_format():
    from fastapi.testclient import TestClient

    from api.main import app

    resp = TestClient(app).get("/metrics")
    assert resp.status_code == 200
    assert "mtbank_calls_total" in resp.text
