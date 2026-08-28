"""Prometheus-метрики для Grafana-дашборда (бонусное задание ТЗ): количество звонков, распределение quality_score, топ тематик."""
from __future__ import annotations

from typing import Any

from prometheus_client import Counter, Histogram, start_http_server

from logging_utils import get_logger

logger = get_logger("metrics")

_exporter_started = False

CALLS_TOTAL = Counter(
    "mtbank_calls_total",
    "Всего проанализировано звонков",
)

CALLS_BY_TOPIC = Counter(
    "mtbank_calls_by_topic_total",
    "Звонки по тематике обращения",
    ["topic"],
)

CALLS_BY_PRIORITY = Counter(
    "mtbank_calls_by_priority_total",
    "Звонки по приоритету",
    ["priority"],
)

QUALITY_SCORE = Histogram(
    "mtbank_quality_score",
    "Распределение оценки качества обслуживания",
    buckets=(20, 40, 50, 60, 70, 80, 90, 100),
)

COMPLIANCE_FAILED = Counter(
    "mtbank_compliance_failed_total",
    "Звонки с найденными compliance-нарушениями",
)

ANALYSES_DEGRADED = Counter(
    "mtbank_analyses_degraded_total",
    "Анализы, где хотя бы один агент не отработал",
    ["agent"],
)


def start_exporter(port: int) -> None:
    """Поднять отдельный HTTP-сервер с /metrics в текущем процессе."""
    global _exporter_started
    if _exporter_started:
        return
    try:
        start_http_server(port)
    except OSError as exc:
        logger.warning(
            "metrics.exporter_failed",
            extra={"extra_data": {"port": port, "error": str(exc)}},
        )
        return
    _exporter_started = True
    logger.info("metrics.exporter_started", extra={"extra_data": {"port": port}})


def _safe_get(section: Any, key: str) -> Any:
    if isinstance(section, dict):
        return section.get(key)
    return None


def record_degraded(errors: dict[str, str]) -> None:
    """Отметить анализ, где часть агентов не отработала."""
    for agent in errors:
        ANALYSES_DEGRADED.labels(agent=agent).inc()


def record_analysis(result: dict[str, Any]) -> None:
    """Обновить метрики по одному завершённому анализу."""
    CALLS_TOTAL.inc()

    topic = _safe_get(result.get("classification"), "topic")
    if topic:
        CALLS_BY_TOPIC.labels(topic=topic).inc()

    priority = _safe_get(result.get("classification"), "priority")
    if priority:
        CALLS_BY_PRIORITY.labels(priority=priority).inc()

    total = _safe_get(result.get("quality_score"), "total")
    if isinstance(total, (int, float)):
        QUALITY_SCORE.observe(total)

    passed = _safe_get(result.get("compliance"), "passed")
    if passed is False:
        COMPLIANCE_FAILED.inc()
