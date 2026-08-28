"""JSON-логирование входа/выхода каждого агента, как того требует ТЗ (раздел «Технический стек» README.md — «JSON-логи с входом/выходом каждого агента»)."""
import json
import logging
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

from config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            payload.update(record.extra_data)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.log_json else logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(settings.log_level)
    logger.propagate = False
    return logger


def log_extra(logger: logging.Logger, level: int, message: str, **data: Any) -> None:
    logger.log(level, message, extra={"extra_data": data})


@contextmanager
def timed(logger: logging.Logger, step_name: str, **context: Any):
    """Контекст-менеджер: логирует старт/финиш шага с длительностью."""
    start = time.perf_counter()
    log_extra(logger, logging.INFO, f"{step_name}.start", step=step_name, **context)
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        log_extra(
            logger, logging.ERROR, f"{step_name}.error",
            step=step_name, error=str(exc), **context,
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        log_extra(
            logger, logging.INFO, f"{step_name}.done",
            step=step_name, duration_ms=duration_ms, **context,
        )


def log_agent_io(agent_name: str) -> Callable:
    """Декоратор для методов агентов вида `async def run(self, input) -> output`. Логирует вход и выход в JSON, как требует ТЗ."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(self, *args, **kwargs):
            logger = get_logger(f"agent.{agent_name}")
            agent_input = args[0] if args else kwargs
            log_extra(
                logger, logging.INFO, "agent.input",
                agent=agent_name,
                **_payload("input", agent_input),
            )
            start = time.perf_counter()
            try:
                result = await fn(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                log_extra(
                    logger, logging.ERROR, "agent.error",
                    agent=agent_name, error=str(exc),
                )
                raise
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            log_extra(
                logger, logging.INFO, "agent.output",
                agent=agent_name,
                duration_ms=duration_ms,
                **_payload("output", result),
            )
            return result

        return wrapper

    return decorator


def _serialize(obj: Any) -> str:
    return obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, default=str)


def _payload(field: str, obj: Any) -> dict[str, Any]:
    """Поля лога для входа/выхода агента."""
    text = _serialize(obj)
    limit = settings.log_truncate_chars
    if limit and len(text) > limit:
        return {
            field: text[:limit] + "…",
            f"{field}_truncated": True,
            f"{field}_chars": len(text),
        }
    return {field: text}
