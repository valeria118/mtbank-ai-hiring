"""Тесты ожидания готовности OpenWebUI при провижининге."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import provision_openwebui  # noqa: E402


class _Resp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def test_wait_timeout_covers_first_boot_model_download():
    """Бюджет ожидания должен покрывать холодную загрузку embedding-модели."""
    assert provision_openwebui.WAIT_TIMEOUT_SEC >= 600


def test_wait_healthy_keeps_polling_until_ready(monkeypatch):
    """Пока OpenWebUI грузит модель, /health рвёт соединение — это не повод сдаваться, ждём до истечения бюджета."""
    calls = {"n": 0}

    def fake_get(url, timeout=5):
        calls["n"] += 1
        if calls["n"] < 4:
            raise httpx.ConnectError("connection refused")
        return _Resp(200)

    monkeypatch.setattr(provision_openwebui.httpx, "get", fake_get)
    monkeypatch.setattr(provision_openwebui.time, "sleep", lambda _: None)

    provision_openwebui._wait_healthy("http://openwebui:8080")

    assert calls["n"] == 4


def test_wait_healthy_raises_after_budget(monkeypatch):
    """Если сервис так и не поднялся — понятная ошибка, а не вечный цикл."""
    monkeypatch.setattr(
        provision_openwebui.httpx,
        "get",
        lambda url, timeout=5: (_ for _ in ()).throw(httpx.ConnectError("down")),
    )
    monkeypatch.setattr(provision_openwebui.time, "sleep", lambda _: None)

    clock = {"t": 0.0}

    def fake_monotonic():
        clock["t"] += 30.0
        return clock["t"]

    monkeypatch.setattr(provision_openwebui.time, "monotonic", fake_monotonic)

    with pytest.raises(TimeoutError):
        provision_openwebui._wait_healthy("http://openwebui:8080")
