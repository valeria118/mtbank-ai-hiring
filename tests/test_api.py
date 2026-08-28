"""Тесты контракта POST /analyze — структура запроса и ответа задана ТЗ."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

EXPECTED_RESULT = {
    "transcript": [{"speaker": "Оператор", "start": 0.0, "end": 4.2, "text": "Добрый день"}],
    "classification": {"topic": "кредиты", "priority": "medium"},
    "quality_score": {
        "total": 78,
        "checklist": {
            "greeting": True,
            "need_detection": True,
            "solution_provided": True,
            "farewell": False,
        },
    },
    "compliance": {"passed": True, "issues": []},
    "summary": "Клиент обратился по вопросу кредита.",
    "action_items": ["Отправить КП на email клиента"],
}

@pytest.fixture
def client(monkeypatch):
    from api import main as api_main

    async def fake_analyze(audio_path: Path) -> dict:
        assert audio_path.exists(), "во временный файл должно быть записано аудио"
        return EXPECTED_RESULT

    monkeypatch.setattr(api_main.pipeline, "analyze", fake_analyze)
    return TestClient(api_main.app)

def test_analyze_accepts_multipart_file(client):
    resp = client.post(
        "/analyze",
        files={"file": ("call.wav", io.BytesIO(b"RIFF0000WAVEfmt "), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json() == EXPECTED_RESULT

def test_analyze_accepts_json_url_body(client, monkeypatch):
    """ТЗ: Body: file=<audio> или { "url": "https://..." }"""
    import httpx

    class FakeResponse:
        content = b"RIFF0000WAVEfmt "

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    resp = client.post("/analyze", json={"url": "https://example.com/call.wav"})
    assert resp.status_code == 200
    assert resp.json() == EXPECTED_RESULT

def test_analyze_rejects_empty_request(client):
    resp = client.post("/analyze")
    assert resp.status_code == 400
    assert "file" in resp.json()["detail"] or "url" in resp.json()["detail"]

def test_analyze_response_has_all_required_keys(client):
    resp = client.post(
        "/analyze",
        files={"file": ("call.wav", io.BytesIO(b"RIFF0000WAVEfmt "), "audio/wav")},
    )
    body = resp.json()
    for key in ("transcript", "classification", "quality_score", "compliance", "summary", "action_items"):
        assert key in body, f"ТЗ требует ключ {key} в ответе"

def test_health_endpoint():
    from api.main import app

    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def _degraded_client(monkeypatch, errors: dict[str, str]):
    from api import main as api_main

    async def fake_analyze(audio_path: Path) -> dict:
        return {**EXPECTED_RESULT, "errors": errors}

    monkeypatch.setattr(api_main.pipeline, "analyze", fake_analyze)
    return TestClient(api_main.app)

def test_analyze_returns_502_when_every_agent_failed(monkeypatch):
    """Отдавать полный отказ как успешный анализ нельзя: клиент не отличит «нарушений нет» от «проверка не выполнялась»."""
    from pipeline import Pipeline

    all_failed = {section: "LLM недоступна" for section in Pipeline.AGENT_SECTIONS}
    resp = _degraded_client(monkeypatch, all_failed).post(
        "/analyze",
        files={"file": ("call.wav", io.BytesIO(b"RIFF0000WAVEfmt "), "audio/wav")},
    )
    assert resp.status_code == 502
    assert resp.json()["transcript"]
    assert set(resp.json()["errors"]) == set(Pipeline.AGENT_SECTIONS)

def test_analyze_returns_200_with_errors_on_partial_failure(monkeypatch):
    resp = _degraded_client(monkeypatch, {"compliance": "LLM недоступна"}).post(
        "/analyze",
        files={"file": ("call.wav", io.BytesIO(b"RIFF0000WAVEfmt "), "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json()["errors"] == {"compliance": "LLM недоступна"}

def test_analyze_maps_bad_audio_to_400(monkeypatch):
    """Ошибка данных клиента — 400, а не 500: обработка ошибок это явный подпункт критерия ASR."""
    from api import main as api_main

    async def fake_analyze(audio_path: Path) -> dict:
        raise ValueError("Файл call.wav не распознан как аудио")

    monkeypatch.setattr(api_main.pipeline, "analyze", fake_analyze)
    resp = TestClient(api_main.app).post(
        "/analyze",
        files={"file": ("call.wav", io.BytesIO(b"not audio"), "audio/wav")},
    )
    assert resp.status_code == 400
    assert "не распознан" in resp.json()["detail"]

def test_openapi_documents_both_request_bodies_and_response():
    """На /docs эндпоинт выглядел как POST без параметров: Content-Type разбирается вручную, и FastAPI нечего было документировать."""
    from api.main import app

    schema = app.openapi()["paths"]["/analyze"]["post"]
    content = schema["requestBody"]["content"]
    assert "multipart/form-data" in content
    assert "application/json" in content
    assert content["multipart/form-data"]["schema"]["properties"]["file"]["format"] == "binary"
    assert "200" in schema["responses"] and "400" in schema["responses"] and "502" in schema["responses"]
    assert schema["responses"]["200"]["content"]["application/json"]["schema"]
