"""Тесты SQLite-хранилища результатов анализа."""
from __future__ import annotations

import pytest

SAMPLE_RESULT = {
    "transcript": [{"speaker": "Оператор", "start": 0.0, "end": 2.0, "text": "Добрый день"}],
    "classification": {"topic": "кредиты", "priority": "high"},
    "quality_score": {"total": 78, "checklist": {"greeting": True, "need_detection": True,
                                                  "solution_provided": True, "farewell": False}},
    "compliance": {"passed": False, "issues": ["гарантированное одобрение"]},
    "summary": "Клиент спрашивал про кредит.",
    "action_items": ["Перезвонить"],
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    import storage

    monkeypatch.setattr(storage.settings, "db_path", str(tmp_path / "test.db"))
    storage.init_db()
    return storage


def test_init_db_is_idempotent(store):
    store.init_db()
    store.init_db()
    assert store.recent_analyses() == []


def test_save_analysis_returns_row_id(store):
    row_id = store.save_analysis(SAMPLE_RESULT)
    assert isinstance(row_id, int)
    assert row_id > 0


def test_recent_analyses_returns_saved_fields(store):
    store.save_analysis(SAMPLE_RESULT)
    rows = store.recent_analyses()
    assert len(rows) == 1
    row = rows[0]
    assert row["topic"] == "кредиты"
    assert row["priority"] == "high"
    assert row["quality_total"] == 78
    assert row["compliance_passed"] is False
    assert row["summary"] == "Клиент спрашивал про кредит."
    assert row["created_at"]


def test_recent_analyses_respects_limit_and_order(store):
    for i in range(5):
        result = dict(SAMPLE_RESULT, summary=f"звонок {i}")
        store.save_analysis(result)
    rows = store.recent_analyses(limit=2)
    assert len(rows) == 2
    assert rows[0]["summary"] == "звонок 4", "последние записи должны идти первыми"


def test_save_analysis_tolerates_missing_agent_output(store):
    """Оркестратор кладёт {"error": ...} вместо результата упавшего агента — хранилище не должно на этом падать."""
    broken = dict(SAMPLE_RESULT, classification={"error": "LLM недоступна"})
    row_id = store.save_analysis(broken)
    assert row_id > 0
    assert store.recent_analyses()[0]["topic"] is None


@pytest.mark.asyncio
async def test_async_wrappers(store):
    await store.save_analysis_async(SAMPLE_RESULT)
    rows = await store.recent_analyses_async()
    assert len(rows) == 1
