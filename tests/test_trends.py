"""Unit-тесты агента трендов. LLM замокан — как и у остальных агентов."""
from __future__ import annotations

import pytest

from agents.trends import TrendsAgent
from tests.test_agents import FakeLLMClient

SAMPLE_CALLS = [
    {"id": 3, "created_at": "2026-08-24T10:00:00", "topic": "жалобы", "priority": "high",
     "quality_total": 45, "compliance_passed": False, "summary": "Клиент жалуется на блокировку карты."},
    {"id": 2, "created_at": "2026-08-24T09:30:00", "topic": "жалобы", "priority": "high",
     "quality_total": 52, "compliance_passed": True, "summary": "Клиент недоволен сроками перевода."},
    {"id": 1, "created_at": "2026-08-24T09:00:00", "topic": "кредиты", "priority": "medium",
     "quality_total": 88, "compliance_passed": True, "summary": "Вопрос по ставке кредита."},
]

LLM_RESPONSE = {
    "top_topics": ["жалобы", "кредиты"],
    "patterns": ["Две трети обращений — жалобы с высоким приоритетом"],
    "quality_observations": ["На жалобах quality_score падает почти вдвое"],
    "recommendations": ["Провести обучение операторов по работе с конфликтными клиентами"],
}


@pytest.mark.asyncio
async def test_trends_agent_returns_all_sections():
    agent = TrendsAgent(FakeLLMClient(LLM_RESPONSE))
    result = await agent.run(SAMPLE_CALLS)
    assert result["period_calls"] == 3
    assert result["top_topics"] == ["жалобы", "кредиты"]
    assert result["patterns"] == LLM_RESPONSE["patterns"]
    assert result["quality_observations"] == LLM_RESPONSE["quality_observations"]
    assert result["recommendations"] == LLM_RESPONSE["recommendations"]


@pytest.mark.asyncio
async def test_trends_agent_on_empty_history_does_not_call_llm():
    class ExplodingLLM:
        async def complete_json(self, system_prompt, user_prompt):
            raise AssertionError("LLM не должна вызываться на пустой истории")

    agent = TrendsAgent(ExplodingLLM())
    result = await agent.run([])
    assert result["period_calls"] == 0
    assert result["patterns"] == []
    assert "недостаточно" in result["note"].lower()


@pytest.mark.asyncio
async def test_trends_agent_tolerates_missing_llm_fields():
    agent = TrendsAgent(FakeLLMClient({}))
    result = await agent.run(SAMPLE_CALLS)
    assert result["top_topics"] == []
    assert result["patterns"] == []
    assert result["recommendations"] == []


@pytest.mark.asyncio
async def test_trends_agent_passes_compact_history_to_llm():
    """В промпт уходит сводка, а не сырые транскрипты — иначе на 20 звонках вылезем за контекст модели."""
    captured = {}

    class CapturingLLM:
        async def complete_json(self, system_prompt, user_prompt):
            captured["user_prompt"] = user_prompt
            return LLM_RESPONSE

    agent = TrendsAgent(CapturingLLM())
    await agent.run(SAMPLE_CALLS)

    assert "жалобы" in captured["user_prompt"]
    assert "45" in captured["user_prompt"]
    assert "transcript" not in captured["user_prompt"]
