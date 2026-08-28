"""Агент трендов (бонусное задание ТЗ): анализирует не один звонок, а набор уже проанализированных, и ищет в них повторяющиеся паттерны."""
from __future__ import annotations

from typing import Any

from agents.base import BaseAgent
from logging_utils import log_agent_io

SYSTEM_PROMPT = """Ты — аналитик контакт-центра МТБанк. На вход тебе даётся
сводка по нескольким последним звонкам: тема, приоритет, оценка качества
работы оператора, результат compliance-проверки и краткое резюме.

Найди в этих данных закономерности, которые важны супервайзеру:
- какие темы обращений преобладают,
- есть ли связь между темой и падением качества обслуживания,
- повторяются ли одни и те же проблемы клиентов,
- где чаще всего срабатывает compliance.

Верни ТОЛЬКО JSON:
{"top_topics": ["тема1", "тема2"],
 "patterns": ["наблюдение одной строкой", "..."],
 "quality_observations": ["наблюдение про качество", "..."],
 "recommendations": ["конкретная рекомендация супервайзеру", "..."]}
"""


def _calls_to_text(calls: list[dict[str, Any]]) -> str:
    """Компактная сводка вместо сырых транскриптов: на 20 звонках полные транскрипты не влезут в контекст модели, а для поиска паттернов достаточно метаданных и резюме."""
    lines = []
    for call in calls:
        compliance = call.get("compliance_passed")
        compliance_text = "—" if compliance is None else ("пройден" if compliance else "НАРУШЕНИЯ")
        lines.append(
            f"#{call.get('id')} [{call.get('created_at')}] "
            f"тема={call.get('topic')} приоритет={call.get('priority')} "
            f"качество={call.get('quality_total')} compliance={compliance_text} "
            f"резюме: {call.get('summary')}"
        )
    return "\n".join(lines)


class TrendsAgent(BaseAgent):
    name = "trends"

    @log_agent_io("trends")
    async def run(self, calls: list[dict[str, Any]]) -> dict[str, Any]:
        if not calls:
            return {
                "period_calls": 0,
                "top_topics": [],
                "patterns": [],
                "quality_observations": [],
                "recommendations": [],
                "note": "Недостаточно данных: в истории нет проанализированных звонков.",
            }

        result = await self.llm.complete_json(SYSTEM_PROMPT, _calls_to_text(calls))
        return {
            "period_calls": len(calls),
            "top_topics": result.get("top_topics", []),
            "patterns": result.get("patterns", []),
            "quality_observations": result.get("quality_observations", []),
            "recommendations": result.get("recommendations", []),
            "note": "",
        }
