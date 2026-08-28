from __future__ import annotations

from typing import Any

from agents.base import BaseAgent, transcript_to_text
from logging_utils import log_agent_io

SYSTEM_PROMPT = """Ты — супервайзер контакт-центра МТБанк, оцениваешь качество
работы оператора по транскрипту звонка. Проверь чеклист (каждый пункт true/false):
- greeting: оператор поприветствовал клиента и представился
- need_detection: оператор выяснил потребность клиента (задал уточняющие вопросы)
- solution_provided: клиенту было предложено конкретное решение/продукт/ответ
- farewell: оператор корректно попрощался

Также поставь итоговую оценку total от 0 до 100 (учитывай тон, вежливость,
полноту ответа, а не только чеклист).

Отвечай только на русском языке, включая поле comment.

Верни ТОЛЬКО JSON:
{"total": 0-100, "checklist": {"greeting": bool, "need_detection": bool,
"solution_provided": bool, "farewell": bool}, "comment": "краткий комментарий"}
"""


def parse_total(raw: Any) -> int:
    """Итоговая оценка: целое число 0-100, и ничего кроме."""
    if raw is None or isinstance(raw, bool):
        raise ValueError(f"Агент качества не вернул оценку total (получено: {raw!r})")
    if isinstance(raw, str):
        raw = raw.strip().replace(",", ".").rstrip("%")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Агент качества вернул total не числом: {raw!r}") from exc
    if not 0 <= value <= 100:
        raise ValueError(f"Оценка total вне диапазона 0-100: {value}")
    return int(round(value))


class QualityAgent(BaseAgent):
    name = "quality"

    @log_agent_io("quality")
    async def run(self, transcript: list[dict[str, Any]]) -> dict[str, Any]:
        text = transcript_to_text(transcript)
        result = await self.llm.complete_json(SYSTEM_PROMPT, text)
        checklist = result.get("checklist", {})
        return {
            "total": parse_total(result.get("total")),
            "checklist": {
                "greeting": bool(checklist.get("greeting", False)),
                "need_detection": bool(checklist.get("need_detection", False)),
                "solution_provided": bool(checklist.get("solution_provided", False)),
                "farewell": bool(checklist.get("farewell", False)),
            },
            "comment": result.get("comment", ""),
        }
