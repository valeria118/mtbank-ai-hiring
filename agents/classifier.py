from __future__ import annotations

from typing import Any

from agents.base import BaseAgent, transcript_to_text
from logging_utils import log_agent_io

SYSTEM_PROMPT = """Ты — классификатор обращений в контакт-центр банка МТБанк.
По транскрипту звонка определи:
- topic: одна из ["кредиты", "карты", "переводы", "жалобы", "прочее"]
- priority: одна из ["low", "medium", "high"] (high — если клиент злится,
  угрожает уйти, речь о мошенничестве/блокировке средств)

Отвечай только на русском языке, включая поле reasoning.

Верни ТОЛЬКО JSON вида:
{"topic": "...", "priority": "...", "reasoning": "краткое обоснование в одном предложении"}
"""


class ClassifierAgent(BaseAgent):
    name = "classifier"

    @log_agent_io("classifier")
    async def run(self, transcript: list[dict[str, Any]]) -> dict[str, Any]:
        text = transcript_to_text(transcript)
        result = await self.llm.complete_json(SYSTEM_PROMPT, text)
        return {
            "topic": result.get("topic", "прочее"),
            "priority": result.get("priority", "medium"),
            "reasoning": result.get("reasoning", ""),
        }
