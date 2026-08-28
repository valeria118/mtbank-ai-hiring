from __future__ import annotations

import re
from typing import Any

from agents.base import BaseAgent, transcript_to_text
from logging_utils import get_logger, log_agent_io

logger = get_logger("agents.summarizer")

MIN_SENTENCES = 3
MAX_SENTENCES = 5

SYSTEM_PROMPT = """Ты — ассистент, который суммаризирует звонки в контакт-центр
МТБанк для супервайзеров. По транскрипту составь:
- summary: ровно 3-5 предложений о сути обращения и его результате
- action_items: список конкретных дальнейших действий (может быть пустым),
  например "отправить КП на email клиента", "перезвонить через 2 дня"

Отвечай только на русском языке, включая summary и action_items.

Верни ТОЛЬКО JSON:
{"summary": "...", "action_items": ["...", "..."]}
"""

RETRY_SYSTEM_PROMPT = """Ты редактируешь резюме звонка. Перепиши присланный
текст так, чтобы в нём было РОВНО от 3 до 5 предложений, не добавляя фактов,
которых в нём нет, и не выбрасывая существенное. Если предложений было мало —
раскрой уже сказанное подробнее; если много — объедини близкие.

Отвечай только на русском языке. Верни ТОЛЬКО JSON:
{"summary": "..."}
"""

_INNER_DOT = re.compile(r"(?<=\w)\.(?=\w)")

_ABBREVIATION_DOT = re.compile(
    r"\b(?i:т·д|т·п|т·е|и·о|др|проч|руб|коп|стр|рис|табл|гг|мин|сек|тыс|млн|млрд)"
    r"\.(?=\s+[а-яёa-z0-9])"
)

_SENTENCE_END = re.compile(r"[.!?…]+(?=\s|$)")

def count_sentences(text: str) -> int:
    """Число предложений в тексте — эвристикой по знакам конца предложения."""
    masked = _INNER_DOT.sub("·", text)
    masked = _ABBREVIATION_DOT.sub(lambda m: m.group(0).replace(".", "·"), masked)
    return len([part for part in _SENTENCE_END.split(masked) if part.strip()])

class SummarizerAgent(BaseAgent):
    name = "summarizer"

    @log_agent_io("summarizer")
    async def run(self, transcript: list[dict[str, Any]]) -> dict[str, Any]:
        text = transcript_to_text(transcript)
        result = await self.llm.complete_json(SYSTEM_PROMPT, text)
        summary = str(result.get("summary") or "").strip()
        action_items = result.get("action_items", [])

        if summary and not (MIN_SENTENCES <= count_sentences(summary) <= MAX_SENTENCES):
            summary = await self._retry_length(summary)

        return {"summary": summary, "action_items": action_items}

    async def _retry_length(self, summary: str) -> str:
        before = count_sentences(summary)
        try:
            retry = await self.llm.complete_json(RETRY_SYSTEM_PROMPT, summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "summary.length_retry_failed",
                extra={"extra_data": {"sentences": before, "error": str(exc)}},
            )
            return summary

        candidate = str(retry.get("summary") or "").strip()
        after = count_sentences(candidate) if candidate else 0
        if candidate and MIN_SENTENCES <= after <= MAX_SENTENCES:
            logger.info(
                "summary.length_fixed",
                extra={"extra_data": {"sentences_before": before, "sentences_after": after}},
            )
            return candidate

        logger.warning(
            "summary.length_out_of_range",
            extra={"extra_data": {"sentences_before": before, "sentences_after": after}},
        )
        return summary
