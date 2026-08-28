"""Compliance-агент."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agents.base import BaseAgent, transcript_to_text
from asr.diarizer import OPERATOR
from logging_utils import log_agent_io

FORBIDDEN_PHRASES = [
    "гарантированное одобрение",
    "100% одобрение",
    "без проверки кредитной истории",
    "без документов",
]

@dataclass(frozen=True)
class Disclosure:
    """Обязательное раскрытие: как назвать его в issues и как распознать, что оно прозвучало."""

    title: str
    pattern: re.Pattern[str]

REQUIRED_DISCLOSURES: list[tuple[re.Pattern[str], tuple[Disclosure, ...]]] = [
    (
        re.compile(r"кредит|рассрочк|заё?м|займ|ссуд|рефинансир", re.IGNORECASE),
        (
            Disclosure(
                "решение о выдаче принимает банк",
                re.compile(
                    r"решени\w*[^.!?]{0,40}принима\w+[^.!?]{0,25}банк"
                    r"|банк[^.!?]{0,30}принима\w+[^.!?]{0,25}решени"
                    r"|окончательн\w+\s+решени",
                    re.IGNORECASE,
                ),
            ),
            Disclosure(
                "полная процентная ставка / полная стоимость кредита",
                re.compile(
                    r"полн\w*\s+(годов\w*\s+)?(процентн\w*\s+ставк|стоимост)",
                    re.IGNORECASE,
                ),
            ),
            Disclosure(
                "предложение не является публичной офертой",
                re.compile(r"не\s+являе\w*[^.!?]{0,30}оферт|не\s+(публичн\w+\s+)?оферт", re.IGNORECASE),
            ),
        ),
    ),
    (
        re.compile(r"страхов", re.IGNORECASE),
        (
            Disclosure(
                "добровольность страхования",
                re.compile(
                    r"доброволь\w*"
                    r"|по\s+ваш\w+\s+желани\w*"
                    r"|не\s+обязательн\w*"
                    r"|(можете|вправе|вправе\s+в\s+любой\s+момент)\s+отказат\w+",
                    re.IGNORECASE,
                ),
            ),
        ),
    ),
]

SYSTEM_PROMPT = """Ты — compliance-агент банка МТБанк. Проверь транскрипт звонка на:
1. Использование оператором вводящих в заблуждение формулировок
   (гарантии одобрения, отсутствие проверок, нереалистичные обещания).
2. Обязательные раскрытия. Если речь шла о кредите/рассрочке/займе, оператор
   обязан сообщить: решение о выдаче принимает банк; полную процентную ставку
   (полную стоимость кредита), а не только «ставку от»; что предложение не
   является публичной офертой. Если речь шла о страховании — что оно
   добровольное. Отметь то, чего не прозвучало.
3. Корректность предложения продукта: если оператор называет процентную
   ставку или условия кредита, они не должны звучать как безусловная гарантия.
4. Отсутствие давления на клиента / грубости.

Претензии предъявляй только к репликам ОПЕРАТОРА: слова клиента нарушением
оператора не являются, даже если клиент цитирует запрещённую формулировку.

Отвечай только на русском языке, включая формулировки в issues.

Верни ТОЛЬКО JSON:
{"issues": ["список найденных проблем, каждая — короткая строка"], "passed": bool}
passed = true, если issues пустой список.
"""

def _operator_text(transcript: list[dict[str, Any]]) -> str:
    """Только реплики оператора."""
    return "\n".join(
        seg.get("text", "") for seg in transcript if seg.get("speaker") == OPERATOR
    )

def _check_forbidden_phrases(operator_text: str) -> list[str]:
    lowered = operator_text.lower()
    return [
        f"Запрещённая формулировка в реплике оператора: «{phrase}»"
        for phrase in FORBIDDEN_PHRASES
        if phrase in lowered
    ]

def _check_required_disclosures(full_text: str, operator_text: str) -> list[str]:
    """Обязательные раскрытия, которых оператор не сделал."""
    missing: list[str] = []
    for topic_trigger, disclosures in REQUIRED_DISCLOSURES:
        if not topic_trigger.search(full_text):
            continue
        for disclosure in disclosures:
            if not disclosure.pattern.search(operator_text):
                missing.append(f"Не прозвучало обязательное раскрытие: {disclosure.title}")
    return missing

class ComplianceAgent(BaseAgent):
    name = "compliance"

    @log_agent_io("compliance")
    async def run(self, transcript: list[dict[str, Any]]) -> dict[str, Any]:
        text = transcript_to_text(transcript)
        operator_text = _operator_text(transcript)

        rule_issues = _check_forbidden_phrases(operator_text)
        rule_issues += _check_required_disclosures(text, operator_text)

        llm_result = await self.llm.complete_json(SYSTEM_PROMPT, text)
        llm_issues = llm_result.get("issues", [])

        all_issues = rule_issues + [i for i in llm_issues if i not in rule_issues]
        return {"passed": len(all_issues) == 0, "issues": all_issues}
