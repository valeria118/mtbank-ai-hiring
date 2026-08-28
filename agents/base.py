"""Общий базовый класс для агентов + утилита сериализации транскрипта в текст для промптов."""
from __future__ import annotations

from typing import Any

from agents.llm_client import LLMClient


def transcript_to_text(transcript: list[dict[str, Any]]) -> str:
    """[{"speaker": "Оператор", "start":.., "end":.., "text":..}, ...] -> текст диалога"""
    lines = []
    for seg in transcript:
        ts = f"[{seg['start']:.1f}-{seg['end']:.1f}]"
        lines.append(f"{ts} {seg['speaker']}: {seg['text']}")
    return "\n".join(lines)


class BaseAgent:
    name: str = "base"

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient()

    async def run(self, transcript: list[dict[str, Any]]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError
