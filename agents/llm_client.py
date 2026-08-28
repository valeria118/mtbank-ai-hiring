"""Тонкая обёртка над OpenAI SDK — работает с любым OpenAI-совместимым эндпоинтом (Groq, Together, OpenRouter, локальный vLLM/Ollama)."""
from __future__ import annotations

import json
from typing import Any

from openai import APIStatusError, AsyncOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from config import settings

def _is_retryable(exc: BaseException) -> bool:
    """Повторять стоит только то, что может пройти со второй попытки."""
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return True

class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.model = model or settings.llm_model
        self._client = AsyncOpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key,
            timeout=30.0,
        )

    @retry(
        wait=wait_exponential(min=1, max=8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Запрос к LLM с требованием вернуть строго JSON. Ретраится до 3 раз при сетевых ошибках, 429/5xx и невалидном JSON; 4xx не ретраится."""
        response = await self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM вернула невалидный JSON: {content[:200]}") from exc
