"""OpenWebUI Pipeline — точка входа, которую подхватывает OpenWebUI Pipelines сервер (кладётся в /pipelines в контейнере openwebui-pipelines)."""
from __future__ import annotations

import asyncio
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Generator, Optional

import httpx
from pydantic import BaseModel

from agents.llm_client import LLMClient
from agents.orchestrator import AgentOrchestrator
from agents.trends import TrendsAgent
from asr.diarizer import diarize
from asr.transcriber import SUPPORTED_EXTENSIONS, Transcriber, probe_duration
import metrics
import storage
from config import settings
from logging_utils import get_logger, timed

logger = get_logger("pipeline")

class Pipeline:
    class Valves(BaseModel):
        """Настройки, доступные для правки из админки OpenWebUI."""

        LLM_BASE_URL: str = settings.llm_base_url
        LLM_MODEL: str = settings.llm_model
        WHISPER_MODEL: str = settings.whisper_model
        WHISPER_DEVICE: str = settings.whisper_device

    def __init__(self):
        self.name = "MTBank Speech Analytics Pipeline"
        self.valves = self.Valves()
        self.transcriber: Optional[Transcriber] = None
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.trends_agent: Optional[TrendsAgent] = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_lock = threading.Lock()

    def _build_components(self) -> None:
        """Пересобрать всё, что зависит от валвов."""
        self.transcriber = Transcriber(
            model_size=self.valves.WHISPER_MODEL, device=self.valves.WHISPER_DEVICE
        )
        llm_client = LLMClient(
            base_url=self.valves.LLM_BASE_URL,
            api_key=settings.llm_api_key,
            model=self.valves.LLM_MODEL,
        )
        self.orchestrator = AgentOrchestrator(llm_client)
        self.trends_agent = TrendsAgent(llm_client)

    async def on_startup(self):
        logger.info("pipeline.startup")
        storage.init_db()
        if settings.metrics_exporter_enabled:
            metrics.start_exporter(settings.metrics_port)
        self._build_components()

    async def on_valves_updated(self):
        """Хук сервера Pipelines: вызывается после POST /{id}/valves/update."""
        logger.info(
            "pipeline.valves_updated",
            extra={"extra_data": {
                "whisper_model": self.valves.WHISPER_MODEL,
                "whisper_device": self.valves.WHISPER_DEVICE,
                "llm_model": self.valves.LLM_MODEL,
            }},
        )
        self._build_components()

    async def on_shutdown(self):
        logger.info("pipeline.shutdown")

    AUDIO_URL_RE = re.compile(r"https?://\S+\.(?:wav|mp3|ogg|flac|m4a)(?:\?\S*)?", re.IGNORECASE)
    FILE_ID_RE = re.compile(r"/api/v1/files/([\w-]+)/content")
    ATTACHED_FILES_BLOCK_RE = re.compile(
        r"<attached_files>(.*?)</attached_files>", re.DOTALL | re.IGNORECASE
    )
    ATTACHED_FILE_TAG_RE = re.compile(r"<file\s+([^>]*?)/?>", re.IGNORECASE)
    XML_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')

    @classmethod
    def _inline_attachments(cls, content: str) -> list[dict[str, Any]]:
        """Вложения, инлайненные в текст сообщения блоком <attached_files>."""
        entries: list[dict[str, Any]] = []
        for block in cls.ATTACHED_FILES_BLOCK_RE.findall(content or ""):
            for tag in cls.ATTACHED_FILE_TAG_RE.findall(block):
                attrs = dict(cls.XML_ATTR_RE.findall(tag))
                file_id = attrs.get("id")
                if not file_id:
                    continue
                name = attrs.get("name", "")
                content_type = attrs.get("content_type", "")
                is_audio = content_type.lower().startswith("audio/") or (
                    Path(name).suffix.lower() in SUPPORTED_EXTENSIONS
                )
                if not is_audio:
                    continue
                entries.append({"id": file_id, "name": name})
        return entries

    @classmethod
    def _file_entries(cls, body: dict[str, Any]) -> list[dict[str, Any]]:
        """Записи о вложениях из тела запроса OpenWebUI."""
        entries = list(body.get("files", []) or [])
        messages = body.get("messages", [])
        if messages:
            last = messages[-1]
            entries += list(last.get("files", []) or [])
            content = last.get("content")
            if isinstance(content, str):
                entries += cls._inline_attachments(content)
        return entries

    @classmethod
    def _attachment_filename(cls, body: dict[str, Any]) -> str | None:
        """Оригинальное имя вложения из тела запроса OpenWebUI."""
        for f in cls._file_entries(body):
            file_obj = f.get("file") or {}
            name = file_obj.get("filename") or f.get("name") or file_obj.get("name")
            if name:
                return str(name)
        return None

    def _extract_audio_ref(self, body: dict[str, Any], user_message: str = "") -> str | None:
        """Ищем аудио тремя способами, в порядке надёжности."""
        for f in self._file_entries(body):
            file_obj = f.get("file") or {}
            url = f.get("url") or file_obj.get("url")
            if not url:
                file_id = file_obj.get("id") or f.get("id")
                if file_id:
                    url = f"/api/v1/files/{file_id}/content"
            if url:
                logger.info("audio.source", extra={"extra_data": {"source": "attachment", "ref": url}})
                return url

        candidates = [user_message]
        messages = body.get("messages", [])
        if messages:
            last = messages[-1].get("content", "")
            if isinstance(last, str):
                candidates.append(last)

        for text in candidates:
            if not text:
                continue
            match = self.AUDIO_URL_RE.search(text)
            if match:
                logger.info("audio.source", extra={"extra_data": {"source": "url_in_text"}})
                return match.group(0)
            stripped = text.strip()
            if stripped.startswith(("http://", "https://")) and " " not in stripped:
                logger.info("audio.source", extra={"extra_data": {"source": "url_in_text"}})
                return stripped
        return None

    def _local_upload_path(self, ref: str, filename_hint: str | None = None) -> Path | None:
        """Вложение чата, лежащее на общем томе OpenWebUI, — путь к нему на диске или None, если это не вложение / тома нет."""
        match = self.FILE_ID_RE.search(ref)
        if not match:
            return None
        uploads_dir = Path(settings.openwebui_uploads_dir)
        if not uploads_dir.is_dir():
            return None

        file_id = match.group(1)
        if filename_hint:
            exact = uploads_dir / f"{file_id}_{filename_hint}"
            if exact.is_file():
                return exact
        candidates = [
            p for p in uploads_dir.glob(f"{file_id}_*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_size)

    async def _download_if_url(self, ref: str, filename_hint: str | None = None) -> Path:
        """Кладём аудио во временный файл."""
        local = self._local_upload_path(ref, filename_hint)
        if local is not None:
            logger.info("audio.source_resolved", extra={"extra_data": {
                "source": "openwebui_volume", "path": str(local)}})
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=local.suffix or ".wav")
            tmp.write(local.read_bytes())
            tmp.close()
            return Path(tmp.name)

        url = ref
        headers: dict[str, str] = {}
        if ref.startswith("/"):
            url = settings.openwebui_base_url.rstrip("/") + ref
            if settings.openwebui_api_key:
                headers["Authorization"] = f"Bearer {settings.openwebui_api_key}"

        if not url.startswith(("http://", "https://")):
            return Path(ref)

        with timed(logger, "download_audio", url=url):
            async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
            suffix = Path(url.split("?")[0]).suffix.lower()
            if suffix not in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}:
                suffix = ".wav"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(resp.content)
            tmp.close()
            return Path(tmp.name)

    AGENT_SECTIONS = ("classification", "quality_score", "compliance", "summary")

    @classmethod
    def _agent_errors(cls, agent_results: dict[str, Any]) -> dict[str, str]:
        """Агенты, которые не отработали."""
        errors: dict[str, str] = {}
        for section in cls.AGENT_SECTIONS:
            value = agent_results.get(section)
            if isinstance(value, dict) and value.get("error"):
                errors[section] = str(value["error"])
        summary_error = agent_results.get("summary_error")
        if summary_error:
            errors["summary"] = str(summary_error)
        return errors

    async def analyze(self, audio_path: Path, *, include_details: bool = False) -> dict[str, Any]:
        """Полный анализ файла: ASR -> диаризация -> 4 агента."""
        assert self.transcriber and self.orchestrator, "Pipeline не инициализирован (on_startup)"

        self.transcriber.validate_format(audio_path)

        duration = await asyncio.to_thread(probe_duration, audio_path)
        if duration > settings.max_audio_duration_sec:
            raise ValueError(
                f"Превышена максимальная длительность записи: {duration:.0f} сек "
                f"при лимите {settings.max_audio_duration_sec} сек."
            )
        logger.info("audio.accepted", extra={"extra_data": {"duration_sec": round(duration, 1)}})

        diarized = await asyncio.to_thread(
            diarize, audio_path, None, self.transcriber
        )
        transcript = [
            {"speaker": s.speaker, "start": s.start, "end": s.end, "text": s.text}
            for s in diarized
        ]
        if not any(seg["text"].strip() for seg in transcript):
            raise ValueError(
                "В записи не распознана речь: транскрипт пуст. Проверьте, что "
                "файл содержит разговор, а не тишину или шум."
            )

        agent_results = await self.orchestrator.run_all(transcript)
        errors = self._agent_errors(agent_results)

        result: dict[str, Any] = {
            "transcript": transcript,
            "classification": {
                "topic": agent_results.get("classification", {}).get("topic"),
                "priority": agent_results.get("classification", {}).get("priority"),
            },
            "quality_score": {
                "total": agent_results.get("quality_score", {}).get("total"),
                "checklist": agent_results.get("quality_score", {}).get("checklist"),
            },
            "compliance": {
                "passed": agent_results.get("compliance", {}).get("passed"),
                "issues": agent_results.get("compliance", {}).get("issues"),
            },
            "summary": agent_results.get("summary"),
            "action_items": agent_results.get("action_items"),
        }

        if include_details:
            details = {
                "classification_reasoning": agent_results.get("classification", {}).get("reasoning"),
                "quality_comment": agent_results.get("quality_score", {}).get("comment"),
            }
            if any(details.values()):
                result["details"] = details

        if errors:
            result["errors"] = errors
            logger.error(
                "analyze.degraded",
                extra={"extra_data": {"failed_agents": sorted(errors), "errors": errors}},
            )
            metrics.record_degraded(errors)
            return result

        try:
            await storage.save_analysis_async(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("storage.save_failed", extra={"extra_data": {"error": str(exc)}})

        metrics.record_analysis(result)

        return result

    @staticmethod
    def _merge_consecutive_speakers(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Склеивает подряд идущие сегменты одного спикера в одну реплику."""
        merged: list[dict[str, Any]] = []
        for seg in transcript:
            if merged and merged[-1]["speaker"] == seg["speaker"]:
                merged[-1]["end"] = seg["end"]
                merged[-1]["text"] = f"{merged[-1]['text']} {seg['text']}".strip()
            else:
                merged.append(dict(seg))
        return merged

    AGENT_TITLES = {
        "classification": "классификация",
        "quality_score": "оценка качества",
        "compliance": "проверка комплаенса",
        "summary": "резюме",
    }

    @staticmethod
    def _compliance_line(compliance: dict[str, Any], failed: bool) -> str:
        """Три состояния, а не два."""
        if failed or compliance.get("passed") is None:
            return "⚠️ данные недоступны"
        return "✅ пройдено" if compliance.get("passed") else "❌ есть замечания"

    def _format_response(self, result: dict[str, Any]) -> str:
        """Markdown-отчёт для чата OpenWebUI."""
        q = result["quality_score"]
        c = result["compliance"]
        cls = result["classification"]
        errors = result.get("errors") or {}

        checklist_lines = "\n".join(
            f"  - {'✅' if v else '❌'} {k}" for k, v in (q.get("checklist") or {}).items()
        )
        if not checklist_lines:
            checklist_lines = "  - данные недоступны"

        if "compliance" in errors:
            issues_block = "  - данные недоступны"
        else:
            issues = c.get("issues") or []
            issues_block = "\n".join(f"  - ⚠️ {i}" for i in issues) if issues else "  - нет замечаний"

        action_items = result.get("action_items") or []
        action_block = "\n".join(f"- {a}" for a in action_items) if action_items else "—"
        transcript_lines = self._merge_consecutive_speakers(result["transcript"])

        quality_total = q.get("total")
        quality_line = "данные недоступны" if quality_total is None else f"{quality_total}/100"

        details = result.get("details") or {}
        quality_comment = details.get("quality_comment")
        quality_note = f"\n\n_{quality_comment}_" if quality_comment else ""
        reasoning = details.get("classification_reasoning")
        reasoning_note = f" \n_{reasoning}_" if reasoning else ""
        summary_text = result.get("summary") or (
            "_данные недоступны_" if "summary" in errors else "—"
        )
        topic = cls.get("topic") or "—"
        priority = cls.get("priority") or "—"

        if errors:
            failed = ", ".join(self.AGENT_TITLES.get(k, k) for k in sorted(errors))
            degraded_banner = (
                f"> ⚠️ **Анализ неполный.** Не отработали: {failed}. "
                "Пустые разделы ниже — это отказ сервиса, а не результат "
                "проверки. Звонок не записан в историю и не учтён в метриках.\n\n"
            )
        else:
            degraded_banner = ""

        return f"""## 📞 Анализ звонка

{degraded_banner}**Тема:** {topic} | **Приоритет:** {priority}{reasoning_note}

### ⭐ Качество обслуживания: {quality_line}
{checklist_lines}{quality_note}

### 🛡️ Compliance: {self._compliance_line(c, "compliance" in errors)}
{issues_block}

### 📝 Резюме
{summary_text}

### ✅ Action items
{action_block}

### 🗒️ Транскрипт
```
{chr(10).join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['speaker']}: {s['text']}" for s in transcript_lines)}
```
"""

    def _run_sync(self, coro):
        """Выполнить корутину из синхронного контекста на собственном loop."""
        with self._loop_lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop.run_until_complete(coro)

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: list[dict],
        body: dict,
    ) -> str | Generator[str, None, None]:
        """Точка входа сервера OpenWebUI Pipelines."""
        text = (user_message or "").strip().lower()
        if text.startswith(("/trends", "тренды")):
            return self._trends_report()

        audio_ref = self._extract_audio_ref(body, user_message)
        if not audio_ref:
            return (
                "Пришлите аудиофайл (WAV/MP3/OGG) или ссылку на него — "
                "я сделаю транскрипт и анализ звонка."
            )
        return self._analysis_stream(audio_ref, self._attachment_filename(body))

    def _trends_report(self) -> str:
        """Текстовая команда `тренды` / `/trends` в чате — бонусное задание."""
        calls = self._run_sync(storage.recent_analyses_async(limit=20))
        result = self._run_sync(self.trends_agent.run(calls))
        if result.get("note"):
            return f"📈 **Тренды**\n\n{result['note']}"

        def bullets(items: list[str]) -> str:
            return "\n".join(f"- {i}" for i in items) if items else "- нет данных"

        return f"""## 📈 Тренды по последним {result['period_calls']} звонкам

**Топ тематик:** {', '.join(result['top_topics']) or '—'}

### Закономерности
{bullets(result['patterns'])}

### Качество обслуживания
{bullets(result['quality_observations'])}

### Рекомендации
{bullets(result['recommendations'])}
"""

    def _analysis_stream(
        self, audio_ref: str, filename_hint: str | None = None
    ) -> Generator[str, None, None]:
        """Генератор прогресса: OpenWebUI по умолчанию шлёт stream=true, и пользователь видит стадии вместо пустого экрана на 30-60 секунд."""
        yield "⏳ Загружаю аудио…\n\n"
        try:
            audio_path = self._run_sync(self._download_if_url(audio_ref, filename_hint))
        except Exception as exc:  # noqa: BLE001
            logger.error("pipe.download_failed", extra={"extra_data": {"error": str(exc)}})
            yield f"❌ Не удалось загрузить аудио: {exc}"
            return

        yield "🎧 Транскрибирую и анализирую (это может занять до минуты)…\n\n"
        try:
            result = self._run_sync(self.analyze(audio_path, include_details=True))
        except ValueError as exc:
            yield f"❌ {exc}"
            return
        except Exception as exc:  # noqa: BLE001
            logger.error("pipe.analyze_failed", extra={"extra_data": {"error": str(exc)}})
            yield "❌ Внутренняя ошибка анализа. Подробности — в логах сервиса."
            return
        finally:
            if audio_ref.startswith(("http://", "https://", "/")):
                audio_path.unlink(missing_ok=True)

        yield self._format_response(result)
