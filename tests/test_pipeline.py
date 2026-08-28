"""Интеграционные тесты."""
from __future__ import annotations

import threading

import pytest

import metrics
import storage
from agents.orchestrator import AgentOrchestrator
from asr.transcriber import Segment, Transcriber
from config import settings
from pipeline import Pipeline
from tests.test_agents import SAMPLE_TRANSCRIPT, FakeLLMClient

class FakeOrchestrator(AgentOrchestrator):
    """Оркестратор с 4 фейковыми агентами вместо реальных LLM-вызовов."""

    def __init__(self):
        from agents.classifier import ClassifierAgent
        from agents.compliance import ComplianceAgent
        from agents.quality import QualityAgent
        from agents.summarizer import SummarizerAgent

        self.agents = {
            "classification": ClassifierAgent(FakeLLMClient({"topic": "кредиты", "priority": "medium"})),
            "quality_score": QualityAgent(FakeLLMClient({
                "total": 78,
                "checklist": {"greeting": True, "need_detection": True, "solution_provided": True, "farewell": False},
            })),
            "compliance": ComplianceAgent(FakeLLMClient({"issues": [], "passed": True})),
            "summary": SummarizerAgent(FakeLLMClient({
                "summary": "Клиент обратился по вопросу кредита наличными.",
                "action_items": ["Отправить КП на email клиента"],
            })),
        }

@pytest.mark.asyncio
async def test_orchestrator_merges_all_agent_outputs():
    orchestrator = FakeOrchestrator()
    result = await orchestrator.run_all(SAMPLE_TRANSCRIPT)

    assert result["classification"]["topic"] == "кредиты"
    assert result["quality_score"]["total"] == 78
    assert result["compliance"]["passed"] is False
    assert result["compliance"]["issues"]
    assert "кредита" in result["summary"]
    assert result["action_items"] == ["Отправить КП на email клиента"]

@pytest.mark.asyncio
async def test_orchestrator_survives_partial_agent_failure(monkeypatch):
    orchestrator = FakeOrchestrator()

    async def boom(self, transcript):
        raise RuntimeError("LLM недоступна")

    monkeypatch.setattr(orchestrator.agents["compliance"], "run", boom.__get__(orchestrator.agents["compliance"]))

    result = await orchestrator.run_all(SAMPLE_TRANSCRIPT)
    assert "error" in result["compliance"]
    assert result["classification"]["topic"] == "кредиты"

@pytest.mark.asyncio
async def test_pipeline_format_response_contains_key_sections():
    pipeline = Pipeline()
    fake_result = {
        "transcript": SAMPLE_TRANSCRIPT,
        "classification": {"topic": "кредиты", "priority": "medium"},
        "quality_score": {"total": 78, "checklist": {"greeting": True, "need_detection": True,
                                                       "solution_provided": True, "farewell": False}},
        "compliance": {"passed": True, "issues": []},
        "summary": "Клиент обратился по вопросу кредита наличными.",
        "action_items": ["Отправить КП на email клиента"],
    }
    markdown = pipeline._format_response(fake_result)
    assert "Качество обслуживания: 78/100" in markdown
    assert "Compliance: ✅ пройдено" in markdown
    assert "Отправить КП на email клиента" in markdown
    assert "Оператор: Добрый день" in markdown

def test_merge_consecutive_speakers_joins_same_speaker_segments():
    """faster-whisper режет длинную реплику на несколько сегментов — подряд идущие сегменты одного спикера должны склеиваться в одну реплику с одной меткой, а не выводиться отдельными строками."""
    transcript = [
        {"speaker": "Оператор", "start": 0.0, "end": 4.8, "text": "МТ Банк, оператор Ирина, слушаю вас."},
        {"speaker": "Клиент", "start": 4.8, "end": 8.2, "text": "Здравствуйте, у меня третий день не проходит платеж по карте,"},
        {"speaker": "Клиент", "start": 8.2, "end": 13.3, "text": "я уже второй раз звоню, и никто ничего не делает."},
        {"speaker": "Оператор", "start": 13.3, "end": 17.0, "text": "Понимаю ваше возмущение, давайте разберемся."},
        {"speaker": "Оператор", "start": 17.0, "end": 22.5, "text": "Назовите, пожалуйста, последние четыре цифры карты."},
    ]
    merged = Pipeline._merge_consecutive_speakers(transcript)

    assert [m["speaker"] for m in merged] == ["Оператор", "Клиент", "Оператор"]
    assert merged[0] == {"speaker": "Оператор", "start": 0.0, "end": 4.8, "text": "МТ Банк, оператор Ирина, слушаю вас."}
    assert merged[1]["start"] == 4.8
    assert merged[1]["end"] == 13.3
    assert merged[1]["text"] == (
        "Здравствуйте, у меня третий день не проходит платеж по карте, "
        "я уже второй раз звоню, и никто ничего не делает."
    )
    assert merged[2]["start"] == 13.3
    assert merged[2]["end"] == 22.5
    assert merged[2]["text"] == (
        "Понимаю ваше возмущение, давайте разберемся. "
        "Назовите, пожалуйста, последние четыре цифры карты."
    )

def test_merge_consecutive_speakers_handles_empty_and_single():
    assert Pipeline._merge_consecutive_speakers([]) == []
    one = [{"speaker": "Клиент", "start": 0.0, "end": 2.0, "text": "Алло."}]
    assert Pipeline._merge_consecutive_speakers(one) == one

def test_pipe_accepts_openwebui_pipelines_signature():
    """Сервер open-webui/pipelines вызывает pipe() именно так: pipe(user_message=..., model_id=..., messages=..., body=...) — синхронно."""
    import inspect

    pipeline = Pipeline()
    sig = inspect.signature(pipeline.pipe)
    assert list(sig.parameters) == ["user_message", "model_id", "messages", "body"]
    assert not inspect.iscoroutinefunction(pipeline.pipe), (
        "pipe() должен быть синхронным: сервер вызывает его через run_in_threadpool"
    )

def test_pipe_without_audio_returns_hint_string():
    pipeline = Pipeline()
    result = pipeline.pipe(
        user_message="привет",
        model_id="mtbank_pipeline",
        messages=[{"role": "user", "content": "привет"}],
        body={"messages": [{"role": "user", "content": "привет"}], "stream": False},
    )
    assert isinstance(result, str)
    assert "аудио" in result.lower()

def test_extract_audio_ref_from_attached_file_with_url():
    pipeline = Pipeline()
    body = {"files": [{"file": {"url": "/api/v1/files/abc123/content", "id": "abc123"}}]}
    assert pipeline._extract_audio_ref(body) == "/api/v1/files/abc123/content"

def test_extract_audio_ref_builds_url_from_file_id_only():
    """OpenWebUI не всегда кладёт url — иногда только id вложения."""
    pipeline = Pipeline()
    body = {"files": [{"file": {"id": "abc123", "filename": "call.wav"}}]}
    assert pipeline._extract_audio_ref(body) == "/api/v1/files/abc123/content"

def test_extract_audio_ref_from_last_message_files():
    """Реальная форма живого запроса (не то, что документация/старый код предполагали): OpenWebUI кладёт вложение в files последнего сообщения, а не в body['files'] верхнего уровня."""
    pipeline = Pipeline()
    body = {
        "messages": [
            {"role": "user", "content": "test",
             "files": [{"type": "file", "id": "abc123", "name": "call.wav"}]}
        ]
    }
    assert pipeline._extract_audio_ref(body) == "/api/v1/files/abc123/content"

def test_attachment_filename_from_last_message_files():
    pipeline = Pipeline()
    body = {
        "messages": [
            {"role": "user", "content": "test",
             "files": [{"type": "file", "id": "abc123", "name": "call.wav"}]}
        ]
    }
    assert pipeline._attachment_filename(body) == "call.wav"

ATTACHED_FILES_CONTENT = (
    '<attached_files>\n'
    '<file type="file" id="59cbd670-9e2b-4593-9e4f-39c4ee323af3" '
    'url="59cbd670-9e2b-4593-9e4f-39c4ee323af3" content_type="audio/mpeg" '
    'name="short_credit_question.mp3"/>\n'
    '</attached_files>\n\nПроанализируй этот звонок'
)

def test_extract_audio_ref_from_inline_attached_files_block():
    """Форма OpenWebUI 0.11+/main, снятая с живого запроса из браузера: вложения вообще не приходят в `files` — ни на верхнем уровне, ни в сообщении."""
    pipeline = Pipeline()
    body = {"messages": [{"role": "user", "content": ATTACHED_FILES_CONTENT}]}
    assert pipeline._extract_audio_ref(body) == (
        "/api/v1/files/59cbd670-9e2b-4593-9e4f-39c4ee323af3/content"
    )

def test_attachment_filename_from_inline_attached_files_block():
    pipeline = Pipeline()
    body = {"messages": [{"role": "user", "content": ATTACHED_FILES_CONTENT}]}
    assert pipeline._attachment_filename(body) == "short_credit_question.mp3"

def test_inline_attachment_url_attribute_is_not_used_as_ref():
    """В этой форме атрибут url содержит голый id, а не путь."""
    pipeline = Pipeline()
    body = {"messages": [{"role": "user", "content": ATTACHED_FILES_CONTENT}]}
    ref = pipeline._extract_audio_ref(body)
    assert ref.startswith("/api/v1/files/")

def test_inline_non_audio_attachment_is_ignored():
    """Приложенный PDF — не повод пытаться его транскрибировать."""
    pipeline = Pipeline()
    content = (
        '<attached_files>\n'
        '<file type="file" id="deadbeef" url="deadbeef" '
        'content_type="application/pdf" name="справка.pdf"/>\n'
        '</attached_files>\n\nчто это'
    )
    body = {"messages": [{"role": "user", "content": content}]}
    assert pipeline._extract_audio_ref(body) is None

def test_extract_audio_ref_finds_url_inside_message_text():
    pipeline = Pipeline()
    body = {"messages": [{"role": "user", "content": "проанализируй https://example.com/call.mp3 пожалуйста"}]}
    assert pipeline._extract_audio_ref(body) == "https://example.com/call.mp3"

def test_extract_audio_ref_prefers_user_message_argument():
    pipeline = Pipeline()
    result = pipeline._extract_audio_ref({}, "https://example.com/a.wav")
    assert result == "https://example.com/a.wav"

def test_extract_audio_ref_returns_none_without_audio():
    pipeline = Pipeline()
    assert pipeline._extract_audio_ref({"messages": [{"role": "user", "content": "привет"}]}) is None

def test_attachment_filename_read_from_body():
    pipeline = Pipeline()
    body = {"files": [{"file": {"id": "abc123", "filename": "call.wav"}}]}
    assert pipeline._attachment_filename(body) == "call.wav"

def test_attachment_filename_none_without_files():
    pipeline = Pipeline()
    assert pipeline._attachment_filename({"messages": []}) is None

@pytest.mark.asyncio
async def test_download_reads_chat_attachment_from_volume(monkeypatch, tmp_path):
    """Главный сценарий ТЗ: вложение чата берётся с общего тома OpenWebUI."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "abc123_call.wav").write_bytes(b"RIFFfake-audio")
    monkeypatch.setattr(settings, "openwebui_uploads_dir", str(uploads))

    pipeline = Pipeline()
    result = await pipeline._download_if_url("/api/v1/files/abc123/content", "call.wav")

    assert result.read_bytes() == b"RIFFfake-audio"
    assert result.suffix == ".wav"
    assert result != uploads / "abc123_call.wav"
    assert (uploads / "abc123_call.wav").exists()
    result.unlink(missing_ok=True)

@pytest.mark.asyncio
async def test_download_prefers_original_over_derived_copy(monkeypatch, tmp_path):
    """OpenWebUI кладёт рядом производные копии ({id}_call.mp3 сведён в моно)."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "abc123_call.wav").write_bytes(b"x" * 500)
    (uploads / "abc123_call.mp3").write_bytes(b"x" * 50)
    (uploads / "abc123_call.json").write_bytes(b"x" * 900)  # не аудио вовсе
    monkeypatch.setattr(settings, "openwebui_uploads_dir", str(uploads))

    pipeline = Pipeline()
    result = await pipeline._download_if_url("/api/v1/files/abc123/content")

    assert result.suffix == ".wav"
    assert len(result.read_bytes()) == 500
    result.unlink(missing_ok=True)

def test_local_upload_path_none_when_volume_absent(monkeypatch, tmp_path):
    """Тома нет (например, деплой без общего диска) — молча уходим на HTTP."""
    monkeypatch.setattr(settings, "openwebui_uploads_dir", str(tmp_path / "missing"))
    pipeline = Pipeline()
    assert pipeline._local_upload_path("/api/v1/files/abc123/content") is None

def test_local_upload_path_none_for_plain_url(monkeypatch, tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(settings, "openwebui_uploads_dir", str(uploads))
    pipeline = Pipeline()
    assert pipeline._local_upload_path("https://example.com/call.wav") is None

@pytest.mark.asyncio
async def test_analyze_rejects_too_long_audio(monkeypatch, tmp_path):
    """MAX_AUDIO_DURATION_SEC был объявлен в конфиге, но не использовался."""
    import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module.settings, "max_audio_duration_sec", 60)
    monkeypatch.setattr(pipeline_module, "probe_duration", lambda path: 3600.0)

    audio = tmp_path / "long.wav"
    audio.write_bytes(b"RIFF0000WAVEfmt ")

    p = Pipeline()
    p.transcriber = FakeTranscriber()
    p.orchestrator = object()

    with pytest.raises(ValueError, match="длительность"):
        await p.analyze(audio)

@pytest.mark.asyncio
async def test_analyze_rejects_unsupported_format_before_probing(monkeypatch, tmp_path):
    """Формат проверяется до ffprobe: раньше validate_format вызывался внутри transcribe(), то есть уже после probe_duration, и на .txt пользователь получал «не удалось прочитать длительность» вместо…"""
    import pipeline as pipeline_module

    def fail_if_called(path):
        raise AssertionError("probe_duration не должен вызываться до проверки формата")

    monkeypatch.setattr(pipeline_module, "probe_duration", fail_if_called)

    doc = tmp_path / "spravka.txt"
    doc.write_text("не аудио", encoding="utf-8")

    p = Pipeline()
    p.transcriber = FakeTranscriber()
    p.orchestrator = object()

    with pytest.raises(ValueError, match="Формат .txt не поддержан"):
        await p.analyze(doc)

@pytest.mark.asyncio
async def test_analyze_rejects_silence_instead_of_hallucinating(monkeypatch, tmp_path):
    """Пустой транскрипт (тишина, шум) отправлял всех четырёх агентов в LLM с пустым промптом: получался выдуманный разбор несуществующего разговора, записанный в историю как настоящий анализ."""
    p, audio, saved, _ = _prepare_analyze(monkeypatch, tmp_path)
    p.transcriber = FakeTranscriber(segments=[])

    with pytest.raises(ValueError, match="не распознана речь"):
        await p.analyze(audio)
    assert saved == []

class FakeTranscriber:
    """Заменяет faster-whisper. Считает вызовы и запоминает, из какого потока её позвали — то и другое проверяется тестами ниже."""

    def __init__(self, segments: list[Segment] | None = None):
        self.calls: list[str] = []
        self.threads: list[str] = []
        self._segments = segments if segments is not None else [
            Segment(start=0.0, end=3.0, text="Добрый день, МТ Банк, меня зовут Анна."),
            Segment(start=3.5, end=6.0, text="Здравствуйте, не работает вход в приложение."),
        ]

    def validate_format(self, audio_path):
        Transcriber.validate_format(self, audio_path)

    def transcribe(self, audio_path):
        self.calls.append(str(audio_path))
        self.threads.append(threading.current_thread().name)
        return list(self._segments)

def _prepare_analyze(monkeypatch, tmp_path, *, stereo: bool = False):
    """Общая обвязка: настоящий analyze() без ffprobe, без модели и без сети."""
    import asr.diarizer as diarizer_module
    import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "probe_duration", lambda path: 12.0)
    monkeypatch.setattr(diarizer_module, "is_stereo", lambda path: stereo)
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [3.2])
    monkeypatch.setattr(diarizer_module.settings, "diarization_backend", "pause_heuristic")

    saved: list[dict] = []

    async def fake_save(result):
        saved.append(result)
        return len(saved)

    monkeypatch.setattr(storage, "save_analysis_async", fake_save)
    monkeypatch.setattr(metrics, "record_analysis", lambda result: None)
    monkeypatch.setattr(metrics, "record_degraded", lambda errors: None)

    audio = tmp_path / "call.wav"
    audio.write_bytes(b"RIFF0000WAVEfmt ")

    transcriber = FakeTranscriber()
    p = Pipeline()
    p.transcriber = transcriber
    p.orchestrator = FakeOrchestrator()
    return p, audio, saved, transcriber

@pytest.mark.asyncio
async def test_analyze_end_to_end_returns_contract_shape(monkeypatch, tmp_path):
    p, audio, saved, _ = _prepare_analyze(monkeypatch, tmp_path)

    result = await p.analyze(audio)

    assert set(result) == {
        "transcript", "classification", "quality_score", "compliance",
        "summary", "action_items",
    }, "на успешном пути схема ответа должна совпадать с ТЗ буква в букву"
    assert result["classification"] == {"topic": "кредиты", "priority": "medium"}
    assert result["quality_score"]["total"] == 78
    assert result["compliance"]["passed"] is True
    assert result["transcript"][0]["speaker"] in {"Оператор", "Клиент"}
    assert saved == [result], "успешный анализ должен попасть в историю"

@pytest.mark.asyncio
async def test_analyze_runs_asr_off_the_event_loop(monkeypatch, tmp_path):
    """ASR и диаризация — синхронный CPU-bound код."""
    p, audio, _, transcriber = _prepare_analyze(monkeypatch, tmp_path)

    await p.analyze(audio)

    assert transcriber.threads, "транскрибация не вызывалась вовсе"
    assert threading.main_thread().name not in transcriber.threads

@pytest.mark.asyncio
async def test_analyze_does_not_transcribe_stereo_file_twice(monkeypatch, tmp_path):
    """Стерео — приоритетный формат контакт-центра, то есть основной путь."""
    import pydub

    exported: list[str] = []

    class FakeChannel:
        def export(self, path, format=None):  # noqa: A002 - имя параметра pydub
            from pathlib import Path as _Path

            _Path(path).write_bytes(b"RIFFfake")
            exported.append(str(path))

    class FakeAudio:
        channels = 2

        def split_to_mono(self):
            return [FakeChannel(), FakeChannel()]

    monkeypatch.setattr(pydub.AudioSegment, "from_file", lambda *a, **k: FakeAudio())

    p, audio, _, transcriber = _prepare_analyze(monkeypatch, tmp_path, stereo=True)

    result = await p.analyze(audio)

    assert len(transcriber.calls) == 2, (
        f"стерео-файл должен транскрибироваться по одному разу на канал, "
        f"а вызовов было {len(transcriber.calls)}: {transcriber.calls}"
    )
    assert str(audio) not in transcriber.calls, (
        "файл целиком транскрибировать не нужно — каналы разбираются отдельно"
    )
    assert {s["speaker"] for s in result["transcript"]} == {"Оператор", "Клиент"}

@pytest.mark.asyncio
async def test_stereo_channels_are_not_written_next_to_source(monkeypatch, tmp_path):
    """Каналы писались рядом с исходником, а ./test_data в docker-compose смонтирован только на чтение — анализ падал бы на записи временного файла."""
    import pydub

    class FakeChannel:
        def export(self, path, format=None):  # noqa: A002 - имя параметра pydub
            from pathlib import Path as _Path

            _Path(path).write_bytes(b"RIFFfake")

    class FakeAudio:
        channels = 2

        def split_to_mono(self):
            return [FakeChannel(), FakeChannel()]

    monkeypatch.setattr(pydub.AudioSegment, "from_file", lambda *a, **k: FakeAudio())

    p, audio, _, _ = _prepare_analyze(monkeypatch, tmp_path, stereo=True)
    before = set(tmp_path.iterdir())

    await p.analyze(audio)

    assert set(tmp_path.iterdir()) == before, (
        "рядом с исходником не должно появляться временных файлов каналов"
    )

@pytest.mark.asyncio
async def test_analyze_marks_agent_failure_instead_of_reporting_violation(monkeypatch, tmp_path):
    """Ключевая регрессия для банковского контекста."""
    p, audio, saved, _ = _prepare_analyze(monkeypatch, tmp_path)

    async def boom(transcript):
        raise RuntimeError("LLM недоступна")

    monkeypatch.setattr(p.orchestrator.agents["compliance"], "run", boom)

    result = await p.analyze(audio)

    assert result["errors"] == {"compliance": "LLM недоступна"}
    assert result["compliance"]["passed"] is None
    assert result["classification"]["topic"] == "кредиты"
    assert saved == [], "деградированный анализ не должен попадать в историю"

@pytest.mark.asyncio
async def test_analyze_reports_summarizer_failure(monkeypatch, tmp_path):
    """summary — плоское поле верхнего уровня, и его ошибка терялась на уровне оркестратора, в отличие от трёх остальных агентов."""
    p, audio, _, _ = _prepare_analyze(monkeypatch, tmp_path)

    async def boom(transcript):
        raise RuntimeError("таймаут суммаризатора")

    monkeypatch.setattr(p.orchestrator.agents["summary"], "run", boom)

    result = await p.analyze(audio)

    assert result["errors"] == {"summary": "таймаут суммаризатора"}
    assert result["summary"] is None
    assert result["action_items"] == []

@pytest.mark.asyncio
async def test_analyze_reports_every_failed_agent(monkeypatch, tmp_path):
    p, audio, saved, _ = _prepare_analyze(monkeypatch, tmp_path)

    async def boom(transcript):
        raise RuntimeError("LLM недоступна")

    for agent in p.orchestrator.agents.values():
        monkeypatch.setattr(agent, "run", boom)

    result = await p.analyze(audio)

    assert set(result["errors"]) == set(Pipeline.AGENT_SECTIONS)
    assert result["transcript"]
    assert saved == []

def test_format_response_renders_agent_failure_as_unavailable():
    """`passed=None` в булевом контексте ложен, поэтому отказ LLM печатался как «❌ есть замечания» — супервайзер видел нарушение комплаенса там, где просто отвалился сервис."""
    p = Pipeline()
    degraded = {
        "transcript": SAMPLE_TRANSCRIPT,
        "classification": {"topic": "кредиты", "priority": "medium"},
        "quality_score": {"total": None, "checklist": None},
        "compliance": {"passed": None, "issues": None},
        "summary": None,
        "action_items": [],
        "errors": {"compliance": "LLM недоступна", "quality_score": "LLM недоступна"},
    }
    markdown = p._format_response(degraded)

    assert "есть замечания" not in markdown
    assert "Compliance: ⚠️ данные недоступны" in markdown
    assert "Анализ неполный" in markdown
    assert "проверка комплаенса" in markdown
    assert "Качество обслуживания: данные недоступны" in markdown

def test_format_response_still_reports_real_violation():
    """Обратная сторона: настоящее отрицательное заключение должно остаться отрицательным, а не смешаться с «данные недоступны»."""
    p = Pipeline()
    result = {
        "transcript": SAMPLE_TRANSCRIPT,
        "classification": {"topic": "кредиты", "priority": "high"},
        "quality_score": {"total": 40, "checklist": {"greeting": False}},
        "compliance": {"passed": False, "issues": ["обещано гарантированное одобрение"]},
        "summary": "Клиент недоволен.",
        "action_items": [],
    }
    markdown = p._format_response(result)

    assert "Compliance: ❌ есть замечания" in markdown
    assert "гарантированное одобрение" in markdown
    assert "Анализ неполный" not in markdown

def test_valves_do_not_expose_secrets():
    """Сервер Pipelines отдаёт `GET /{id}/valves` без аутентификации, поэтому всё, что попало в Valves, читается по HTTP кем угодно."""
    fields = set(Pipeline.Valves.model_fields)
    assert "LLM_API_KEY" not in fields
    assert not any("KEY" in name or "TOKEN" in name or "SECRET" in name for name in fields), (
        f"в Valves не должно быть секретов, а там: {sorted(fields)}"
    )

@pytest.mark.asyncio
async def test_on_valves_updated_rebuilds_components(monkeypatch):
    """Сервер зовёт on_valves_updated() после обновления валвов из UI. Без этого хука правка модели в админке молча ничего не делала до перезапуска контейнера."""
    p = Pipeline()
    p.valves.WHISPER_MODEL = "tiny"
    await p.on_valves_updated()
    assert p.transcriber.model_size == "tiny"

    p.valves.WHISPER_MODEL = "base"
    await p.on_valves_updated()
    assert p.transcriber.model_size == "base"

@pytest.mark.asyncio
async def test_llm_key_comes_from_environment_not_valves(monkeypatch):
    """Ключ читается из окружения; в валвах его нет вовсе."""
    import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module.settings, "llm_api_key", "sk-from-env")
    p = Pipeline()
    await p.on_valves_updated()
    assert p.orchestrator.agents["classification"].llm._client.api_key == "sk-from-env"
