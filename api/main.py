"""FastAPI-сервис: REST-обёртка над тем же Pipeline, что использует OpenWebUI. Запуск: uvicorn api.main:app --host 0.0.0.0 --port 8080"""
from __future__ import annotations

import asyncio
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.datastructures import UploadFile as StarletteUploadFile

import storage
from agents.llm_client import LLMClient
from agents.trends import TrendsAgent
from config import settings
from logging_utils import get_logger, timed
from pipeline import Pipeline
from realtime import StreamingTranscriber

logger = get_logger("api")
pipeline = Pipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipeline.on_startup()
    yield
    await pipeline.on_shutdown()

app = FastAPI(
    title="MTBank Speech Analytics API",
    version="1.0.0",
    lifespan=lifespan,
    root_path=settings.api_root_path,
)

class AnalyzeUrlRequest(BaseModel):
    url: str

class TranscriptSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str

class Classification(BaseModel):
    topic: str | None = None
    priority: str | None = None

class QualityChecklist(BaseModel):
    greeting: bool
    need_detection: bool
    solution_provided: bool
    farewell: bool

class QualityScore(BaseModel):
    total: int | None = None
    checklist: QualityChecklist | None = None

class Compliance(BaseModel):
    passed: bool | None = None
    issues: list[str] | None = None

class AnalyzeResponse(BaseModel):
    transcript: list[TranscriptSegment]
    classification: Classification
    quality_score: QualityScore
    compliance: Compliance
    summary: str | None = None
    action_items: list[str] = []
    errors: dict[str, str] | None = None

    model_config = {
        "json_schema_extra": {
            "description": (
                "Поле errors появляется только при деградации: агент не "
                "отработал. Пустые значения в его секции — отказ сервиса, а "
                "не результат проверки. Если не отработал ни один агент, "
                "ответ приходит со статусом 502."
            )
        }
    }

ANALYZE_REQUEST_BODY = {
    "required": True,
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "format": "binary", "description": "Аудиофайл WAV/MP3/OGG/FLAC/M4A"}
                },
                "required": ["file"],
            }
        },
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {"url": {"type": "string", "format": "uri"}},
                "required": ["url"],
                "example": {"url": "https://example.com/call.wav"},
            }
        },
    },
}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def prometheus_metrics():
    """Экспозиция метрик для Prometheus (источник данных Grafana-дашборда)."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/trends")
async def trends(last: int = 20):
    """Бонус ТЗ: агент трендов по последним `last` проанализированным звонкам."""
    if last < 1 or last > 200:
        raise HTTPException(400, "Параметр last должен быть в диапазоне 1..200")
    calls = await storage.recent_analyses_async(limit=last)
    agent = TrendsAgent(LLMClient())
    with timed(logger, "trends_request", n_calls=len(calls)):
        result = await agent.run(calls)
    return JSONResponse(result)

async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await upload.read())
        return Path(tmp.name)

async def _download_to_temp(url: str) -> Path:
    with timed(logger, "download_url", url=url):
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    suffix = Path(url.split("?")[0]).suffix.lower()
    if suffix not in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}:
        suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(resp.content)
        return Path(tmp.name)

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    openapi_extra={"requestBody": ANALYZE_REQUEST_BODY},
    responses={
        400: {"description": "Файл не распознан как аудио, формат не поддержан, превышена длительность или в записи нет речи"},
        502: {"description": "Не отработал ни один агент — анализ не выполнен (транскрипт в теле ответа остаётся)"},
    },
)
async def analyze(request: Request):
    """ТЗ, компонент 3: принимает либо multipart/form-data с полем `file`, либо application/json с телом {"url": "https://..."}."""
    content_type = request.headers.get("content-type", "")
    tmp_path: Path | None = None

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, StarletteUploadFile):
            raise HTTPException(400, "Передайте аудио в поле file или JSON {\"url\": \"...\"}")
        tmp_path = await _save_upload(upload)
    elif content_type.startswith("application/json"):
        try:
            payload = AnalyzeUrlRequest(**await request.json())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, "Ожидается JSON вида {\"url\": \"https://...\"}") from exc
        try:
            tmp_path = await _download_to_temp(payload.url)
        except httpx.HTTPError as exc:
            raise HTTPException(400, f"Не удалось скачать файл по ссылке: {exc}") from exc
    else:
        raise HTTPException(400, "Передайте аудио в поле file или JSON {\"url\": \"...\"}")

    try:
        with timed(logger, "analyze_request", filename=str(tmp_path)):
            result = await pipeline.analyze(tmp_path)
        errors = result.get("errors") or {}
        if len(errors) >= len(Pipeline.AGENT_SECTIONS):
            logger.error("analyze.all_agents_failed", extra={"extra_data": {"errors": errors}})
            return JSONResponse(result, status_code=502)
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("analyze.failed", extra={"extra_data": {"error": str(exc)}})
        raise HTTPException(500, "Внутренняя ошибка анализа") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """Бонус ТЗ: потоковая транскрибация."""
    await websocket.accept()
    stream = StreamingTranscriber(
        pipeline.transcriber,
        sample_rate=settings.realtime_sample_rate,
        window_sec=settings.realtime_window_sec,
    )
    logger.info("ws.connected")
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))

            text = message.get("text")
            if text is not None:
                # Клиент сообщает, что запись окончена.
                if text.strip().lower() in {"stop", '{"type":"stop"}'}:
                    tail = await asyncio.to_thread(stream.flush)
                    await websocket.send_json({"type": "final", "segments": tail})
                    logger.info("ws.finalized", extra={"extra_data": {"tail_segments": len(tail)}})
                    await websocket.close()
                    return
                continue

            chunk = message.get("bytes")
            if chunk is None:
                continue
            started = time.perf_counter()
            segments = await asyncio.to_thread(stream.feed, chunk)
            if segments:
                await websocket.send_json({
                    "type": "partial",
                    "segments": segments,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                })
    except WebSocketDisconnect:
        tail = await asyncio.to_thread(stream.flush)
        logger.info("ws.disconnected", extra={"extra_data": {"tail_segments": len(tail)}})
    except Exception as exc:  # noqa: BLE001
        logger.error("ws.failed", extra={"extra_data": {"error": str(exc)}})
        await websocket.close(code=1011)

@app.get("/realtime")
async def realtime_page():
    """Демо-страница потокового режима."""
    return FileResponse(Path(__file__).parent.parent / "static" / "realtime.html")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=False)
