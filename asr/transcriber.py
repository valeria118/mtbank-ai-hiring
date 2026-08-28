"""Обёртка над faster-whisper: аудио любого поддерживаемого формата -> список сегментов с таймкодами и текстом (без спикеров — диаризация в diarizer.py)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config import settings
from logging_utils import get_logger, timed

logger = get_logger("asr.transcriber")

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

class Transcriber:
    """Ленивая загрузка модели: она тяжёлая, поднимаем один раз при старте пайплайна (on_startup), не при импорте модуля."""

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model_size = model_size or settings.whisper_model
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self.language = language or settings.whisper_language
        self.batch_size = batch_size or settings.whisper_batch_size
        self._model = None

    def _ensure_model(self):
        """Ленивая загрузка: модель тяжёлая, поднимаем один раз."""
        if self._model is None:
            from faster_whisper import WhisperModel  # импорт здесь: тяжёлая зависимость

            with timed(logger, "model.load", model=self.model_size, device=self.device):
                model = WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )
                if self.device == "cuda":
                    from faster_whisper import BatchedInferencePipeline

                    self._model = BatchedInferencePipeline(model=model)
                else:
                    self._model = model
        return self._model

    def validate_format(self, path: str | Path) -> None:
        ext = Path(path).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Формат {ext} не поддержан. Поддерживаются: {sorted(SUPPORTED_EXTENSIONS)}"
            )

    def transcribe(self, audio_path: str | Path) -> list[Segment]:
        """Синхронная транскрибация файла."""
        self.validate_format(audio_path)
        model = self._ensure_model()

        kwargs: dict = {
            "language": self.language,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500},
            "word_timestamps": True,
        }
        if self.device == "cuda":
            kwargs["batch_size"] = self.batch_size

        with timed(logger, "transcribe", audio_path=str(audio_path)):
            segments_iter, info = model.transcribe(str(audio_path), **kwargs)
            segments = []
            for s in segments_iter:
                words = getattr(s, "words", None)
                start = words[0].start if words else s.start
                end = words[-1].end if words else s.end
                segments.append(
                    Segment(
                        start=round(start, 2),
                        end=round(end, 2),
                        text=s.text.strip(),
                        avg_logprob=s.avg_logprob,
                        no_speech_prob=s.no_speech_prob,
                    )
                )
        logger.info(
            "transcribe.result",
            extra={"extra_data": {
                "n_segments": len(segments),
                "duration": info.duration,
                "language_probability": info.language_probability,
            }},
        )
        return segments

def probe_duration(audio_path: str | Path) -> float:
    """Длительность аудио в секундах через ffprobe (pydub.utils.mediainfo)."""
    from pydub.utils import mediainfo

    info = mediainfo(str(audio_path))
    raw = info.get("duration")
    if raw is None:
        raise ValueError(f"Не удалось определить длительность файла {audio_path}")
    return float(raw)
