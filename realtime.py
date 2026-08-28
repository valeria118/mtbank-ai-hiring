"""Потоковая транскрибация по WebSocket (бонусное задание ТЗ)."""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Any

from logging_utils import get_logger

logger = get_logger("realtime")

BYTES_PER_SAMPLE = 2  # PCM16


class StreamingTranscriber:
    def __init__(self, transcriber, sample_rate: int, window_sec: float) -> None:
        self.transcriber = transcriber
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.window_bytes = int(sample_rate * BYTES_PER_SAMPLE * window_sec)
        self._buffer = bytearray()
        self._stream_offset_sec = 0.0

    def feed(self, chunk: bytes) -> list[dict[str, Any]] | None:
        """Добавить чанк. Пока окно не заполнено — None; как заполнилось — список сегментов с абсолютными таймкодами."""
        self._buffer.extend(chunk)
        if len(self._buffer) < self.window_bytes:
            return None
        return self._drain()

    def flush(self) -> list[dict[str, Any]]:
        """Дотранскрибировать хвост при закрытии соединения."""
        if not self._buffer:
            return []
        return self._drain()

    def _drain(self) -> list[dict[str, Any]]:
        audio_bytes = bytes(self._buffer)
        window_duration = len(audio_bytes) / (self.sample_rate * BYTES_PER_SAMPLE)
        self._buffer.clear()

        tmp_path = self._write_wav(audio_bytes)
        try:
            segments = self.transcriber.transcribe(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        offset = self._stream_offset_sec
        self._stream_offset_sec += window_duration

        return [
            {
                "start": round(s.start + offset, 2),
                "end": round(s.end + offset, 2),
                "text": s.text,
            }
            for s in segments
            if s.text
        ]

    def _write_wav(self, audio_bytes: bytes) -> Path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        with wave.open(tmp.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(BYTES_PER_SAMPLE)
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_bytes)
        return Path(tmp.name)
