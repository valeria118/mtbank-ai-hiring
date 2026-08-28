"""Тесты обёртки над faster-whisper."""
from __future__ import annotations

from pathlib import Path

import pytest

from asr.transcriber import Segment, Transcriber

class FakeWord:
    def __init__(self, start, end, word):
        self.start = start
        self.end = end
        self.word = word

class FakeRawSegment:
    def __init__(self, start, end, text, words=None):
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = -0.2
        self.no_speech_prob = 0.01
        self.words = words

class FakeInfo:
    duration = 12.0
    language_probability = 0.99

class FakeModel:
    """Записывает, с какими аргументами его позвали — так проверяем, что батчинг включается только на cuda."""

    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        return iter([FakeRawSegment(0.0, 4.2, " Добрый день ")]), FakeInfo()

def test_validate_format_rejects_unsupported_extension():
    t = Transcriber()
    with pytest.raises(ValueError, match="не поддержан"):
        t.validate_format(Path("call.txt"))

def test_validate_format_accepts_wav_mp3_ogg():
    t = Transcriber()
    for name in ("a.wav", "a.mp3", "a.ogg"):
        t.validate_format(Path(name))

def test_transcribe_strips_text_and_rounds_timestamps(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    t = Transcriber(device="cpu")
    fake = FakeModel()
    monkeypatch.setattr(t, "_ensure_model", lambda: fake)

    segments = t.transcribe(audio)

    assert segments == [Segment(start=0.0, end=4.2, text="Добрый день",
                                avg_logprob=-0.2, no_speech_prob=0.01)]

def test_transcribe_on_cpu_uses_vad_without_batching(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    t = Transcriber(device="cpu")
    fake = FakeModel()
    monkeypatch.setattr(t, "_ensure_model", lambda: fake)

    t.transcribe(audio)

    assert fake.calls[0]["vad_filter"] is True
    assert "batch_size" not in fake.calls[0]

def test_transcribe_on_cuda_passes_batch_size(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    t = Transcriber(device="cuda", batch_size=8)
    fake = FakeModel()
    monkeypatch.setattr(t, "_ensure_model", lambda: fake)

    t.transcribe(audio)

    assert fake.calls[0]["batch_size"] == 8

def test_transcribe_requests_word_timestamps(monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    t = Transcriber(device="cpu")
    fake = FakeModel()
    monkeypatch.setattr(t, "_ensure_model", lambda: fake)

    t.transcribe(audio)

    assert fake.calls[0]["word_timestamps"] is True

def test_transcribe_uses_last_word_end_not_stretched_segment_end(monkeypatch, tmp_path):
    """Регрессия: без word_timestamps faster-whisper иногда "дотягивает" segment.end до начала следующего сегмента того же канала, проглатывая тишину, пока говорит другой канал в стерео-диаризации…"""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    t = Transcriber(device="cpu")
    fake = FakeModel()

    words = [FakeWord(0.0, 1.5, "МТ"), FakeWord(1.5, 2.8, "Банк,"), FakeWord(2.8, 4.1, "слушаю")]
    fake.transcribe = lambda path, **kwargs: (
        iter([FakeRawSegment(0.0, 13.51, "МТ Банк, слушаю", words=words)]),
        FakeInfo(),
    )
    monkeypatch.setattr(t, "_ensure_model", lambda: fake)

    segments = t.transcribe(audio)

    assert segments[0].start == 0.0
    assert segments[0].end == 4.1  # конец последнего слова, а не 13.51
