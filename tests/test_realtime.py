"""Тесты буферизации потокового режима."""
from __future__ import annotations

import pytest

from asr.transcriber import Segment
from realtime import StreamingTranscriber

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


class FakeTranscriber:
    def __init__(self):
        self.calls = 0

    def transcribe(self, path):
        self.calls += 1
        return [Segment(start=0.0, end=1.0, text=f"фрагмент {self.calls}")]


def _chunk(seconds: float) -> bytes:
    return b"\x00" * int(SAMPLE_RATE * BYTES_PER_SAMPLE * seconds)


def test_feed_returns_none_until_window_is_full():
    st = StreamingTranscriber(FakeTranscriber(), sample_rate=SAMPLE_RATE, window_sec=2.0)
    assert st.feed(_chunk(0.5)) is None
    assert st.feed(_chunk(0.5)) is None


def test_feed_transcribes_when_window_is_full():
    fake = FakeTranscriber()
    st = StreamingTranscriber(fake, sample_rate=SAMPLE_RATE, window_sec=2.0)
    st.feed(_chunk(1.0))
    result = st.feed(_chunk(1.2))
    assert result is not None
    assert result[0]["text"] == "фрагмент 1"
    assert fake.calls == 1


def test_window_resets_after_transcription():
    fake = FakeTranscriber()
    st = StreamingTranscriber(fake, sample_rate=SAMPLE_RATE, window_sec=1.0)
    st.feed(_chunk(1.1))
    st.feed(_chunk(0.5))
    assert fake.calls == 1, "второй, неполный буфер не должен запускать модель"


def test_timestamps_are_offset_by_stream_position():
    """Модель считает время от начала окна — в потоке нужно абсолютное."""
    fake = FakeTranscriber()
    st = StreamingTranscriber(fake, sample_rate=SAMPLE_RATE, window_sec=1.0)
    st.feed(_chunk(1.1))
    second = st.feed(_chunk(1.1))
    assert second[0]["start"] >= 1.0, "второе окно должно начинаться после первого"


def test_flush_transcribes_remaining_tail():
    fake = FakeTranscriber()
    st = StreamingTranscriber(fake, sample_rate=SAMPLE_RATE, window_sec=10.0)
    st.feed(_chunk(1.0))
    result = st.flush()
    assert fake.calls == 1
    assert result[0]["text"] == "фрагмент 1"


def test_flush_on_empty_buffer_returns_empty_list():
    fake = FakeTranscriber()
    st = StreamingTranscriber(fake, sample_rate=SAMPLE_RATE, window_sec=1.0)
    assert st.flush() == []
    assert fake.calls == 0
