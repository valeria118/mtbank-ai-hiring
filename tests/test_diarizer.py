"""Тесты обеих стратегий диаризации."""
from __future__ import annotations

import pytest

import asr.diarizer as diarizer_module
from asr.diarizer import (
    CLIENT,
    OPERATOR,
    DiarizedSegment,
    diarize,
    diarize_by_pause_heuristic,
)
from asr.transcriber import Segment

class _FakeTurn:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

class _FakeAnnotation:
    """Имитирует pyannote.core.Annotation ровно настолько, насколько нужно для diarize_by_pyannote: перечисление (поворот, имя_дорожки, метка)."""

    def __init__(self, tracks: list[tuple[float, float, str]]):
        self._tracks = tracks

    def itertracks(self, yield_label: bool = False):
        for start, end, label in self._tracks:
            yield _FakeTurn(start, end), "_", label

class _FakeDiarizeOutput:
    """Имитирует обёртку pyannote.audio>=4.0 (DiarizeOutput)."""

    def __init__(self, annotation: _FakeAnnotation):
        self.speaker_diarization = annotation

def test_pause_heuristic_starts_with_operator_when_greeting_detected(monkeypatch):
    """Одна непрерывная реплика (монолог), в тексте — типичное приветствие оператора: банк себя называет, спрашивает "чем могу помочь"."""
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [])
    segments = [Segment(start=0.0, end=3.0, text="МТБанк, слушаю вас, чем могу помочь")]
    result = diarize_by_pause_heuristic(segments, "dummy.wav")
    assert result[0].speaker == OPERATOR

def test_pause_heuristic_client_monologue_not_mislabeled_as_operator(monkeypatch):
    """Регрессия на реальный баг: аудиофайл, где говорит только клиент (без смены реплик — значит и без границ тишины), раньше целиком помечался как Оператор из-за слепого дефолта."""
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [])
    segments = [
        Segment(start=0.0, end=3.0, text="Мне нужно срочно заблокировать карту"),
        Segment(start=3.2, end=6.0, text="я потерял её сегодня в метро"),
    ]
    result = diarize_by_pause_heuristic(segments, "dummy.wav")
    assert [s.speaker for s in result] == [CLIENT, CLIENT]

def test_pause_heuristic_switches_speaker_at_boundary(monkeypatch):
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [4.0])
    segments = [
        Segment(start=0.0, end=3.0, text="Добрый день"),
        Segment(start=5.0, end=7.0, text="Здравствуйте"),  # граница 4.0 между сегментами
    ]
    result = diarize_by_pause_heuristic(segments, "dummy.wav")
    assert [s.speaker for s in result] == [OPERATOR, CLIENT]

def test_pause_heuristic_keeps_speaker_without_boundary(monkeypatch):
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [])
    segments = [
        Segment(start=0.0, end=3.0, text="Добрый день"),
        Segment(start=3.2, end=5.0, text="меня зовут Анна"),
    ]
    result = diarize_by_pause_heuristic(segments, "dummy.wav")
    assert [s.speaker for s in result] == [OPERATOR, OPERATOR]

def test_pause_heuristic_handles_multiple_boundaries(monkeypatch):
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [4.0, 8.0, 12.0])
    segments = [
        Segment(start=0.0, end=3.0, text="A"),
        Segment(start=5.0, end=7.0, text="B"),
        Segment(start=9.0, end=11.0, text="C"),
        Segment(start=13.0, end=15.0, text="D"),
    ]
    result = diarize_by_pause_heuristic(segments, "dummy.wav")
    assert [s.speaker for s in result] == [OPERATOR, CLIENT, OPERATOR, CLIENT]

def test_pause_heuristic_preserves_timestamps_and_text(monkeypatch):
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [])
    segments = [Segment(start=1.5, end=4.25, text="Слушаю вас, меня зовут Анна")]
    result = diarize_by_pause_heuristic(segments, "dummy.wav")
    assert result == [
        DiarizedSegment(speaker=OPERATOR, start=1.5, end=4.25, text="Слушаю вас, меня зовут Анна")
    ]

def test_pause_heuristic_on_empty_input(monkeypatch):
    boundaries_called = False

    def fail_if_called(path):
        nonlocal boundaries_called
        boundaries_called = True
        return []

    monkeypatch.setattr(diarizer_module, "_turn_boundaries", fail_if_called)
    assert diarize_by_pause_heuristic([], "dummy.wav") == []
    assert boundaries_called is False

def test_turn_boundaries_detects_real_silence_gap(tmp_path):
    from pydub import AudioSegment
    from pydub.generators import Sine

    tone = Sine(440).to_audio_segment(duration=1000).apply_gain(-3)
    silence = AudioSegment.silent(duration=2000)
    audio = tone + silence + tone
    path = tmp_path / "gap.wav"
    audio.export(path, format="wav")

    boundaries = diarizer_module._turn_boundaries(path)
    assert len(boundaries) == 1
    assert 1.0 < boundaries[0] < 3.0

def test_turn_boundaries_merges_adjacent_silences(tmp_path):
    """Одна реальная пауза, случайно разбитая коротким тихим звуком на два отрезка тишины, должна давать одну границу, а не две."""
    from pydub import AudioSegment
    from pydub.generators import Sine

    tone = Sine(440).to_audio_segment(duration=1000).apply_gain(-3)
    blip = Sine(440).to_audio_segment(duration=200).apply_gain(-3)
    audio = tone + AudioSegment.silent(duration=1600) + blip + AudioSegment.silent(duration=1600) + tone
    path = tmp_path / "split_gap.wav"
    audio.export(path, format="wav")

    boundaries = diarizer_module._turn_boundaries(path)
    assert len(boundaries) == 1

def test_diarize_falls_back_to_pause_heuristic_without_transcriber(monkeypatch, tmp_path):
    audio = tmp_path / "mono.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(diarizer_module, "is_stereo", lambda path: False)
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [4.0])
    segments = [
        Segment(start=0.0, end=3.0, text="Добрый день"),
        Segment(start=5.0, end=7.0, text="Здравствуйте"),
    ]
    result = diarize(audio, segments, transcriber=None)
    assert [s.speaker for s in result] == [OPERATOR, CLIENT]

def test_diarize_uses_stereo_strategy_when_file_is_stereo(monkeypatch, tmp_path):
    audio = tmp_path / "stereo.wav"
    audio.write_bytes(b"RIFF")

    monkeypatch.setattr(diarizer_module, "is_stereo", lambda path: True)

    def fake_stereo(audio_path, transcriber):
        return [DiarizedSegment(speaker=OPERATOR, start=0.0, end=1.0, text="из стерео")]

    monkeypatch.setattr(diarizer_module, "diarize_by_stereo_channels", fake_stereo)

    result = diarize(audio, [], transcriber=object())
    assert result[0].text == "из стерео"

def test_diarize_by_pyannote_maps_clusters_via_greeting_heuristic(monkeypatch):
    """pyannote отдаёт условные метки SPEAKER_00/01 — сопоставляем их с Оператор/Клиент по содержанию (кто здоровается как банк)."""
    fake_annotation = _FakeAnnotation(
        [(0.0, 4.8, "SPEAKER_01"), (4.8, 13.3, "SPEAKER_00"), (13.3, 22.5, "SPEAKER_01")]
    )
    monkeypatch.setattr(diarizer_module, "_get_pyannote_pipeline", lambda: (lambda path: fake_annotation))
    segments = [
        Segment(start=0.0, end=4.8, text="МТ Банк, слушаю вас, чем могу помочь"),
        Segment(start=4.8, end=13.3, text="Здравствуйте, у меня проблема с картой"),
        Segment(start=13.3, end=22.5, text="Понимаю ваше возмущение, разберёмся"),
    ]
    result = diarizer_module.diarize_by_pyannote(segments, "dummy.wav")
    assert [r.speaker for r in result] == [OPERATOR, CLIENT, OPERATOR]

def test_diarize_by_pyannote_supports_diarize_output_wrapper(monkeypatch):
    """pyannote.audio>=4.0 оборачивает Annotation в DiarizeOutput (.speaker_diarization) — код должен работать с обеими формами."""
    fake_annotation = _FakeAnnotation([(0.0, 3.0, "SPEAKER_00")])
    fake_output = _FakeDiarizeOutput(fake_annotation)
    monkeypatch.setattr(diarizer_module, "_get_pyannote_pipeline", lambda: (lambda path: fake_output))
    segments = [Segment(start=0.0, end=3.0, text="Алло, добрый день")]
    result = diarizer_module.diarize_by_pyannote(segments, "dummy.wav")
    assert len(result) == 1
    assert result[0].text == "Алло, добрый день"

def test_diarize_by_pyannote_falls_back_to_pause_heuristic_without_turns(monkeypatch):
    """pyannote не нашёл речи вовсе (пустой/тихий файл) — не роняем пайплайн, а откатываемся на эвристику по паузам."""
    fake_annotation = _FakeAnnotation([])
    monkeypatch.setattr(diarizer_module, "_get_pyannote_pipeline", lambda: (lambda path: fake_annotation))
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [])
    segments = [Segment(start=0.0, end=3.0, text="МТБанк, слушаю вас")]
    result = diarizer_module.diarize_by_pyannote(segments, "dummy.wav")
    assert result[0].speaker == OPERATOR

def test_diarize_auto_uses_pyannote_when_hf_token_set(monkeypatch, tmp_path):
    audio = tmp_path / "mono.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(diarizer_module, "is_stereo", lambda path: False)
    monkeypatch.setattr(diarizer_module.settings, "diarization_backend", "auto")
    monkeypatch.setattr(diarizer_module.settings, "hf_token", "fake-token")

    called = {}

    def fake_pyannote(segments, audio_path):
        called["yes"] = True
        return [DiarizedSegment(speaker=OPERATOR, start=0.0, end=1.0, text="из pyannote")]

    monkeypatch.setattr(diarizer_module, "diarize_by_pyannote", fake_pyannote)
    result = diarize(audio, [], transcriber=None)
    assert called.get("yes") is True
    assert result[0].text == "из pyannote"

def test_diarize_auto_skips_pyannote_without_hf_token(monkeypatch, tmp_path):
    audio = tmp_path / "mono.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(diarizer_module, "is_stereo", lambda path: False)
    monkeypatch.setattr(diarizer_module.settings, "diarization_backend", "auto")
    monkeypatch.setattr(diarizer_module.settings, "hf_token", "")
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [4.0])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("pyannote не должен вызываться без HF_TOKEN в режиме auto")

    monkeypatch.setattr(diarizer_module, "diarize_by_pyannote", fail_if_called)
    segments = [Segment(start=0.0, end=3.0, text="Добрый день")]
    result = diarize(audio, segments, transcriber=None)
    assert result[0].speaker == OPERATOR

def test_diarize_auto_falls_back_silently_when_pyannote_fails(monkeypatch, tmp_path):
    audio = tmp_path / "mono.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(diarizer_module, "is_stereo", lambda path: False)
    monkeypatch.setattr(diarizer_module.settings, "diarization_backend", "auto")
    monkeypatch.setattr(diarizer_module.settings, "hf_token", "fake-token")

    def boom(segments, audio_path):
        raise RuntimeError("модель недоступна")

    monkeypatch.setattr(diarizer_module, "diarize_by_pyannote", boom)
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [])
    segments = [Segment(start=0.0, end=3.0, text="Мне нужно заблокировать карту")]
    result = diarize(audio, segments, transcriber=None)
    assert result[0].speaker == CLIENT

def test_diarize_pyannote_backend_forced_reraises_on_failure(monkeypatch, tmp_path):
    """backend="pyannote" выбран явно — при отказе модели ошибка должна дойти до вызывающего кода, а не молча подмениться эвристикой."""
    audio = tmp_path / "mono.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(diarizer_module, "is_stereo", lambda path: False)
    monkeypatch.setattr(diarizer_module.settings, "diarization_backend", "pyannote")

    def boom(segments, audio_path):
        raise RuntimeError("модель недоступна")

    monkeypatch.setattr(diarizer_module, "diarize_by_pyannote", boom)
    with pytest.raises(RuntimeError, match="модель недоступна"):
        diarize(audio, [], transcriber=None)

def test_diarize_pause_heuristic_backend_never_calls_pyannote(monkeypatch, tmp_path):
    audio = tmp_path / "mono.wav"
    audio.write_bytes(b"RIFF")
    monkeypatch.setattr(diarizer_module, "is_stereo", lambda path: False)
    monkeypatch.setattr(diarizer_module.settings, "diarization_backend", "pause_heuristic")
    monkeypatch.setattr(diarizer_module.settings, "hf_token", "fake-token")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("pyannote не должен вызываться при backend=pause_heuristic")

    monkeypatch.setattr(diarizer_module, "diarize_by_pyannote", fail_if_called)
    monkeypatch.setattr(diarizer_module, "_turn_boundaries", lambda path: [4.0])
    segments = [
        Segment(start=0.0, end=3.0, text="Добрый день"),
        Segment(start=5.0, end=7.0, text="Здравствуйте"),
    ]
    result = diarize(audio, segments, transcriber=None)
    assert [s.speaker for s in result] == [OPERATOR, CLIENT]
