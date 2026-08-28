"""Диаризация «Оператор / Клиент»."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from asr.transcriber import Segment
from config import settings
from logging_utils import get_logger, timed

logger = get_logger("asr.diarizer")

OPERATOR = "Оператор"
CLIENT = "Клиент"

TURN_SILENCE_MS = 1500

PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

_OPERATOR_GREETING_RE = re.compile(
    r"мт\s*банк|банка?\b.{0,20}(оператор|слушаю)|"
    r"меня зовут|слушаю вас|"
    r"чем (я )?могу (вам )?(быть )?(полезен|помочь)|"
    r"контакт-?центр",
    re.IGNORECASE,
)

def _looks_like_operator_greeting(text: str) -> bool:
    return bool(_OPERATOR_GREETING_RE.search(text))

@dataclass
class DiarizedSegment:
    speaker: str
    start: float
    end: float
    text: str

def is_stereo(audio_path: str | Path) -> bool:
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(audio_path))
    return audio.channels >= 2

def _turn_boundaries(audio_path: str | Path) -> list[float]:
    """Моменты (сек) значимой тишины в аудио — трактуются как границы смены говорящего в моно-эвристике."""
    from pydub import AudioSegment
    from pydub.silence import detect_silence

    audio = AudioSegment.from_file(str(audio_path))
    silences = detect_silence(audio, min_silence_len=TURN_SILENCE_MS, silence_thresh=audio.dBFS - 16)

    merged: list[list[int]] = []
    for start, end in silences:
        if merged and start - merged[-1][1] < TURN_SILENCE_MS:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [(start + end) / 2 / 1000 for start, end in merged]

def diarize_by_pause_heuristic(
    segments: list[Segment], audio_path: str | Path
) -> list[DiarizedSegment]:
    boundaries = _turn_boundaries(audio_path) if segments else []

    if not boundaries and segments:
        full_text = " ".join(seg.text for seg in segments)
        current_speaker = OPERATOR if _looks_like_operator_greeting(full_text) else CLIENT
    else:
        current_speaker = OPERATOR

    result: list[DiarizedSegment] = []
    boundary_idx = 0
    for seg in segments:
        while boundary_idx < len(boundaries) and boundaries[boundary_idx] < seg.start:
            current_speaker = CLIENT if current_speaker == OPERATOR else OPERATOR
            boundary_idx += 1
        result.append(
            DiarizedSegment(speaker=current_speaker, start=seg.start, end=seg.end, text=seg.text)
        )
    return result

def diarize_by_stereo_channels(audio_path: str | Path, transcriber) -> list[DiarizedSegment]:
    """Разделяем стерео-файл на 2 моно-канала и транскрибируем каждый отдельно, затем сливаем по таймкодам."""
    import tempfile

    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(audio_path))
    channels = audio.split_to_mono()
    channel_speakers = [OPERATOR, CLIENT]

    merged: list[DiarizedSegment] = []
    with tempfile.TemporaryDirectory(prefix="diarize_") as tmp_dir:
        for index, (channel_audio, speaker) in enumerate(zip(channels, channel_speakers)):
            tmp_path = Path(tmp_dir) / f"channel{index}.wav"
            channel_audio.export(tmp_path, format="wav")
            segments = transcriber.transcribe(tmp_path)
            merged.extend(
                DiarizedSegment(speaker=speaker, start=s.start, end=s.end, text=s.text)
                for s in segments
            )
    merged.sort(key=lambda s: s.start)
    return merged

_pyannote_pipeline_cache = None

def _get_pyannote_pipeline():
    """Ленивая загрузка и кэширование pyannote-пайплайна (тяжёлая модель — загружать её на каждый запрос слишком дорого)."""
    global _pyannote_pipeline_cache
    if _pyannote_pipeline_cache is None:
        from pyannote.audio import Pipeline as PyannotePipeline

        pipeline = PyannotePipeline.from_pretrained(
            PYANNOTE_MODEL, token=settings.hf_token or None
        )
        if pipeline is None:
            raise RuntimeError(
                f"Pipeline.from_pretrained вернул None для {PYANNOTE_MODEL} — "
                "проверьте HF_TOKEN и что условия доступа к модели приняты "
                "на huggingface.co под этим аккаунтом."
            )
        _pyannote_pipeline_cache = pipeline
    return _pyannote_pipeline_cache

def diarize_by_pyannote(segments: list[Segment], audio_path: str | Path) -> list[DiarizedSegment]:
    """Диаризация по эмбеддингам голоса. pyannote сам находит границы речи и группирует их в произвольные кластеры (SPEAKER_00, SPEAKER_01, ...) — он не знает про роли "Оператор"/"Клиент", поэтому."""
    pipeline = _get_pyannote_pipeline()
    output = pipeline(str(audio_path))
    annotation = getattr(output, "speaker_diarization", output)

    turns = sorted(
        ((turn.start, turn.end, label) for turn, _, label in annotation.itertracks(yield_label=True)),
        key=lambda t: t[0],
    )
    if not turns:
        logger.warning("diarize.pyannote_no_turns", extra={"extra_data": {"audio_path": str(audio_path)}})
        return diarize_by_pause_heuristic(segments, audio_path)

    raw: list[tuple[str, Segment]] = []
    for seg in segments:
        best_label, best_overlap = None, 0.0
        for t_start, t_end, label in turns:
            overlap = min(seg.end, t_end) - max(seg.start, t_start)
            if overlap > best_overlap:
                best_overlap, best_label = overlap, label
        if best_label is None:
            best_label = min(turns, key=lambda t: abs(t[0] - seg.start))[2]
        raw.append((best_label, seg))

    cluster_labels = sorted({label for label, _ in raw})
    operator_label = None
    for label in cluster_labels:
        cluster_text = " ".join(seg.text for lbl, seg in raw if lbl == label)
        if _looks_like_operator_greeting(cluster_text):
            operator_label = label
            break
    if operator_label is None and raw:
        operator_label = raw[0][0]

    return [
        DiarizedSegment(
            speaker=OPERATOR if label == operator_label else CLIENT,
            start=seg.start,
            end=seg.end,
            text=seg.text,
        )
        for label, seg in raw
    ]

def diarize(
    audio_path: str | Path,
    segments: list[Segment] | None = None,
    transcriber=None,
) -> list[DiarizedSegment]:
    """ASR-сегменты -> сегменты со спикерами."""
    with timed(logger, "diarize", audio_path=str(audio_path)):
        if transcriber is not None and is_stereo(audio_path):
            logger.info("diarize.strategy", extra={"extra_data": {"strategy": "stereo_channels"}})
            return diarize_by_stereo_channels(audio_path, transcriber)

        if segments is None:
            if transcriber is None:
                raise ValueError(
                    "diarize(): нужны либо готовые segments, либо transcriber, "
                    "чтобы их получить."
                )
            segments = transcriber.transcribe(audio_path)

        backend = settings.diarization_backend
        want_pyannote = backend == "pyannote" or (backend == "auto" and settings.hf_token)
        if want_pyannote:
            try:
                logger.info("diarize.strategy", extra={"extra_data": {"strategy": "pyannote"}})
                return diarize_by_pyannote(segments, audio_path)
            except Exception as exc:
                if backend == "pyannote":
                    raise
                logger.warning(
                    "diarize.pyannote_failed_fallback",
                    extra={"extra_data": {"error": str(exc), "audio_path": str(audio_path)}},
                )

        logger.info("diarize.strategy", extra={"extra_data": {"strategy": "pause_heuristic"}})
        return diarize_by_pause_heuristic(segments, audio_path)
