#!/usr/bin/env python3
"""Прогон всех файлов из test_data/ через пайплайн ASR + диаризации и подсчёт метрик против эталонных транскриптов."""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).parent.parent))

from asr.diarizer import diarize  # noqa: E402
from asr.transcriber import Transcriber, probe_duration  # noqa: E402

TEST_DATA = Path(__file__).parent.parent / "test_data"

CASES = [
    ("call_dialog_mono.wav", "call_dialog_reference.txt"),
    ("call_dialog_stereo.wav", "call_dialog_reference.txt"),
    ("call_dialog_8khz.wav", "call_dialog_reference.txt"),
    ("call_complaint_mono.wav", "call_complaint_reference.txt"),
    ("call_complaint_stereo.wav", "call_complaint_reference.txt"),
    ("short_credit_question.wav", "short_credit_question_reference.txt"),
    ("short_complaint.wav", "short_complaint_reference.txt"),
    ("short_transfer_question.wav", "short_transfer_question_reference.txt"),
    ("short_mortgage_question.wav", "short_mortgage_question_reference.txt"),
    ("short_card_block.wav", "short_card_block_reference.txt"),
    ("short_operator_greeting_variant.wav", "short_operator_greeting_variant_reference.txt"),
    ("short_credit_question.mp3", "short_credit_question_reference.txt"),
    ("short_complaint.ogg", "short_complaint_reference.txt"),
]

PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")

def normalize(text: str) -> str:
    """Приводим гипотезу и эталон к одному виду: нижний регистр, без пунктуации, ё→е, схлопнутые пробелы."""
    text = text.lower().replace("ё", "е")
    text = PUNCTUATION_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()

def reference_words(ref_path: Path) -> list[tuple[str, str]]:
    """Эталон -> список (слово, говорящий)."""
    words: list[tuple[str, str]] = []
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        speaker, text = line.split(":", 1)
        for word in normalize(text).split():
            words.append((word, speaker.strip()))
    return words

def hypothesis_words(diarized) -> list[tuple[str, str]]:
    return [
        (word, segment.speaker)
        for segment in diarized
        for word in normalize(segment.text).split()
    ]

def role_accuracy(
    reference: list[tuple[str, str]], hypothesis: list[tuple[str, str]]
) -> tuple[int, int]:
    """(верно атрибутированных слов, всего сопоставленных слов)."""
    ref_text = " ".join(word for word, _ in reference)
    hyp_text = " ".join(word for word, _ in hypothesis)
    if not ref_text or not hyp_text:
        return 0, 0

    output = jiwer.process_words(ref_text, hyp_text)
    correct = matched = 0
    for chunk in output.alignments[0]:
        if chunk.type not in ("equal", "substitute"):
            continue
        for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
            ref_speaker = reference[chunk.ref_start_idx + offset][1]
            hyp_speaker = hypothesis[chunk.hyp_start_idx + offset][1]
            matched += 1
            correct += int(ref_speaker == hyp_speaker)
    return correct, matched

def main() -> None:
    transcriber = Transcriber()
    print("Прогреваю модель…")
    transcriber.warmup()

    rows = []

    for audio_name, ref_name in CASES:
        audio_path = TEST_DATA / audio_name
        ref_path = TEST_DATA / ref_name
        if not audio_path.exists() or not ref_path.exists():
            print(f"⚠️  пропуск {audio_name}: файл не найден")
            continue

        duration = probe_duration(audio_path)
        started = time.perf_counter()
        diarized = diarize(audio_path, transcriber=transcriber)
        elapsed = time.perf_counter() - started

        ref_words = reference_words(ref_path)
        hyp_words = hypothesis_words(diarized)
        reference = " ".join(word for word, _ in ref_words)
        hypothesis = " ".join(word for word, _ in hyp_words)

        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        correct_roles, matched_roles = role_accuracy(ref_words, hyp_words)
        n_ref_speakers = len({speaker for _, speaker in ref_words})

        rows.append({
            "name": audio_name,
            "duration": duration,
            "n_words": len(ref_words),
            "wer": wer * 100,
            "cer": cer * 100,
            "elapsed": elapsed,
            "roles_correct": correct_roles,
            "roles_matched": matched_roles,
            "dialog": n_ref_speakers > 1,
        })
        roles_pct = 100 * correct_roles / matched_roles if matched_roles else 0.0
        print(f"{audio_name}: WER={wer * 100:.1f}% CER={cer * 100:.1f}% "
              f"роли={roles_pct:.1f}% ({elapsed:.1f} сек на {duration:.0f} сек аудио)")

    print("\n| Файл | Длит., сек | Слов (эталон) | WER, % | CER, % | Роли, % | Время ASR+диар., сек |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        roles = (
            f"{100 * row['roles_correct'] / row['roles_matched']:.1f}"
            if row["roles_matched"] else "—"
        )
        print(
            f"| `{row['name']}` | {row['duration']:.0f} | {row['n_words']} | "
            f"{row['wer']:.1f} | {row['cer']:.1f} | {roles} | {row['elapsed']:.1f} |"
        )

    if not rows:
        return

    avg_wer = sum(r["wer"] for r in rows) / len(rows)
    total_audio = sum(r["duration"] for r in rows)
    total_time = sum(r["elapsed"] for r in rows)
    print(f"\n**Средний WER: {avg_wer:.1f}%** | "
          f"Суммарно аудио: {total_audio / 60:.1f} мин | "
          f"RTF: {total_time / total_audio:.2f}")

    dialogs = [r for r in rows if r["dialog"] and r["roles_matched"]]
    if dialogs:
        correct = sum(r["roles_correct"] for r in dialogs)
        matched = sum(r["roles_matched"] for r in dialogs)
        print(f"**Точность ролей на диалогах: {100 * correct / matched:.1f}%** "
              f"({correct} из {matched} сопоставленных слов, {len(dialogs)} файлов). "
              "Односпикерные записи в эту цифру не входят: на них диаризации "
              "нечего решать.")

if __name__ == "__main__":
    main()
