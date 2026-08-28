"""Тесты подсчёта метрик по test_data: нормализация текста для WER и точность атрибуции ролей для диаризации."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_wer_eval import (  # noqa: E402
    hypothesis_words,
    normalize,
    reference_words,
    role_accuracy,
)


def test_normalize_lowercases():
    assert normalize("Добрый День") == "добрый день"


def test_normalize_strips_punctuation():
    assert normalize("Здравствуйте, МТБанк!") == "здравствуйте мтбанк"


def test_normalize_collapses_whitespace():
    assert normalize("добрый   день\n\nвсем") == "добрый день всем"


def test_normalize_unifies_yo():
    """ё/е — типичное расхождение Whisper с эталоном, не считаем это ошибкой."""
    assert normalize("ещё") == normalize("еще")


def test_reference_words_keeps_speaker_of_each_word(tmp_path):
    """Разметка «Спикер: текст» в эталонах есть, но раньше отбрасывалась — из-за этого диаризация, один из четырёх подпунктов двадцатибалльного критерия, не измерялась вообще."""
    ref = tmp_path / "ref.txt"
    ref.write_text("Оператор: Добрый день!\nКлиент: Здравствуйте.\n", encoding="utf-8")

    words = reference_words(ref)

    assert [w for w, _ in words] == ["добрый", "день", "здравствуйте"]
    assert [s for _, s in words] == ["Оператор", "Оператор", "Клиент"]


class _Seg:
    def __init__(self, speaker, text):
        self.speaker = speaker
        self.text = text


def test_hypothesis_words_flattens_diarized_segments():
    words = hypothesis_words([_Seg("Оператор", "Добрый день!"), _Seg("Клиент", "Здравствуйте.")])
    assert words == [("добрый", "Оператор"), ("день", "Оператор"), ("здравствуйте", "Клиент")]


def test_role_accuracy_all_correct():
    reference = [("добрый", "Оператор"), ("день", "Оператор"), ("здравствуйте", "Клиент")]
    assert role_accuracy(reference, list(reference)) == (3, 3)


def test_role_accuracy_counts_wrong_speaker():
    reference = [("добрый", "Оператор"), ("день", "Оператор"), ("здравствуйте", "Клиент")]
    hypothesis = [("добрый", "Оператор"), ("день", "Клиент"), ("здравствуйте", "Клиент")]
    assert role_accuracy(reference, hypothesis) == (2, 3)


def test_role_accuracy_ignores_words_asr_did_not_recognize():
    """Вставки и удаления в знаменатель не идут: там сравнивать нечего, и включать их значило бы смешивать ошибку распознавания с ошибкой диаризации."""
    reference = [("добрый", "Оператор"), ("день", "Оператор"), ("здравствуйте", "Клиент")]
    hypothesis = [("добрый", "Оператор"), ("здравствуйте", "Клиент")]
    correct, matched = role_accuracy(reference, hypothesis)
    assert matched == 2
    assert correct == 2


def test_role_accuracy_on_empty_input():
    assert role_accuracy([], []) == (0, 0)
