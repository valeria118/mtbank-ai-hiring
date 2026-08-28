#!/usr/bin/env python3
"""Генерация тестового аудио для WER-оценки."""
from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path

import edge_tts
from pydub import AudioSegment

HERE = Path(__file__).parent

OPERATOR_VOICE = "ru-RU-SvetlanaNeural"
CLIENT_VOICE = "ru-RU-DmitryNeural"

OPERATOR = "Оператор"
CLIENT = "Клиент"

PAUSE_MS = 1000

DIALOG: list[tuple[str, str]] = [
    (OPERATOR, "Добрый день, МТБанк, меня зовут Анна, чем могу помочь?"),
    (CLIENT, "Здравствуйте. Хочу узнать про условия по кредиту наличными."),
    (OPERATOR, "Конечно, подскажите, пожалуйста, какая сумма вас интересует и на какой срок?"),
    (CLIENT, "Примерно десять тысяч белорусских рублей, на год."),
    (OPERATOR, "Отлично. На данный момент ставка от четырнадцати и девяти процентов годовых, "
               "решение за пятнадцать минут. Вы уже являетесь клиентом МТБанка?"),
    (CLIENT, "Да, у меня есть карточка ваша."),
    (OPERATOR, "Прекрасно, тогда для вас действуют специальные условия. Ежемесячный платёж "
               "составит около девятисот рублей. Вам удобно подать заявку онлайн через "
               "приложение или предпочитаете приехать в отделение?"),
    (CLIENT, "Лучше онлайн. Но у меня вопрос — если я захочу досрочно погасить, есть штрафы?"),
    (OPERATOR, "Нет, досрочное погашение без штрафов и комиссий, в любое время и в любом объёме."),
    (CLIENT, "Хорошо, а страховка обязательна?"),
    (OPERATOR, "Страхование жизни подключается по вашему желанию, это не обязательное условие "
               "получения кредита. Однако при подключении страховки ставка может быть немного снижена."),
    (CLIENT, "Понятно. Тогда я попробую подать через приложение."),
    (OPERATOR, "Отлично. Если возникнут вопросы в процессе заполнения — звоните, мы поможем. "
               "Также могу отправить вам краткую инструкцию на email, если хотите."),
    (CLIENT, "Да, пожалуйста, отправьте."),
    (OPERATOR, "Хорошо, подскажите ваш email."),
    (CLIENT, "Михаил собака пример точка бай."),
    (OPERATOR, "Записала. В течение нескольких минут получите письмо с инструкцией и ссылкой "
               "на заявку. Есть ещё вопросы?"),
    (CLIENT, "Нет, всё понятно, спасибо."),
    (OPERATOR, "Спасибо за обращение в МТБанк, хорошего дня!"),
    (CLIENT, "И вам, до свидания."),
]

COMPLAINT_DIALOG: list[tuple[str, str]] = [
    (OPERATOR, "МТБанк, оператор Ирина, слушаю вас."),
    (CLIENT, "Здравствуйте, у меня третий день не проходит платёж по карте, "
             "я уже второй раз звоню, и никто ничего не делает!"),
    (OPERATOR, "Понимаю ваше возмущение, давайте разберёмся. Назовите, пожалуйста, "
               "последние четыре цифры карты."),
    (CLIENT, "Четыре восемь два один. И я хочу сказать, что если сегодня не решится, "
             "я закрываю все счета и ухожу в другой банк."),
    (OPERATOR, "Вижу вашу заявку. Платёж заблокирован системой антифрода. "
               "Я сейчас передам обращение в профильное подразделение с высоким приоритетом."),
    (CLIENT, "И сколько ждать?"),
    (OPERATOR, "В течение двух часов с вами свяжется специалист. "
               "У нас гарантированное одобрение разблокировки для клиентов вашего статуса."),
    (CLIENT, "Хорошо, жду. До свидания."),
    (OPERATOR, "Всего доброго."),
]

MONOLOGUES: list[tuple[str, str, str]] = [
    ("short_credit_question", CLIENT,
     "Здравствуйте, подскажите пожалуйста, какие сейчас условия по кредиту наличными "
     "на пятнадцать тысяч белорусских рублей сроком на два года, и что нужно из документов."),
    ("short_complaint", CLIENT,
     "Я очень недоволен обслуживанием. Мне заблокировали карту без предупреждения, "
     "деньги висят, дозвониться невозможно. Это уже второй случай за месяц, "
     "я буду писать жалобу в Национальный банк."),
    ("short_transfer_question", CLIENT,
     "Добрый день, хочу уточнить комиссию за перевод в другой банк по системе "
     "мгновенных платежей, и есть ли лимит на сумму в сутки."),
    ("short_mortgage_question", CLIENT,
     "Скажите, а какая минимальная сумма первоначального взноса по ипотеке "
     "на строящееся жильё, и рассматриваете ли вы доход супруги."),
    ("short_card_block", CLIENT,
     "Мне нужно срочно заблокировать карту, я потерял её сегодня в метро, "
     "и по ней могли уже что-то оплатить."),
    ("short_operator_greeting_variant", OPERATOR,
     "Добрый вечер, вы позвонили в контакт-центр МТБанка, меня зовут Дмитрий, "
     "чем я могу быть вам полезен сегодня?"),
]

async def synth_line(text: str, voice: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))

def _equal_length(a: AudioSegment, b: AudioSegment) -> tuple[AudioSegment, AudioSegment]:
    """from_mono_audiosegments требует одинаковое число сэмплов в каналах."""
    n_a, n_b = len(a.get_array_of_samples()), len(b.get_array_of_samples())
    target = max(n_a, n_b)

    def pad(seg: AudioSegment, n: int) -> AudioSegment:
        if n >= target:
            return seg
        extra_ms = (target - n) * 1000 // seg.frame_rate + 1
        seg = seg + AudioSegment.silent(duration=extra_ms, frame_rate=seg.frame_rate)
        return seg._spawn(seg.get_array_of_samples()[:target])

    return pad(a, n_a), pad(b, n_b)

async def build_track(lines: list[tuple[str, str]], tmp_dir: Path) -> tuple[AudioSegment, AudioSegment]:
    """Возвращает (дорожка оператора, дорожка клиента) одинаковой длины."""
    operator_track = AudioSegment.silent(duration=0)
    client_track = AudioSegment.silent(duration=0)

    for i, (speaker, text) in enumerate(lines):
        voice = OPERATOR_VOICE if speaker == OPERATOR else CLIENT_VOICE
        line_path = tmp_dir / f"line_{i:03d}.mp3"
        await synth_line(text, voice, line_path)
        clip = AudioSegment.from_file(line_path)
        silence = AudioSegment.silent(duration=len(clip))
        pause = AudioSegment.silent(duration=PAUSE_MS)

        if speaker == OPERATOR:
            operator_track += clip + pause
            client_track += silence + pause
        else:
            operator_track += silence + pause
            client_track += clip + pause

    return _equal_length(operator_track, client_track)

def write_reference(lines: list[tuple[str, str]], path: Path) -> None:
    path.write_text(
        "\n".join(f"{speaker}: {text}" for speaker, text in lines) + "\n",
        encoding="utf-8",
    )

async def build_dialog(lines: list[tuple[str, str]], base_name: str, make_8khz: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        operator_track, client_track = await build_track(lines, tmp_dir)

        stereo = AudioSegment.from_mono_audiosegments(operator_track, client_track)
        stereo = stereo.set_frame_rate(16000)
        stereo.export(HERE / f"{base_name}_stereo.wav", format="wav")

        mono = stereo.set_channels(1)
        mono.export(HERE / f"{base_name}_mono.wav", format="wav")

        if make_8khz:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(HERE / f"{base_name}_mono.wav"),
                 "-ar", "8000", "-acodec", "pcm_mulaw", str(HERE / f"{base_name}_8khz.wav")],
                check=True, capture_output=True,
            )

    write_reference(lines, HERE / f"{base_name}_reference.txt")
    print(f"✓ {base_name}: {len(stereo) / 1000:.1f} сек")

async def build_monologues() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for name, speaker, text in MONOLOGUES:
            voice = OPERATOR_VOICE if speaker == OPERATOR else CLIENT_VOICE
            mp3_path = tmp_dir / f"{name}.mp3"
            await synth_line(text, voice, mp3_path)
            clip = AudioSegment.from_file(mp3_path).set_frame_rate(16000).set_channels(1)
            clip.export(HERE / f"{name}.wav", format="wav")
            write_reference([(speaker, text)], HERE / f"{name}_reference.txt")
            print(f"✓ {name}: {len(clip) / 1000:.1f} сек")

FORMAT_VARIANTS: list[tuple[str, str]] = [
    ("short_credit_question", "mp3"),
    ("short_complaint", "ogg"),
]

def build_format_variants() -> None:
    for name, fmt in FORMAT_VARIANTS:
        src = HERE / f"{name}.wav"
        clip = AudioSegment.from_file(src)
        out_path = HERE / f"{name}.{fmt}"
        clip.export(out_path, format=fmt)
        print(f"✓ {name}.{fmt} (формат-вариант {src.name}, тот же эталон)")

async def main() -> None:
    await build_dialog(DIALOG, "call_dialog", make_8khz=True)
    await build_dialog(COMPLAINT_DIALOG, "call_complaint", make_8khz=False)
    await build_monologues()
    build_format_variants()

    total_ms = sum(
        len(AudioSegment.from_file(p))
        for p in HERE.glob("*.wav")
        if not p.name.endswith("_8khz.wav")
    )
    print(f"\nСуммарная длительность (без дублей 8kHz): {total_ms / 1000 / 60:.1f} мин")
    if total_ms < 5 * 60 * 1000:
        print("⚠️  Меньше требуемых ТЗ 5 минут — добавьте реплик в диалоги")

if __name__ == "__main__":
    asyncio.run(main())
