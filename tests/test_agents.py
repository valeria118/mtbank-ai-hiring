"""Unit-тесты на каждого агента — LLM-вызовы замоканы (без сети/ключей)."""
import pytest

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.quality import QualityAgent, parse_total
from agents.summarizer import SummarizerAgent, count_sentences

SAMPLE_TRANSCRIPT = [
    {"speaker": "Оператор", "start": 0.0, "end": 3.0, "text": "Добрый день, МТБанк, меня зовут Анна."},
    {"speaker": "Клиент", "start": 3.5, "end": 6.0, "text": "Здравствуйте, хочу узнать про кредит наличными."},
    {"speaker": "Оператор", "start": 6.5, "end": 10.0, "text": "Ставка от четырнадцати и девяти процентов годовых."},
    {"speaker": "Клиент", "start": 10.5, "end": 12.0, "text": "Спасибо, до свидания."},
]

class FakeLLMClient:
    """Подменяет agents.llm_client.LLMClient — возвращает заранее заданный JSON."""

    def __init__(self, fixed_response: dict):
        self._fixed_response = fixed_response

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        return self._fixed_response

@pytest.mark.asyncio
async def test_classifier_returns_topic_and_priority():
    fake_llm = FakeLLMClient({"topic": "кредиты", "priority": "medium", "reasoning": "звонок про кредит"})
    agent = ClassifierAgent(fake_llm)
    result = await agent.run(SAMPLE_TRANSCRIPT)
    assert result["topic"] == "кредиты"
    assert result["priority"] == "medium"

@pytest.mark.asyncio
async def test_classifier_falls_back_on_missing_fields():
    fake_llm = FakeLLMClient({})
    agent = ClassifierAgent(fake_llm)
    result = await agent.run(SAMPLE_TRANSCRIPT)
    assert result["topic"] == "прочее"
    assert result["priority"] == "medium"

@pytest.mark.asyncio
async def test_quality_agent_checklist_shape():
    fake_llm = FakeLLMClient({
        "total": 80,
        "checklist": {"greeting": True, "need_detection": True, "solution_provided": True, "farewell": False},
        "comment": "не попрощался",
    })
    agent = QualityAgent(fake_llm)
    result = await agent.run(SAMPLE_TRANSCRIPT)
    assert result["total"] == 80
    assert result["checklist"]["farewell"] is False
    assert set(result["checklist"].keys()) == {"greeting", "need_detection", "solution_provided", "farewell"}

@pytest.mark.asyncio
async def test_compliance_agent_detects_forbidden_phrase_by_rule():
    transcript_with_violation = SAMPLE_TRANSCRIPT + [
        {"speaker": "Оператор", "start": 12.5, "end": 15.0, "text": "У нас гарантированное одобрение для всех клиентов."}
    ]
    fake_llm = FakeLLMClient({"issues": [], "passed": True})  # LLM ничего не находит
    agent = ComplianceAgent(fake_llm)
    result = await agent.run(transcript_with_violation)
    assert result["passed"] is False
    assert any("гарантированное одобрение" in issue for issue in result["issues"])

@pytest.mark.asyncio
async def test_compliance_agent_ignores_forbidden_phrase_quoted_by_client():
    """Поиск шёл по всему транскрипту, поэтому клиент, процитировавший «мне обещали гарантированное одобрение», создавал нарушение оператору."""
    transcript = [
        {"speaker": "Оператор", "start": 0.0, "end": 3.0, "text": "Добрый день, МТБанк, чем могу помочь?"},
        {"speaker": "Клиент", "start": 3.0, "end": 8.0,
         "text": "В отделении мне обещали гарантированное одобрение, а теперь отказ."},
    ]
    fake_llm = FakeLLMClient({"issues": [], "passed": True})
    agent = ComplianceAgent(fake_llm)
    result = await agent.run(transcript)
    assert result["issues"] == []
    assert result["passed"] is True

@pytest.mark.asyncio
async def test_compliance_agent_passes_clean_transcript():
    """Тема — не кредит и не страхование, обязательных раскрытий нет."""
    transcript = [
        {"speaker": "Оператор", "start": 0.0, "end": 3.0, "text": "Добрый день, МТБанк, меня зовут Анна."},
        {"speaker": "Клиент", "start": 3.0, "end": 6.0, "text": "Не работает вход в приложение."},
        {"speaker": "Оператор", "start": 6.0, "end": 10.0, "text": "Давайте переустановим приложение вместе."},
        {"speaker": "Клиент", "start": 10.0, "end": 12.0, "text": "Спасибо, помогло."},
    ]
    fake_llm = FakeLLMClient({"issues": [], "passed": True})
    agent = ComplianceAgent(fake_llm)
    result = await agent.run(transcript)
    assert result["passed"] is True
    assert result["issues"] == []

@pytest.mark.asyncio
async def test_compliance_agent_requires_credit_disclosures():
    """Треть требования ТЗ, которой раньше не было ни правилом, ни в промпте."""
    fake_llm = FakeLLMClient({"issues": [], "passed": True})
    agent = ComplianceAgent(fake_llm)
    result = await agent.run(SAMPLE_TRANSCRIPT)

    assert result["passed"] is False
    titles = " | ".join(result["issues"])
    assert "полная процентная ставка" in titles
    assert "решение о выдаче принимает банк" in titles
    assert "не является публичной офертой" in titles

@pytest.mark.asyncio
async def test_compliance_agent_accepts_disclosed_credit_call():
    transcript = [
        {"speaker": "Клиент", "start": 0.0, "end": 3.0, "text": "Хочу узнать про кредит наличными."},
        {"speaker": "Оператор", "start": 3.0, "end": 14.0, "text": (
            "Полная процентная ставка составит шестнадцать и две десятых процента годовых. "
            "Решение о выдаче принимает банк, предложение не является публичной офертой."
        )},
    ]
    fake_llm = FakeLLMClient({"issues": [], "passed": True})
    agent = ComplianceAgent(fake_llm)
    result = await agent.run(transcript)
    assert result["issues"] == []
    assert result["passed"] is True

@pytest.mark.asyncio
async def test_compliance_agent_requires_insurance_voluntariness():
    pitched = [
        {"speaker": "Клиент", "start": 0.0, "end": 3.0, "text": "А страховка нужна?"},
        {"speaker": "Оператор", "start": 3.0, "end": 8.0, "text": "Страхование жизни оформим вместе с картой."},
    ]
    disclosed = [
        {"speaker": "Клиент", "start": 0.0, "end": 3.0, "text": "А страховка нужна?"},
        {"speaker": "Оператор", "start": 3.0, "end": 9.0,
         "text": "Страхование жизни подключается по вашему желанию, это не обязательное условие."},
    ]
    fake_llm = FakeLLMClient({"issues": [], "passed": True})
    agent = ComplianceAgent(fake_llm)

    missing = await agent.run(pitched)
    assert any("добровольность страхования" in issue for issue in missing["issues"])

    ok = await agent.run(disclosed)
    assert ok["issues"] == []

@pytest.mark.asyncio
async def test_compliance_disclosures_not_required_off_topic():
    """Раскрытия привязаны к теме: в звонке про блокировку карты требовать полную стоимость кредита бессмысленно."""
    transcript = [
        {"speaker": "Клиент", "start": 0.0, "end": 4.0, "text": "У меня заблокирована карта."},
        {"speaker": "Оператор", "start": 4.0, "end": 9.0, "text": "Разблокирую, назовите последние четыре цифры."},
    ]
    fake_llm = FakeLLMClient({"issues": [], "passed": True})
    agent = ComplianceAgent(fake_llm)
    result = await agent.run(transcript)
    assert result["issues"] == []

@pytest.mark.asyncio
async def test_summarizer_agent_returns_summary_and_action_items():
    fake_llm = FakeLLMClient({
        "summary": "Клиент интересовался условиями кредита наличными.",
        "action_items": ["Отправить КП на email клиента"],
    })
    agent = SummarizerAgent(fake_llm)
    result = await agent.run(SAMPLE_TRANSCRIPT)
    assert "кредит" in result["summary"]
    assert result["action_items"] == ["Отправить КП на email клиента"]

@pytest.mark.asyncio
async def test_summarizer_agent_empty_action_items_default():
    fake_llm = FakeLLMClient({"summary": "Короткий звонок без действий."})
    agent = SummarizerAgent(fake_llm)
    result = await agent.run(SAMPLE_TRANSCRIPT)
    assert result["action_items"] == []

@pytest.mark.parametrize("raw,expected", [
    (78, 78),
    (78.0, 78),
    ("78", 78),
    ("78%", 78),
    ("8,5", 8),      # запятая как десятичный разделитель
    (8.5, 8),
    (0, 0),          # честный ноль — валидное значение
    (100, 100),
])
def test_parse_total_accepts_numbers_in_range(raw, expected):
    assert parse_total(raw) == expected

@pytest.mark.parametrize("raw", [None, "отлично", 780, -1, 100.5, True, {}])
def test_parse_total_rejects_garbage_and_out_of_range(raw):
    """Отсутствие поля и мусор — это отказ агента, а не оценка."""
    with pytest.raises(ValueError):
        parse_total(raw)

@pytest.mark.asyncio
async def test_quality_agent_missing_total_is_an_error_not_zero():
    agent = QualityAgent(FakeLLMClient({"checklist": {"greeting": True}}))
    with pytest.raises(ValueError):
        await agent.run(SAMPLE_TRANSCRIPT)

@pytest.mark.parametrize("text,expected", [
    ("Одно предложение.", 1),
    ("Первое. Второе! Третье?", 3),
    ("Без точки в конце", 1),
    ("Отправить КП, счёт на 500 руб. и т.д. Перезвонить завтра.", 2),
    ("Ставка 14.9 процента годовых. Клиент подаст заявку.", 2),
    ("", 0),
])
def test_count_sentences(text, expected):
    assert count_sentences(text) == expected

class _TwoStepLLM:
    """Первый ответ — вне диапазона, второй — исправленный."""

    def __init__(self, first: dict, second: dict):
        self._responses = [first, second]
        self.calls = 0

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls += 1
        return self._responses[min(self.calls - 1, len(self._responses) - 1)]

@pytest.mark.asyncio
async def test_summarizer_retries_when_summary_too_short():
    """ТЗ задаёт длину резюме числом (3-5 предложений), но промпт этого не гарантирует: на двенадцати реальных прогонах пять резюме оказались короче нижней границы."""
    llm = _TwoStepLLM(
        {"summary": "Клиент спросил про кредит.", "action_items": ["Перезвонить"]},
        {"summary": "Клиент спросил про кредит наличными. Оператор назвал ставку. Клиент подаст заявку онлайн."},
    )
    result = await SummarizerAgent(llm).run(SAMPLE_TRANSCRIPT)

    assert llm.calls == 2, "короткое резюме должно вызвать ровно один повтор"
    assert count_sentences(result["summary"]) == 3
    assert result["action_items"] == ["Перезвонить"]

@pytest.mark.asyncio
async def test_summarizer_does_not_retry_when_length_is_fine():
    llm = _TwoStepLLM(
        {"summary": "Первое предложение. Второе предложение. Третье предложение.", "action_items": []},
        {"summary": "не должно понадобиться"},
    )
    result = await SummarizerAgent(llm).run(SAMPLE_TRANSCRIPT)
    assert llm.calls == 1
    assert "Первое" in result["summary"]

@pytest.mark.asyncio
async def test_summarizer_keeps_original_when_retry_does_not_help():
    """Резюме уже есть, оно просто не той длины."""
    llm = _TwoStepLLM(
        {"summary": "Слишком коротко.", "action_items": []},
        {"summary": "Тоже коротко."},
    )
    result = await SummarizerAgent(llm).run(SAMPLE_TRANSCRIPT)
    assert result["summary"] == "Слишком коротко."
