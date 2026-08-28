"""Тесты санитайзинга HF_TOKEN."""
from __future__ import annotations

import importlib
import os

import pytest

from config import Settings, _looks_like_hf_token


TEST_HF_TOKEN = "hf_test_token_for_unit_tests"


@pytest.mark.parametrize(
    "value",
    [
        TEST_HF_TOKEN,
        "hf_x",
    ],
)
def test_valid_token_passes_through(value):
    assert _looks_like_hf_token(value)
    assert Settings(hf_token=value).hf_token == value


@pytest.mark.parametrize(
    "value",
    [
        "<твой токен с huggingface.co, если решишь включать>",
        "<your-token-here>",
        "вставь_сюда_токен",
        '"hf_quoted"',
        "sk-not-a-hf-token",
        "hf_",
    ],
)
def test_bogus_token_becomes_empty(value):
    assert not _looks_like_hf_token(value)
    assert Settings(hf_token=value).hf_token == ""


def test_empty_token_stays_empty():
    assert Settings(hf_token="").hf_token == ""


def test_surrounding_whitespace_is_trimmed():
    token = TEST_HF_TOKEN
    assert Settings(hf_token=f"  {token}\n").hf_token == token


def test_bogus_token_is_removed_from_environment(monkeypatch):
    """Бракованный токен должен исчезнуть и из окружения."""
    monkeypatch.setenv("HF_TOKEN", "<твой токен с huggingface.co>")

    import config

    importlib.reload(config)
    try:
        assert config.settings.hf_token == ""
        assert "HF_TOKEN" not in os.environ
    finally:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        importlib.reload(config)


def test_valid_token_is_kept_in_environment(monkeypatch):
    token = TEST_HF_TOKEN
    monkeypatch.setenv("HF_TOKEN", token)

    import config

    importlib.reload(config)
    try:
        assert config.settings.hf_token == token
        assert os.environ.get("HF_TOKEN") == token
    finally:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        importlib.reload(config)