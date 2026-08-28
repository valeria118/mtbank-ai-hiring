"""SQLite-хранилище результатов анализа звонков."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings
from logging_utils import get_logger

logger = get_logger("storage")

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    topic TEXT,
    priority TEXT,
    quality_total INTEGER,
    compliance_passed INTEGER,
    summary TEXT,
    raw_json TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    db_file = Path(settings.db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def _safe_get(section: Any, key: str) -> Any:
    """Оркестратор подменяет результат упавшего агента на {"error": ...} — из такой секции нужные поля просто отсутствуют."""
    if isinstance(section, dict):
        return section.get(key)
    return None


def save_analysis(result: dict[str, Any]) -> int:
    classification = result.get("classification") or {}
    quality = result.get("quality_score") or {}
    compliance = result.get("compliance") or {}
    passed = _safe_get(compliance, "passed")

    with _connect() as conn:
        cursor = conn.execute(
            """INSERT INTO calls
               (created_at, topic, priority, quality_total, compliance_passed, summary, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                _safe_get(classification, "topic"),
                _safe_get(classification, "priority"),
                _safe_get(quality, "total"),
                None if passed is None else int(bool(passed)),
                result.get("summary"),
                json.dumps(result, ensure_ascii=False),
            ),
        )
        row_id = cursor.lastrowid
    logger.info("storage.saved", extra={"extra_data": {"id": row_id}})
    return int(row_id)


def recent_analyses(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, created_at, topic, priority, quality_total, compliance_passed, summary
               FROM calls ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "created_at": r["created_at"],
            "topic": r["topic"],
            "priority": r["priority"],
            "quality_total": r["quality_total"],
            "compliance_passed": None if r["compliance_passed"] is None else bool(r["compliance_passed"]),
            "summary": r["summary"],
        }
        for r in rows
    ]


async def save_analysis_async(result: dict[str, Any]) -> int:
    return await asyncio.to_thread(save_analysis, result)


async def recent_analyses_async(limit: int = 20) -> list[dict[str, Any]]:
    return await asyncio.to_thread(recent_analyses, limit)
