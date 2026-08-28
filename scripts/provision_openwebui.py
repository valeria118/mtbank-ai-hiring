#!/usr/bin/env python3
"""Одноразовая настройка OpenWebUI после старта стека."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings  # noqa: E402

MODEL_ID = "mtbank_pipeline"
MODEL_NAME = "MTBank Speech Analytics Pipeline"
WAIT_TIMEOUT_SEC = 900

def _wait_healthy(base_url: str) -> None:
    deadline = time.monotonic() + WAIT_TIMEOUT_SEC
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError(f"{base_url}/health не ответил за {WAIT_TIMEOUT_SEC} сек")

def main() -> None:
    base_url = settings.openwebui_base_url.rstrip("/")
    _wait_healthy(base_url)

    with httpx.Client(base_url=base_url, timeout=30) as client:
        resp = client.post(
            "/api/v1/auths/signin",
            json={"email": "provision@mtbank.local", "password": "provision"},
        )
        resp.raise_for_status()
        token = resp.json()["token"]
        headers = {"Authorization": f"Bearer {token}"}

        payload = {
            "id": MODEL_ID,
            "name": MODEL_NAME,
            "meta": {"capabilities": {"file_context": False}},
            "params": {},
            "access_grants": [],
        }

        resp = client.post("/api/v1/models/create", json=payload, headers=headers)
        if resp.status_code == 200:
            print(f"✓ модель {MODEL_ID}: file_context=false (создана)")
            return

        resp = client.post("/api/v1/models/model/update", json=payload, headers=headers)
        resp.raise_for_status()
        print(f"✓ модель {MODEL_ID}: file_context=false (обновлена)")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  provision_openwebui пропущен: {exc}", file=sys.stderr)
        sys.exit(0)
