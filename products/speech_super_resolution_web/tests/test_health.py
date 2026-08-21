from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi.testclient import TestClient

from backend.app import app


def load_health() -> dict[str, object]:
    client = TestClient(app)
    response = client.get("/api/health")
    if response.status_code != 200:
        raise AssertionError(f"health status {response.status_code}: {response.text}")
    payload = response.json()
    if payload.get("app") != "speech_super_resolution_web":
        raise AssertionError(f"unexpected app id: {payload}")
    if payload.get("task") != "speech_super_resolution":
        raise AssertionError(f"unexpected task: {payload}")
    sample_count = payload.get("sample_count")
    if not isinstance(sample_count, int) or sample_count < 1:
        raise AssertionError(f"sample_count must be >= 1: {payload}")
    return payload


def main() -> None:
    payload = load_health()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
