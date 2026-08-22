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
from backend.runtime import APP_NAME, SCENE_NAME, TOOL_IDS, TOOL_SPECS


def load_health() -> dict[str, object]:
    client = TestClient(app)
    response = client.get("/api/health")
    if response.status_code != 200:
        raise AssertionError(f"health status {response.status_code}: {response.text}")
    payload = response.json()
    if payload.get("app") != APP_NAME:
        raise AssertionError(f"unexpected app id: {payload}")
    if payload.get("scene") != SCENE_NAME:
        raise AssertionError(f"unexpected scene: {payload}")
    tools = payload.get("tools")
    if not isinstance(tools, list):
        raise AssertionError(f"tools must be a list: {payload}")
    names = [item.get("id") for item in tools]
    if names != list(TOOL_IDS):
        raise AssertionError(f"tools order/id mismatch {names} != {list(TOOL_IDS)}")
    by_id = {item.get("id"): item for item in tools}
    for tool_id, spec in TOOL_SPECS.items():
        outputs = by_id[tool_id].get("outputs")
        if outputs != list(spec["outputs"]):
            raise AssertionError(f"{tool_id} outputs {outputs} != {list(spec['outputs'])}")
        if "available" not in by_id[tool_id]:
            raise AssertionError(f"{tool_id} missing available")
    sample_count = payload.get("sample_count")
    if not isinstance(sample_count, int) or sample_count < 3:
        raise AssertionError(f"sample_count must be >= 3: {payload}")
    if payload.get("tasks_dir") != "workspace/tasks":
        raise AssertionError(f"tasks_dir must be workspace/tasks: {payload}")
    return payload


def main() -> None:
    payload = load_health()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
