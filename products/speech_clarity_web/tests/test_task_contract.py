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
from backend.runtime import tool_is_available


def _client() -> TestClient:
    return TestClient(app)


def _create_task(client: TestClient, sample_path: str) -> dict[str, object]:
    response = client.post(
        "/api/tasks",
        data={"source_type": "sample", "sample_path": sample_path},
    )
    if response.status_code != 200:
        raise AssertionError(f"create task {response.status_code}: {response.text}")
    payload = response.json()
    if not payload.get("task_id"):
        raise AssertionError(f"missing task_id: {payload}")
    if payload.get("status") != "ready":
        raise AssertionError(f"new task should be ready: {payload}")
    return payload


def _run(client: TestClient, task_id: str, tool: str, input_ref: str):
    return client.post(
        f"/api/tasks/{task_id}/runs",
        data={"tool": tool, "input_ref": input_ref, "waveform_seconds": "4"},
    )


def main() -> None:
    client = _client()
    report: dict[str, object] = {}

    pipeline = client.post("/api/pipeline")
    if pipeline.status_code != 404:
        raise AssertionError(f"pipeline must not exist, got {pipeline.status_code}")
    report["pipeline_status"] = pipeline.status_code

    task = _create_task(client, "assets/noisy_input.wav")
    task_id = str(task["task_id"])
    report["task_id"] = task_id
    listed = client.get("/api/tasks")
    if listed.status_code != 200:
        raise AssertionError(f"list tasks {listed.status_code}: {listed.text}")
    ids = [item.get("task_id") for item in listed.json().get("tasks", [])]
    if task_id not in ids:
        raise AssertionError(f"created task missing from list: {ids}")
    report["listed"] = True

    unknown = _run(client, task_id, "mdx", "input")
    if unknown.status_code != 400:
        raise AssertionError(f"unknown tool should 400, got {unknown.status_code}: {unknown.text}")
    pipeline_tool = _run(client, task_id, "pipeline", "input")
    if pipeline_tool.status_code != 400:
        raise AssertionError(f"pipeline tool should 400, got {pipeline_tool.status_code}")
    bad_ref = _run(client, task_id, "enhance", "step:missing:enhanced")
    if bad_ref.status_code != 400:
        raise AssertionError(f"bad input_ref should 400, got {bad_ref.status_code}: {bad_ref.text}")
    report["unknown_tool"] = unknown.status_code
    report["bad_input_ref"] = bad_ref.status_code

    available = {tool_id: tool_is_available(tool_id) for tool_id in ("enhance", "separate", "super_resolve")}
    report["available"] = available

    for tool_id, is_ready in available.items():
        if is_ready:
            continue
        blocked = _run(client, task_id, tool_id, "input")
        if blocked.status_code != 503:
            raise AssertionError(f"{tool_id} unavailable should 503, got {blocked.status_code}: {blocked.text}")
        report[f"{tool_id}_unavailable"] = 503

    if available["enhance"]:
        enhance = _run(client, task_id, "enhance", "input")
        if enhance.status_code != 200:
            raise AssertionError(f"enhance failed {enhance.status_code}: {enhance.text}")
        body = enhance.json()
        audio = client.get(body["outputs"][0]["audio_url"])
        if audio.status_code != 200:
            raise AssertionError(f"enhance audio {audio.status_code}")
        report["enhance_run"] = body["run_id"]
        report["enhance_process_seconds"] = body["timing"]["process_seconds"]
        replay = client.get(f"/api/tasks/{task_id}/runs/{body['run_id']}")
        if replay.status_code != 200:
            raise AssertionError(f"get run {replay.status_code}: {replay.text}")
        replay_analysis = replay.json().get("analysis") or {}
        if replay_analysis.get("input_sample_rate") is None or replay_analysis.get("output_sample_rate") is None:
            raise AssertionError(f"run analysis missing sample rates: {replay_analysis}")
        report["enhance_replay_sr"] = [
            replay_analysis.get("input_sample_rate"),
            replay_analysis.get("output_sample_rate"),
        ]

    if available["separate"]:
        mix_task = _create_task(client, "assets/mixture_input.wav")
        mix_id = str(mix_task["task_id"])
        separated = _run(client, mix_id, "separate", "input")
        if separated.status_code != 200:
            raise AssertionError(f"separate failed {separated.status_code}: {separated.text}")
        sep_body = separated.json()
        output_ids = [item["id"] for item in sep_body["outputs"]]
        if output_ids != ["speaker-1", "speaker-2"]:
            raise AssertionError(f"separate outputs {output_ids}")
        report["separate_run"] = sep_body["run_id"]
        if available["enhance"]:
            composed = _run(
                client,
                mix_id,
                "enhance",
                f"step:{sep_body['run_id']}:speaker-1",
            )
            if composed.status_code != 200:
                raise AssertionError(f"compose enhance after separate failed {composed.status_code}: {composed.text}")
            if composed.json()["input_ref"] != f"step:{sep_body['run_id']}:speaker-1":
                raise AssertionError("compose input_ref mismatch")
            report["compose_enhance_on_speaker_1"] = composed.json()["run_id"]

    if available["super_resolve"]:
        sr_task = _create_task(client, "assets/bandlimited_input.wav")
        sr_id = str(sr_task["task_id"])
        resolved = _run(client, sr_id, "super_resolve", "input")
        if resolved.status_code != 200:
            raise AssertionError(f"super_resolve failed {resolved.status_code}: {resolved.text}")
        body = resolved.json()
        analysis = body["analysis"]
        input_sr = analysis.get("input_sample_rate")
        output_sr = analysis.get("output_sample_rate")
        if input_sr != 16000:
            raise AssertionError(f"bandlimited input should be 16000 Hz, got {input_sr}")
        if output_sr != 48000:
            raise AssertionError(f"super_resolve output should be 48000 Hz, got {output_sr}")
        if input_sr == output_sr:
            raise AssertionError("super_resolve must not report identical sample rates")
        report["super_resolve_input_sr"] = input_sr
        report["super_resolve_output_sr"] = output_sr

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
