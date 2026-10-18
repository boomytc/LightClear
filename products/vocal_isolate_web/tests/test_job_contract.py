from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

PRODUCT_ROOT = Path(__file__).resolve().parents[1]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)

import soundfile as sf
from fastapi.testclient import TestClient

from backend.app import app
from backend.runtime import DEFAULT_MODEL, MODEL_SPECS, model_is_available


SAMPLE_PATH = "assets/next_station_heaven.mp3"
ANALYSIS_IMAGES = ("waveform_image", "power_spectrum_image", "mel_spectrum_image")


def check_analysis(payload: dict[str, object], expected_stems: tuple[str, ...]) -> dict[str, object]:
    analysis = payload.get("analysis") or {}
    rows = analysis.get("metrics_rows")
    if not isinstance(rows, list) or len(rows) < 5:
        raise AssertionError(f"metrics_rows too short: {rows}")
    columns = list(rows[0])
    if "指标类别" not in columns:
        raise AssertionError(f"metrics_rows missing 指标类别 column: {columns}")
    for row in rows:
        if list(row) != columns:
            raise AssertionError(f"metrics_rows ragged: {row}")
    labels = [row.get("指标类别", "") for row in rows]
    for stem in expected_stems:
        if f"分轨 RMS · {stem}" not in labels:
            raise AssertionError(f"missing per-stem row for {stem}: {labels}")
    for key in ANALYSIS_IMAGES:
        value = analysis.get(key)
        if not isinstance(value, str) or not value.startswith("data:image/png;base64,"):
            raise AssertionError(f"{key} must be a png data uri, got {type(value).__name__}")
    return analysis


def check_job_payload(client: TestClient, payload: dict[str, object], model_name: str) -> dict[str, object]:
    job_id = str(payload["job_id"])
    if not job_id:
        raise AssertionError(f"missing job_id: {payload}")
    if payload.get("model_name") != model_name:
        raise AssertionError(f"payload must echo model {model_name}: {payload.get('model_name')}")

    expected_stems = tuple(MODEL_SPECS[model_name]["outputs"])
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise AssertionError(f"outputs must be a list: {payload}")
    if tuple(item.get("id") for item in outputs) != expected_stems:
        raise AssertionError(f"{model_name} stems must be {expected_stems}: {outputs}")
    input_rate = sf.info(str(PRODUCT_ROOT / payload["input_path"])).samplerate
    for item in outputs:
        path = item.get("path")
        if not isinstance(path, str) or not (PRODUCT_ROOT / path).is_file():
            raise AssertionError(f"output path must exist on disk: {item}")
        if not item.get("label"):
            raise AssertionError(f"stem must carry a label: {item}")
        stem_rate = sf.info(str(PRODUCT_ROOT / path)).samplerate
        if stem_rate != input_rate:
            raise AssertionError(
                f"demucs writes at separator samplerate; analysis aligns by resampling ({input_rate} vs {stem_rate})"
            )

    timing = payload.get("timing") or {}
    for key in ("model_initial_load_seconds", "model_ready_seconds", "process_seconds", "total_seconds"):
        value = timing.get(key)
        if not isinstance(value, (int, float)) or value < 0:
            raise AssertionError(f"timing.{key} must be a non-negative number: {timing}")

    for key in ("input_audio_url", "output_audio_url", "download_url"):
        url = payload.get(key)
        if not isinstance(url, str) or not url.startswith(f"/api/jobs/{job_id}/"):
            raise AssertionError(f"{key} must point at this job: {url}")
        route = client.get(url)
        if route.status_code != 200:
            raise AssertionError(f"{key} {route.status_code}: {route.text}")

    check_analysis(payload, expected_stems)

    for variant in ("original", *expected_stems):
        audio = client.get(f"/api/jobs/{job_id}/audio/{variant}")
        if audio.status_code != 200:
            raise AssertionError(f"audio {variant} {audio.status_code}: {audio.text}")
        if not audio.headers.get("content-type", "").startswith("audio/"):
            raise AssertionError(f"audio {variant} content-type {audio.headers.get('content-type')}")

    unavailable_stem = "guitar" if "guitar" not in expected_stems else "violin"
    unknown = client.get(f"/api/jobs/{job_id}/audio/{unavailable_stem}")
    if unknown.status_code != 404:
        raise AssertionError(f"audio {unavailable_stem} should 404, got {unknown.status_code}")

    for variant in expected_stems:
        download = client.get(f"/api/jobs/{job_id}/download/{variant}")
        if download.status_code != 200:
            raise AssertionError(f"download {variant} {download.status_code}: {download.text}")
        if "attachment" not in download.headers.get("content-disposition", ""):
            raise AssertionError(f"download {variant} must be an attachment")

    missing_job = client.get(f"/api/jobs/{'0' * 32}/download/vocals")
    if missing_job.status_code != 404:
        raise AssertionError(f"unknown job should 404, got {missing_job.status_code}")

    return payload


def main() -> None:
    client = TestClient(app)
    report: dict[str, object] = {}

    health = client.get("/api/health")
    if health.status_code != 200:
        raise AssertionError(f"health {health.status_code}: {health.text}")

    bad_model = client.post(
        "/api/isolate",
        data={"source_type": "sample", "sample_path": SAMPLE_PATH, "model_name": "mdx"},
    )
    if bad_model.status_code != 400:
        raise AssertionError(f"unknown model should 400, got {bad_model.status_code}: {bad_model.text}")
    report["rejects_unknown_model"] = True

    bad_source = client.post("/api/isolate", data={"source_type": "bogus"})
    if bad_source.status_code != 400:
        raise AssertionError(f"bad source should 400, got {bad_source.status_code}: {bad_source.text}")
    report["rejects_bad_source"] = True

    escape = client.post(
        "/api/isolate",
        data={"source_type": "sample", "sample_path": "../pyproject.toml"},
    )
    if escape.status_code != 400:
        raise AssertionError(f"path escape should 400, got {escape.status_code}: {escape.text}")
    report["rejects_path_escape"] = True

    for model_name in MODEL_SPECS:
        if not model_is_available(PRODUCT_ROOT, model_name):
            print(f"SKIP: {model_name} 权重未就绪", file=sys.stderr)
            report[f"{model_name}_skipped"] = True
            continue
        response = client.post(
            "/api/isolate",
            data={
                "source_type": "sample",
                "sample_path": SAMPLE_PATH,
                "model_name": model_name,
                "waveform_seconds": "4",
            },
        )
        if response.status_code != 200:
            raise AssertionError(f"isolate {model_name} {response.status_code}: {response.text}")
        payload = check_job_payload(client, response.json(), model_name)
        report[f"{model_name}_job_id"] = payload["job_id"]
        report[f"{model_name}_process_seconds"] = payload["timing"]["process_seconds"]

    if DEFAULT_MODEL not in MODEL_SPECS:
        raise AssertionError(f"default model {DEFAULT_MODEL} missing from MODEL_SPECS")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
