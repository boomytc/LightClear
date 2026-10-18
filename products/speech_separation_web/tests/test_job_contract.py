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


SAMPLE_PATH = "assets/input_ss.wav"
SPEAKER_IDS = ("speaker-1", "speaker-2")
ANALYSIS_IMAGES = ("waveform_image", "power_spectrum_image", "mel_spectrum_image")


def check_analysis(payload: dict[str, object]) -> dict[str, object]:
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
    for key in ANALYSIS_IMAGES:
        value = analysis.get(key)
        if not isinstance(value, str) or not value.startswith("data:image/png;base64,"):
            raise AssertionError(f"{key} must be a png data uri, got {type(value).__name__}")
    return analysis


def check_job_payload(client: TestClient, payload: dict[str, object]) -> dict[str, object]:
    job_id = str(payload["job_id"])
    if not job_id:
        raise AssertionError(f"missing job_id: {payload}")

    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise AssertionError(f"outputs must be a list: {payload}")
    if [item.get("label") for item in outputs] != ["说话人 1", "说话人 2"]:
        raise AssertionError(f"separate must label both speakers: {outputs}")
    audio_urls = [str(item.get("audio_url", "")) for item in outputs]
    if audio_urls != [f"/api/jobs/{job_id}/audio/{speaker}" for speaker in SPEAKER_IDS]:
        raise AssertionError(f"speaker audio urls must follow speaker-N: {audio_urls}")
    for item in outputs:
        path = item.get("path")
        if not isinstance(path, str) or not (PRODUCT_ROOT / path).is_file():
            raise AssertionError(f"output path must exist on disk: {item}")

    input_rate = sf.info(str(PRODUCT_ROOT / payload["input_path"])).samplerate
    output_rates = [sf.info(str(PRODUCT_ROOT / item["path"])).samplerate for item in outputs]
    if any(rate != input_rate for rate in output_rates):
        raise AssertionError(
            f"separation keeps the input rate; analysis aligns by resampling ({input_rate} vs {output_rates})"
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

    check_analysis(payload)

    for variant in ("original", *SPEAKER_IDS):
        audio = client.get(f"/api/jobs/{job_id}/audio/{variant}")
        if audio.status_code != 200:
            raise AssertionError(f"audio {variant} {audio.status_code}: {audio.text}")
        if not audio.headers.get("content-type", "").startswith("audio/"):
            raise AssertionError(f"audio {variant} content-type {audio.headers.get('content-type')}")

    for variant in ("speaker-0", "speaker-3", "vocals"):
        unknown = client.get(f"/api/jobs/{job_id}/audio/{variant}")
        if unknown.status_code != 404:
            raise AssertionError(f"audio {variant} should 404, got {unknown.status_code}")

    for variant in SPEAKER_IDS:
        download = client.get(f"/api/jobs/{job_id}/download/{variant}")
        if download.status_code != 200:
            raise AssertionError(f"download {variant} {download.status_code}: {download.text}")
        if "attachment" not in download.headers.get("content-disposition", ""):
            raise AssertionError(f"download {variant} must be an attachment")

    ambiguous_download = client.get(f"/api/jobs/{job_id}/download")
    if ambiguous_download.status_code != 404:
        raise AssertionError(
            f"variant-less download must stay removed, got {ambiguous_download.status_code}"
        )

    missing_job = client.get(f"/api/jobs/{'0' * 32}/download/speaker-1")
    if missing_job.status_code != 404:
        raise AssertionError(f"unknown job should 404, got {missing_job.status_code}")

    return payload


def main() -> None:
    client = TestClient(app)
    report: dict[str, object] = {}

    health = client.get("/api/health")
    if health.status_code != 200:
        raise AssertionError(f"health {health.status_code}: {health.text}")
    model_available = bool(health.json().get("model_available"))
    report["model_available"] = model_available

    bad_source = client.post("/api/separate", data={"source_type": "bogus"})
    if bad_source.status_code != 400:
        raise AssertionError(f"bad source should 400, got {bad_source.status_code}: {bad_source.text}")
    report["rejects_bad_source"] = True

    escape = client.post(
        "/api/separate",
        data={"source_type": "sample", "sample_path": "../pyproject.toml"},
    )
    if escape.status_code != 400:
        raise AssertionError(f"path escape should 400, got {escape.status_code}: {escape.text}")
    report["rejects_path_escape"] = True

    if not model_available:
        print("SKIP: 模型权重未就绪，仅校验拒绝分支", file=sys.stderr)
        report["skipped"] = True
    else:
        response = client.post(
            "/api/separate",
            data={"source_type": "sample", "sample_path": SAMPLE_PATH, "waveform_seconds": "4"},
        )
        if response.status_code != 200:
            raise AssertionError(f"separate {response.status_code}: {response.text}")
        payload = check_job_payload(client, response.json())
        report["job_id"] = payload["job_id"]
        report["process_seconds"] = payload["timing"]["process_seconds"]

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
