from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .analysis import build_analysis_payload
from .runtime import (
    AUDIO_EXTENSIONS,
    DEFAULT_OUTPUT_DIR,
    ModelHandle,
    SeparationResult,
    audio_mime_type,
    list_sample_audio,
    load_mossformer2_ss,
    make_output_path,
    model_checkpoint_dir,
    model_is_available,
    resolve_project_output_dir,
    safe_filename,
    separate_audio_file,
    write_uploaded_audio_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = APP_DIR / "frontend"
APP_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "speech_separation_web"
UPLOAD_DIR = APP_OUTPUT_ROOT / "uploads"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


@dataclass
class JobRecord:
    job_id: str
    input_path: Path
    output_paths: list[Path]
    created_at: str
    result: SeparationResult
    analysis: dict[str, object]


app = FastAPI(title="LightClear Speech Separation Web", version="1.0.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

_model_handle: ModelHandle | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_jobs: dict[str, JobRecord] = {}
_jobs_lock = threading.Lock()


def project_relative(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved_path)


def get_model_handle() -> ModelHandle:
    global _model_handle
    with _model_lock:
        if _model_handle is None:
            _model_handle = load_mossformer2_ss(PROJECT_ROOT)
        return _model_handle


def resolve_sample_path(sample_path: str | None) -> Path:
    if not sample_path:
        raise HTTPException(status_code=400, detail="请选择示例音频。")

    candidate = (PROJECT_ROOT / sample_path).resolve()
    sample_dir = (PROJECT_ROOT / "assets" / "clearvoice_samples").resolve()
    try:
        candidate.relative_to(sample_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="示例音频路径不在允许目录内。") from exc

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="示例音频不存在。")
    if candidate.suffix.lower().lstrip(".") not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="不支持的音频格式。")
    return candidate


def store_job(result: SeparationResult, analysis: dict[str, object]) -> JobRecord:
    job_id = uuid.uuid4().hex
    record = JobRecord(
        job_id=job_id,
        input_path=result.input_path,
        output_paths=result.output_paths,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        result=result,
        analysis=analysis,
    )
    with _jobs_lock:
        _jobs[job_id] = record
    return record


def get_job(job_id: str) -> JobRecord:
    with _jobs_lock:
        record = _jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="处理任务不存在或服务已重启。")
    return record


def output_variant_index(variant: str) -> int:
    if not variant.startswith("speaker-"):
        raise HTTPException(status_code=404, detail="音频版本不存在。")
    value = variant.removeprefix("speaker-")
    if not value.isdigit():
        raise HTTPException(status_code=404, detail="音频版本不存在。")
    return int(value) - 1


def output_payloads(record: JobRecord) -> list[dict[str, object]]:
    return [
        {
            "label": f"说话人 {index}",
            "name": path.name,
            "path": project_relative(path),
            "audio_url": f"/api/jobs/{record.job_id}/audio/speaker-{index}",
            "download_url": f"/api/jobs/{record.job_id}/download/speaker-{index}",
        }
        for index, path in enumerate(record.output_paths, start=1)
    ]


def build_result_payload(record: JobRecord) -> dict[str, object]:
    result = record.result
    outputs = output_payloads(record)
    first_output = outputs[0]
    return {
        "job_id": record.job_id,
        "created_at": record.created_at,
        "input_name": result.input_path.name,
        "output_name": f"{len(outputs)} 路分离音频",
        "input_path": project_relative(result.input_path),
        "output_path": first_output["path"],
        "input_audio_url": f"/api/jobs/{record.job_id}/audio/original",
        "output_audio_url": first_output["audio_url"],
        "download_url": first_output["download_url"],
        "outputs": outputs,
        "timing": {
            "model_initial_load_seconds": result.model_initial_load_seconds,
            "model_ready_seconds": result.model_ready_seconds,
            "process_seconds": result.process_seconds,
            "total_seconds": result.total_seconds,
        },
        "analysis": record.analysis,
        "logs": [
            f"输入: {project_relative(result.input_path)}",
            *[
                f"{item['label']}: {item['path']}"
                for item in outputs
            ],
            f"模型准备: {result.model_ready_seconds:.2f} 秒",
            f"音频处理: {result.process_seconds:.2f} 秒",
            f"总执行: {result.total_seconds:.2f} 秒",
        ],
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict[str, object]:
    checkpoint_dir = model_checkpoint_dir(PROJECT_ROOT)
    samples = list_sample_audio(PROJECT_ROOT)
    return {
        "app": "speech_separation_web",
        "model_name": "MossFormer2_SS_16K",
        "task": "speech_separation",
        "model_available": model_is_available(PROJECT_ROOT),
        "checkpoint_dir": project_relative(checkpoint_dir),
        "sample_count": len(samples),
        "default_output_dir": DEFAULT_OUTPUT_DIR,
        "upload_dir": project_relative(UPLOAD_DIR),
        "supported_extensions": list(AUDIO_EXTENSIONS),
    }


@app.get("/api/samples")
def samples() -> dict[str, object]:
    items = [
        {
            "name": path.name,
            "path": project_relative(path),
            "audio_url": f"/api/sample-audio?path={project_relative(path)}",
        }
        for path in list_sample_audio(PROJECT_ROOT)
    ]
    return {"samples": items}


@app.get("/api/sample-audio")
def sample_audio(path: str) -> FileResponse:
    sample_path = resolve_sample_path(path)
    return FileResponse(sample_path, media_type=audio_mime_type(sample_path))


@app.post("/api/separate")
async def separate(
    source_type: str = Form(...),
    sample_path: str | None = Form(default=None),
    output_dir: str = Form(default=DEFAULT_OUTPUT_DIR),
    waveform_seconds: float = Form(default=8.0),
    file: UploadFile | None = File(default=None),
) -> dict[str, object]:
    total_start = time.perf_counter()
    source = source_type.strip().lower()

    if source == "sample":
        input_path = resolve_sample_path(sample_path)
    elif source == "upload":
        if file is None or not file.filename:
            raise HTTPException(status_code=400, detail="请上传音频文件。")
        safe_name = safe_filename(file.filename)
        if Path(safe_name).suffix.lower().lstrip(".") not in AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="不支持的音频格式。")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="上传文件为空。")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="上传文件超过 200MB。")
        input_path, _ = write_uploaded_audio_bytes(data, UPLOAD_DIR, safe_name)
    else:
        raise HTTPException(status_code=400, detail="音频来源无效。")

    try:
        resolved_output_dir = resolve_project_output_dir(PROJECT_ROOT, output_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_path = make_output_path(resolved_output_dir, input_path)
    waveform_window = max(1.0, min(float(waveform_seconds), 30.0))

    try:
        model_ready_start = time.perf_counter()
        model_handle = get_model_handle()
        model_ready_seconds = time.perf_counter() - model_ready_start

        with _inference_lock:
            result = separate_audio_file(
                model_handle=model_handle,
                input_path=input_path,
                output_path=output_path,
                model_ready_seconds=model_ready_seconds,
                total_start_time=total_start,
            )

        analysis = build_analysis_payload(result.input_path, result.output_paths[0], waveform_window)
        record = store_job(result, analysis)
        return build_result_payload(record)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}") from exc


@app.get("/api/jobs/{job_id}/audio/{variant}")
def job_audio(job_id: str, variant: str) -> FileResponse:
    record = get_job(job_id)
    if variant == "original":
        path = record.input_path
    else:
        index = output_variant_index(variant)
        if index < 0 or index >= len(record.output_paths):
            raise HTTPException(status_code=404, detail="音频版本不存在。")
        path = record.output_paths[index]

    return FileResponse(path, media_type=audio_mime_type(path))


@app.get("/api/jobs/{job_id}/download")
def first_job_download(job_id: str) -> FileResponse:
    return job_download(job_id, "speaker-1")


@app.get("/api/jobs/{job_id}/download/{variant}")
def job_download(job_id: str, variant: str) -> FileResponse:
    record = get_job(job_id)
    index = output_variant_index(variant)
    if index < 0 or index >= len(record.output_paths):
        raise HTTPException(status_code=404, detail="音频版本不存在。")
    output_path = record.output_paths[index]
    return FileResponse(
        output_path,
        media_type=audio_mime_type(output_path),
        filename=output_path.name,
    )
