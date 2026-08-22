from __future__ import annotations

from pathlib import Path
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .analysis import build_analysis_payload
from .runtime import (
    APP_NAME,
    AUDIO_EXTENSIONS,
    SCENE_NAME,
    TASKS_DIR,
    TOOL_SPECS,
    ModelHandle,
    audio_mime_type,
    list_sample_audio,
    list_tool_status,
    load_tool,
    resolve_tool_id,
    run_tool,
    safe_filename,
    tool_is_available,
)
from .tasks import (
    append_step,
    create_task,
    find_step,
    list_tasks,
    load_task,
    project_relative,
    resolve_artifact,
    resolve_input_ref,
    save_task,
    task_dir,
    write_upload,
)


PRODUCT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PRODUCT_ROOT / "frontend"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

app = FastAPI(title="LightClear Speech Clarity Web", version="1.0.0")
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

_model_handles: dict[str, ModelHandle] = {}
_model_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_model_handle(tool_id: str) -> ModelHandle:
    with _model_lock:
        handle = _model_handles.get(tool_id)
        if handle is None:
            handle = load_tool(PRODUCT_ROOT, tool_id)
            _model_handles[tool_id] = handle
        return handle


def resolve_sample_path(sample_path: str | None) -> Path:
    if not sample_path:
        raise HTTPException(status_code=400, detail="请选择示例音频。")
    candidate = (PRODUCT_ROOT / sample_path).resolve()
    sample_dir = (PRODUCT_ROOT / "assets").resolve()
    try:
        candidate.relative_to(sample_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="示例音频路径不在允许目录内。") from exc
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="示例音频不存在。")
    if candidate.suffix.lower().lstrip(".") not in AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="不支持的音频格式。")
    return candidate


def require_task(task_id: str) -> dict[str, object]:
    try:
        return load_task(PRODUCT_ROOT, task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def output_payloads(task_id: str, run_id: str, outputs: list[dict[str, object]]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for item in outputs:
        output_id = str(item["id"])
        artifact_id = f"{run_id}_{output_id}"
        items.append(
            {
                "id": output_id,
                "name": item.get("name"),
                "path": item.get("path"),
                "audio_url": f"/api/tasks/{task_id}/audio/{artifact_id}",
                "download_url": f"/api/tasks/{task_id}/download/{artifact_id}",
            }
        )
    return items


def decorate_task(payload: dict[str, object]) -> dict[str, object]:
    task_id = str(payload["task_id"])
    steps = []
    for step in payload.get("steps", []):
        run_id = str(step["run_id"])
        steps.append(
            {
                **step,
                "outputs": output_payloads(task_id, run_id, list(step.get("outputs", []))),
            }
        )
    return {
        **payload,
        "input_audio_url": f"/api/tasks/{task_id}/audio/input",
        "input_path": payload["input"]["path"],
        "steps": steps,
    }


def build_run_payload(
    payload: dict[str, object],
    run_id: str,
    analysis: dict[str, object],
    timing: dict[str, float],
    logs: list[str],
) -> dict[str, object]:
    step = find_step(payload, run_id)
    task_id = str(payload["task_id"])
    outputs = output_payloads(task_id, run_id, list(step.get("outputs", [])))
    return {
        "task_id": task_id,
        "run_id": run_id,
        "tool": step["tool"],
        "input_ref": step["input_ref"],
        "status": step["status"],
        "input_sample_rate": step.get("input_sample_rate"),
        "output_sample_rate": step.get("output_sample_rate"),
        "outputs": outputs,
        "timing": timing,
        "analysis": analysis,
        "logs": logs,
        "task": decorate_task(payload),
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
def health() -> dict[str, object]:
    samples = list_sample_audio(PRODUCT_ROOT)
    return {
        "app": APP_NAME,
        "scene": SCENE_NAME,
        "sample_count": len(samples),
        "tasks_dir": TASKS_DIR,
        "supported_extensions": list(AUDIO_EXTENSIONS),
        "tools": list_tool_status(),
    }


@app.get("/api/tools")
def tools() -> dict[str, object]:
    return {"tools": list_tool_status()}


@app.get("/api/samples")
def samples() -> dict[str, object]:
    items = []
    for sample in list_sample_audio(PRODUCT_ROOT):
        relative = str(sample["path"])
        items.append(
            {
                "name": sample["name"],
                "path": relative,
                "kind": sample["kind"],
                "suggested_tools": sample["suggested_tools"],
                "audio_url": f"/api/sample-audio?path={relative}",
            }
        )
    return {"samples": items}


@app.get("/api/sample-audio")
def sample_audio(path: str) -> FileResponse:
    sample_path = resolve_sample_path(path)
    return FileResponse(sample_path, media_type=audio_mime_type(sample_path))


@app.post("/api/tasks")
async def create_task_endpoint(
    source_type: str = Form(...),
    sample_path: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
) -> dict[str, object]:
    source = source_type.strip().lower()
    if source == "sample":
        input_path = resolve_sample_path(sample_path)
        title = input_path.name
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
        input_path = write_upload(PRODUCT_ROOT, data, safe_name)
        title = file.filename
    else:
        raise HTTPException(status_code=400, detail="音频来源无效。")

    payload = create_task(PRODUCT_ROOT, input_path, title)
    return decorate_task(payload)


@app.get("/api/tasks")
def get_tasks() -> dict[str, object]:
    return {"tasks": list_tasks(PRODUCT_ROOT)}


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, object]:
    return decorate_task(require_task(task_id))


@app.post("/api/tasks/{task_id}/runs")
def create_run(
    task_id: str,
    tool: str = Form(...),
    input_ref: str = Form(...),
    waveform_seconds: float = Form(default=8.0),
) -> dict[str, object]:
    total_start = time.perf_counter()
    payload = require_task(task_id)
    try:
        tool_id = resolve_tool_id(tool)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        source_path = resolve_input_ref(PRODUCT_ROOT, payload, input_ref)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if not tool_is_available(tool_id):
        model_name = TOOL_SPECS[tool_id]["model"]
        raise HTTPException(status_code=503, detail=f"工具 {tool_id} 权重未就绪（{model_name}）。")

    waveform_window = max(1.0, min(float(waveform_seconds), 30.0))
    run_id = uuid.uuid4().hex
    directory = task_dir(PRODUCT_ROOT, task_id)
    payload["status"] = "running"
    save_task(PRODUCT_ROOT, payload)

    try:
        model_ready_start = time.perf_counter()
        handle = get_model_handle(tool_id)
        model_ready_seconds = time.perf_counter() - model_ready_start
        with _inference_lock:
            result = run_tool(
                handle=handle,
                tool_id=tool_id,
                input_path=source_path,
                task_dir=directory,
                run_id=run_id,
                model_ready_seconds=model_ready_seconds,
                total_start_time=total_start,
            )
        analysis = build_analysis_payload(
            tool_id,
            result.input_path,
            result.output_paths,
            waveform_window,
        )
        append_step(
            PRODUCT_ROOT,
            payload,
            run_id=run_id,
            tool_id=tool_id,
            input_ref=input_ref,
            output_paths=result.output_paths,
            input_sample_rate=result.input_sample_rate,
            output_sample_rate=result.output_sample_rate,
            status="succeeded",
        )
        timing = {
            "model_initial_load_seconds": result.model_initial_load_seconds,
            "model_ready_seconds": result.model_ready_seconds,
            "process_seconds": result.process_seconds,
            "total_seconds": result.total_seconds,
        }
        logs = [
            f"工具: {tool_id}",
            f"输入: {project_relative(PRODUCT_ROOT, result.input_path)}",
            *[
                f"{output_id}: {project_relative(PRODUCT_ROOT, path)}"
                for output_id, path in result.output_paths.items()
            ],
            f"输入采样率: {result.input_sample_rate} Hz",
            f"输出采样率: {result.output_sample_rate} Hz",
            f"模型准备: {result.model_ready_seconds:.2f} 秒",
            f"音频处理: {result.process_seconds:.2f} 秒",
        ]
        return build_run_payload(payload, run_id, analysis, timing, logs)
    except HTTPException:
        raise
    except Exception as exc:
        payload["status"] = "failed" if not payload.get("steps") else "succeeded"
        save_task(PRODUCT_ROOT, payload)
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}") from exc


@app.get("/api/tasks/{task_id}/runs/{run_id}")
def get_run(task_id: str, run_id: str, waveform_seconds: float = 8.0) -> dict[str, object]:
    payload = require_task(task_id)
    try:
        step = find_step(payload, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        source_path = resolve_input_ref(PRODUCT_ROOT, payload, str(step["input_ref"]))
        output_paths = {
            str(item["id"]): PRODUCT_ROOT / str(item["path"])
            for item in step.get("outputs", [])
        }
        analysis = build_analysis_payload(
            str(step["tool"]),
            source_path,
            output_paths,
            max(1.0, min(float(waveform_seconds), 30.0)),
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return build_run_payload(
        payload,
        run_id,
        analysis,
        {
            "model_initial_load_seconds": 0.0,
            "model_ready_seconds": 0.0,
            "process_seconds": 0.0,
            "total_seconds": 0.0,
        },
        [],
    )


@app.get("/api/tasks/{task_id}/audio/{artifact_id}")
def task_audio(task_id: str, artifact_id: str) -> FileResponse:
    payload = require_task(task_id)
    try:
        path = resolve_artifact(PRODUCT_ROOT, payload, artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type=audio_mime_type(path))


@app.get("/api/tasks/{task_id}/download/{artifact_id}")
def task_download(task_id: str, artifact_id: str) -> FileResponse:
    payload = require_task(task_id)
    try:
        path = resolve_artifact(PRODUCT_ROOT, payload, artifact_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, media_type=audio_mime_type(path), filename=path.name)
