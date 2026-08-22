from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import time
import uuid

from .runtime import TASKS_DIR, safe_filename


def tasks_root(product_root: Path) -> Path:
    root = product_root / TASKS_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def task_dir(product_root: Path, task_id: str) -> Path:
    return tasks_root(product_root) / task_id


def task_json_path(product_root: Path, task_id: str) -> Path:
    return task_dir(product_root, task_id) / "task.json"


def project_relative(product_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(product_root.resolve()))
    except ValueError:
        return str(resolved)


def load_task(product_root: Path, task_id: str) -> dict[str, object]:
    path = task_json_path(product_root, task_id)
    if not path.is_file():
        raise FileNotFoundError(f"任务不存在: {task_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_task(product_root: Path, payload: dict[str, object]) -> None:
    task_id = str(payload["task_id"])
    path = task_json_path(product_root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_upload(product_root: Path, data: bytes, original_name: str) -> Path:
    digest = hashlib.sha1(data).hexdigest()[:12]
    filename = safe_filename(original_name)
    suffix = Path(filename).suffix
    upload_dir = product_root / "workspace" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{Path(filename).stem}_{digest}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return path


def list_tasks(product_root: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    root = tasks_root(product_root)
    for path in root.glob("*/task.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task_id = str(payload.get("task_id") or path.parent.name)
        steps = payload.get("steps") or []
        items.append(
            {
                "task_id": task_id,
                "title": payload.get("title") or task_id,
                "created_at": payload.get("created_at") or "",
                "status": payload.get("status") or "ready",
                "step_count": len(steps) if isinstance(steps, list) else 0,
            }
        )
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return items[:50]


def create_task(product_root: Path, source_path: Path, title: str) -> dict[str, object]:
    task_id = uuid.uuid4().hex
    directory = task_dir(product_root, task_id)
    directory.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(source_path.name)
    suffix = Path(filename).suffix
    input_path = directory / f"input{suffix}"
    shutil.copy2(source_path, input_path)
    payload = {
        "task_id": task_id,
        "title": title,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ready",
        "input": {
            "id": "input",
            "name": source_path.name,
            "path": project_relative(product_root, input_path),
        },
        "steps": [],
    }
    save_task(product_root, payload)
    return payload


def parse_input_ref(input_ref: str) -> tuple[str, str | None, str | None]:
    value = (input_ref or "").strip()
    if value == "input":
        return "input", None, None
    if value.startswith("step:"):
        rest = value[5:]
        if ":" not in rest:
            raise ValueError("input_ref 格式无效。")
        run_id, output_id = rest.split(":", 1)
        if not run_id or not output_id:
            raise ValueError("input_ref 格式无效。")
        return "step", run_id, output_id
    raise ValueError("input_ref 必须是 input 或 step:<run_id>:<output_id>。")


def resolve_input_ref(product_root: Path, payload: dict[str, object], input_ref: str) -> Path:
    kind, run_id, output_id = parse_input_ref(input_ref)
    if kind == "input":
        path = product_root / str(payload["input"]["path"])
        if not path.is_file():
            raise FileNotFoundError("任务输入音频不存在。")
        return path

    for step in payload.get("steps", []):
        if step.get("run_id") != run_id:
            continue
        for item in step.get("outputs", []):
            if item.get("id") == output_id:
                path = product_root / str(item["path"])
                if not path.is_file():
                    raise FileNotFoundError(f"步骤产物不存在: {output_id}")
                return path
        raise ValueError(f"步骤 {run_id} 没有产物 {output_id}。")
    raise ValueError(f"找不到步骤 {run_id}。")


def resolve_artifact(product_root: Path, payload: dict[str, object], artifact_id: str) -> Path:
    value = (artifact_id or "").strip()
    if value == "input":
        path = product_root / str(payload["input"]["path"])
        if not path.is_file():
            raise FileNotFoundError("任务输入音频不存在。")
        return path

    for step in payload.get("steps", []):
        run_id = str(step.get("run_id", ""))
        prefix = f"{run_id}_"
        if not value.startswith(prefix):
            continue
        output_id = value[len(prefix):]
        for item in step.get("outputs", []):
            if item.get("id") == output_id:
                path = product_root / str(item["path"])
                if not path.is_file():
                    raise FileNotFoundError(f"产物不存在: {artifact_id}")
                return path
        raise FileNotFoundError(f"产物不存在: {artifact_id}")
    raise FileNotFoundError(f"产物不存在: {artifact_id}")


def find_step(payload: dict[str, object], run_id: str) -> dict[str, object]:
    for step in payload.get("steps", []):
        if step.get("run_id") == run_id:
            return step
    raise FileNotFoundError(f"找不到步骤 {run_id}。")


def append_step(
    product_root: Path,
    payload: dict[str, object],
    *,
    run_id: str,
    tool_id: str,
    input_ref: str,
    output_paths: dict[str, Path],
    input_sample_rate: int,
    output_sample_rate: int,
    status: str,
) -> dict[str, object]:
    step = {
        "run_id": run_id,
        "tool": tool_id,
        "input_ref": input_ref,
        "status": status,
        "input_sample_rate": input_sample_rate,
        "output_sample_rate": output_sample_rate,
        "outputs": [
            {
                "id": output_id,
                "path": project_relative(product_root, path),
                "name": path.name,
            }
            for output_id, path in output_paths.items()
        ],
    }
    payload["steps"].append(step)
    payload["status"] = "succeeded" if status == "succeeded" else "failed"
    save_task(product_root, payload)
    return step
