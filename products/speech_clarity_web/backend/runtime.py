from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys
import time

import soundfile as sf


AUDIO_EXTENSIONS = ("wav", "mp3", "flac", "ogg", "aac", "aiff", "m4a")
SCENE_NAME = "speech_clarity"
APP_NAME = "speech_clarity_web"
TASKS_DIR = "workspace/tasks"

TOOL_SPECS = {
    "enhance": {
        "task": "speech_enhancement",
        "model": "MossFormer2_SE_48K",
        "model_root": Path("/Users/boom/Model/SE"),
        "outputs": ("enhanced",),
        "label": "增强",
    },
    "separate": {
        "task": "speech_separation",
        "model": "MossFormer2_SS_16K",
        "model_root": Path("/Users/boom/Model/SS"),
        "outputs": ("speaker-1", "speaker-2"),
        "label": "分离",
    },
    "super_resolve": {
        "task": "speech_super_resolution",
        "model": "MossFormer2_SR_48K",
        "model_root": Path("/Users/boom/Model/SR"),
        "outputs": ("super_resolved",),
        "label": "超分",
    },
}
TOOL_IDS = tuple(TOOL_SPECS)
SAMPLE_AUDIO = (
    {
        "file": "assets/noisy_input.wav",
        "kind": "noisy",
        "suggested_tools": ["enhance"],
    },
    {
        "file": "assets/mixture_input.wav",
        "kind": "mixture",
        "suggested_tools": ["separate"],
    },
    {
        "file": "assets/bandlimited_input.wav",
        "kind": "bandlimited",
        "suggested_tools": ["super_resolve"],
    },
)


@dataclass
class ModelHandle:
    tool_id: str
    clearvoice: object
    initial_load_seconds: float


@dataclass
class ToolRunResult:
    tool_id: str
    input_path: Path
    output_paths: dict[str, Path]
    model_initial_load_seconds: float
    model_ready_seconds: float
    process_seconds: float
    total_seconds: float
    input_sample_rate: int
    output_sample_rate: int


def bootstrap_product_paths(product_root: Path) -> None:
    third_party_dir = str(product_root / "third_party")
    if third_party_dir not in sys.path:
        sys.path.insert(0, third_party_dir)


def resolve_tool_id(raw_value: str | None) -> str:
    tool_id = (raw_value or "").strip()
    if tool_id not in TOOL_SPECS:
        known = "、".join(TOOL_IDS)
        raise ValueError(f"不支持的工具: {tool_id or '(空)'}。可选 {known}。")
    return tool_id


def model_checkpoint_dir(tool_id: str) -> Path:
    spec = TOOL_SPECS[tool_id]
    return spec["model_root"] / spec["model"]


def tool_is_available(tool_id: str) -> bool:
    checkpoint_dir = model_checkpoint_dir(tool_id)
    best_checkpoint = checkpoint_dir / "last_best_checkpoint"
    if not checkpoint_dir.is_dir() or not best_checkpoint.is_file():
        return False
    checkpoint_names = [
        line.strip()
        for line in best_checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return bool(checkpoint_names) and all((checkpoint_dir / name).is_file() for name in checkpoint_names)


def list_tool_status() -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for tool_id, spec in TOOL_SPECS.items():
        checkpoint_dir = model_checkpoint_dir(tool_id)
        items.append(
            {
                "id": tool_id,
                "task": spec["task"],
                "model": spec["model"],
                "available": tool_is_available(tool_id),
                "outputs": list(spec["outputs"]),
                "checkpoint_dir": str(checkpoint_dir),
                "label": spec["label"],
            }
        )
    return items


def list_sample_audio(product_root: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for sample in SAMPLE_AUDIO:
        path = product_root / sample["file"]
        if path.is_file() and path.suffix.lower().lstrip(".") in AUDIO_EXTENSIONS:
            items.append(
                {
                    "name": path.name,
                    "path": sample["file"],
                    "kind": sample["kind"],
                    "suggested_tools": list(sample["suggested_tools"]),
                    "full_path": path,
                }
            )
    return items


def safe_filename(filename: str) -> str:
    path = Path(filename)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    suffix = path.suffix.lower()
    if suffix.replace(".", "") not in AUDIO_EXTENSIONS:
        suffix = ".wav"
    if not stem:
        stem = "audio"
    return f"{stem}{suffix}"


def audio_mime_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "mp3":
        return "audio/mp3"
    if suffix == "ogg":
        return "audio/ogg"
    if suffix in {"aac", "m4a"}:
        return "audio/aac"
    if suffix == "flac":
        return "audio/flac"
    return "audio/wav"


def probe_sample_rate(path: Path) -> int:
    return int(sf.info(str(path)).samplerate)


def load_tool(product_root: Path, tool_id: str) -> ModelHandle:
    spec = TOOL_SPECS[tool_id]
    bootstrap_product_paths(product_root)
    from clearvoice import ClearVoice

    start_time = time.perf_counter()
    clearvoice = ClearVoice(task=spec["task"], model_names=[spec["model"]])
    return ModelHandle(
        tool_id=tool_id,
        clearvoice=clearvoice,
        initial_load_seconds=time.perf_counter() - start_time,
    )


def _write_clearvoice(handle: ModelHandle, input_path: Path, output_path: Path) -> None:
    output_wav = handle.clearvoice(
        input_path=str(input_path),
        online_write=False,
    )
    handle.clearvoice.write(output_wav, output_path=str(output_path))


def run_tool(
    handle: ModelHandle,
    tool_id: str,
    input_path: Path,
    task_dir: Path,
    run_id: str,
    model_ready_seconds: float,
    total_start_time: float,
) -> ToolRunResult:
    spec = TOOL_SPECS[tool_id]
    task_dir.mkdir(parents=True, exist_ok=True)
    input_sample_rate = probe_sample_rate(input_path)
    process_start = time.perf_counter()

    if tool_id == "separate":
        base_path = task_dir / f"{run_id}.wav"
        _write_clearvoice(handle, input_path, base_path)
        output_paths: dict[str, Path] = {}
        for index, output_id in enumerate(spec["outputs"], start=1):
            generated = base_path.with_name(f"{base_path.stem}_s{index}{base_path.suffix}")
            if not generated.is_file():
                raise FileNotFoundError(f"分离输出缺失: {generated.name}")
            final_path = task_dir / f"{run_id}_{output_id}.wav"
            shutil.move(str(generated), str(final_path))
            output_paths[output_id] = final_path
        if base_path.exists():
            base_path.unlink()
    else:
        output_id = spec["outputs"][0]
        output_path = task_dir / f"{run_id}_{output_id}.wav"
        _write_clearvoice(handle, input_path, output_path)
        if not output_path.is_file():
            raise FileNotFoundError(f"工具输出缺失: {output_path.name}")
        output_paths = {output_id: output_path}

    first_output = next(iter(output_paths.values()))
    output_sample_rate = probe_sample_rate(first_output)
    process_seconds = time.perf_counter() - process_start
    return ToolRunResult(
        tool_id=tool_id,
        input_path=input_path,
        output_paths=output_paths,
        model_initial_load_seconds=handle.initial_load_seconds,
        model_ready_seconds=model_ready_seconds,
        process_seconds=process_seconds,
        total_seconds=time.perf_counter() - total_start_time,
        input_sample_rate=input_sample_rate,
        output_sample_rate=output_sample_rate,
    )
