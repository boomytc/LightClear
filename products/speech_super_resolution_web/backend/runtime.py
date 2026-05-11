from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import sys
import time


AUDIO_EXTENSIONS = ("wav", "mp3", "flac", "ogg", "aac", "aiff", "m4a")
MODEL_NAME = "MossFormer2_SR_48K"
TASK_NAME = "speech_super_resolution"
DEFAULT_OUTPUT_DIR = "outputs/speech_super_resolution_web/super_resolved"
SAMPLE_AUDIO_FILES = ("assets/clearvoice_samples/input_sr.wav",)
SAMPLE_AUDIO_DIRS = ("assets/clearvoice_samples/path_to_input_wavs_sr",)


@dataclass
class ModelHandle:
    clearvoice: object
    initial_load_seconds: float


@dataclass
class SuperResolutionResult:
    input_path: Path
    output_path: Path
    model_initial_load_seconds: float
    model_ready_seconds: float
    process_seconds: float
    total_seconds: float


def bootstrap_project_paths(project_root: Path) -> None:
    third_party_dir = project_root / "third_party"
    for path in (project_root, third_party_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_mossformer2_sr(project_root: Path) -> ModelHandle:
    bootstrap_project_paths(project_root)
    from clearvoice import ClearVoice

    start_time = time.perf_counter()
    clearvoice = ClearVoice(task=TASK_NAME, model_names=[MODEL_NAME])
    return ModelHandle(
        clearvoice=clearvoice,
        initial_load_seconds=time.perf_counter() - start_time,
    )


def model_checkpoint_dir(project_root: Path) -> Path:
    return project_root / "models" / "sr_models" / MODEL_NAME


def model_is_available(project_root: Path) -> bool:
    checkpoint_dir = model_checkpoint_dir(project_root)
    best_checkpoint = checkpoint_dir / "last_best_checkpoint"
    if not checkpoint_dir.is_dir() or not best_checkpoint.is_file():
        return False

    checkpoint_names = [
        line.strip()
        for line in best_checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return bool(checkpoint_names) and all((checkpoint_dir / name).is_file() for name in checkpoint_names)


def list_sample_audio(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for sample_file in SAMPLE_AUDIO_FILES:
        path = project_root / sample_file
        if path.is_file() and path.suffix.lower().lstrip(".") in AUDIO_EXTENSIONS:
            files.append(path)
    for sample_dir in SAMPLE_AUDIO_DIRS:
        directory = project_root / sample_dir
        for extension in AUDIO_EXTENSIONS:
            files.extend(directory.glob(f"*.{extension}"))
    return sorted(files)


def resolve_project_output_dir(project_root: Path, raw_value: str) -> Path:
    value = raw_value.strip() or DEFAULT_OUTPUT_DIR
    output_dir = Path(value).expanduser()
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError("输出目录必须位于当前项目目录内。") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_filename(filename: str) -> str:
    path = Path(filename)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")
    suffix = path.suffix.lower()

    if suffix.replace(".", "") not in AUDIO_EXTENSIONS:
        suffix = ".wav"
    if not stem:
        stem = "audio"

    return f"{stem}{suffix}"


def write_uploaded_audio_bytes(data: bytes, upload_dir: Path, original_name: str) -> tuple[Path, str]:
    digest = hashlib.sha1(data).hexdigest()[:12]
    filename = safe_filename(original_name)
    path = Path(filename)
    upload_dir.mkdir(parents=True, exist_ok=True)

    upload_path = upload_dir / f"{path.stem}_{digest}{path.suffix}"
    if not upload_path.exists():
        upload_path.write_bytes(data)
    return upload_path, digest


def make_output_path(output_dir: Path, input_path: Path) -> Path:
    safe_name = safe_filename(input_path.name)
    path = Path(safe_name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{path.stem}_super_resolved_{timestamp}{path.suffix}"


def super_resolve_audio_file(
    model_handle: ModelHandle,
    input_path: Path,
    output_path: Path,
    model_ready_seconds: float,
    total_start_time: float,
) -> SuperResolutionResult:
    process_start = time.perf_counter()
    output_wav = model_handle.clearvoice(
        input_path=str(input_path),
        online_write=False,
    )
    model_handle.clearvoice.write(output_wav, output_path=str(output_path))
    process_seconds = time.perf_counter() - process_start

    return SuperResolutionResult(
        input_path=input_path,
        output_path=output_path,
        model_initial_load_seconds=model_handle.initial_load_seconds,
        model_ready_seconds=model_ready_seconds,
        process_seconds=process_seconds,
        total_seconds=time.perf_counter() - total_start_time,
    )


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
