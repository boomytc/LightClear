from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import sys
import time


AUDIO_EXTENSIONS = ("wav", "mp3", "flac", "ogg", "aac", "aiff", "m4a")
MODEL_NAME = "htdemucs"
TASK_NAME = "vocal_isolation"
DEFAULT_OUTPUT_DIR = "workspace/outputs"
MSS_ROOT = Path("/Users/boom/Model/MSS")
MODEL_REPO = MSS_ROOT / MODEL_NAME
SAMPLE_AUDIO_FILES = ("assets/next_station_heaven.mp3",)
STEM_NAMES = ("vocals", "accompaniment", "drums", "bass", "other")
DEVICE = "cpu"
SHIFTS = 1
OVERLAP = 0.25
SEGMENT = 7


@dataclass
class ModelHandle:
    separator: object
    initial_load_seconds: float


@dataclass
class IsolationResult:
    input_path: Path
    output_paths: dict[str, Path]
    model_initial_load_seconds: float
    model_ready_seconds: float
    process_seconds: float
    total_seconds: float
    samplerate: int = 44100


def bootstrap_product_paths(product_root: Path) -> None:
    third_party_dir = str(product_root / "third_party")
    if third_party_dir not in sys.path:
        sys.path.insert(0, third_party_dir)


def model_checkpoint_dir(product_root: Path) -> Path:
    return MODEL_REPO


def model_is_available(product_root: Path) -> bool:
    yaml_path = MODEL_REPO / f"{MODEL_NAME}.yaml"
    if not yaml_path.is_file():
        return False
    return any(path.suffix == ".th" for path in MODEL_REPO.glob("*.th"))


def load_htdemucs(product_root: Path) -> ModelHandle:
    bootstrap_product_paths(product_root)
    if not model_is_available(product_root):
        raise FileNotFoundError(
            f"中心模型目录未就绪: {MODEL_REPO}。先运行 explore/light_demucs/scripts/install_htdemucs.py"
        )
    from demucs.api import Separator

    start_time = time.perf_counter()
    separator = Separator(
        model=MODEL_NAME,
        repo=MODEL_REPO,
        device=DEVICE,
        shifts=SHIFTS,
        overlap=OVERLAP,
        segment=SEGMENT,
        progress=True,
    )
    return ModelHandle(
        separator=separator,
        initial_load_seconds=time.perf_counter() - start_time,
    )


def list_sample_audio(product_root: Path) -> list[Path]:
    files: list[Path] = []
    for sample_file in SAMPLE_AUDIO_FILES:
        path = product_root / sample_file
        if path.is_file() and path.suffix.lower().lstrip(".") in AUDIO_EXTENSIONS:
            files.append(path)
    return files


def resolve_project_output_dir(product_root: Path, raw_value: str) -> Path:
    value = raw_value.strip() or DEFAULT_OUTPUT_DIR
    output_dir = Path(value).expanduser()
    if not output_dir.is_absolute():
        output_dir = product_root / output_dir

    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(product_root.resolve())
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


def make_job_output_dir(output_dir: Path, input_path: Path) -> Path:
    safe_name = safe_filename(input_path.name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_dir = output_dir / f"{Path(safe_name).stem}_isolated_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def isolate_audio_file(
    model_handle: ModelHandle,
    input_path: Path,
    output_dir: Path,
    model_ready_seconds: float,
    total_start_time: float,
) -> IsolationResult:
    from demucs.api import save_audio

    process_start = time.perf_counter()
    _origin, stems = model_handle.separator.separate_audio_file(input_path)
    vocals = stems["vocals"]
    accompaniment = stems["drums"] + stems["bass"] + stems["other"]
    waves = {
        "vocals": vocals,
        "accompaniment": accompaniment,
        "drums": stems["drums"],
        "bass": stems["bass"],
        "other": stems["other"],
    }
    job_dir = make_job_output_dir(output_dir, input_path)
    output_paths: dict[str, Path] = {}
    samplerate = int(model_handle.separator.samplerate)
    for name in STEM_NAMES:
        path = job_dir / f"{name}.wav"
        save_audio(waves[name], str(path), samplerate=samplerate)
        output_paths[name] = path
    process_seconds = time.perf_counter() - process_start

    missing = [name for name, path in output_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"隔离输出缺失: {', '.join(missing)}")

    return IsolationResult(
        input_path=input_path,
        output_paths=output_paths,
        model_initial_load_seconds=model_handle.initial_load_seconds,
        model_ready_seconds=model_ready_seconds,
        process_seconds=process_seconds,
        total_seconds=time.perf_counter() - total_start_time,
        samplerate=samplerate,
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
