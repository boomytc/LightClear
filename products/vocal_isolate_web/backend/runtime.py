from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import sys
import time


AUDIO_EXTENSIONS = ("wav", "mp3", "flac", "ogg", "aac", "aiff", "m4a")
DEFAULT_MODEL = "htdemucs"
TASK_NAME = "vocal_isolation"
DEFAULT_OUTPUT_DIR = "workspace/outputs"
SAMPLE_AUDIO_FILES = ("assets/next_station_heaven.mp3",)
DEVICE = "cpu"
SHIFTS = 1
OVERLAP = 0.25
SEGMENT = 7
FOUR_STEM_OUTPUTS = ("vocals", "accompaniment", "drums", "bass", "other")
SIX_STEM_OUTPUTS = ("vocals", "accompaniment", "drums", "bass", "other", "guitar", "piano")
MODEL_SPECS = {
    "htdemucs": {
        "label": "htdemucs 默认四轨",
        "outputs": FOUR_STEM_OUTPUTS,
        "summary": "人声 / 伴奏 / 四轨",
        "note": "默认人声隔离。一次推理得到人声、伴奏和 drums / bass / other。",
    },
    "htdemucs_ft": {
        "label": "htdemucs_ft 四轨质量档",
        "outputs": FOUR_STEM_OUTPUTS,
        "summary": "人声 / 伴奏 / 四轨",
        "note": "同一四轨合同的质量袋，大约四倍耗时。",
    },
    "htdemucs_6s": {
        "label": "htdemucs_6s 六轨",
        "outputs": SIX_STEM_OUTPUTS,
        "summary": "人声 / 伴奏 / 六轨",
        "note": "另拆吉他和钢琴。人声仍取 vocals，伴奏为其余五轨相加。",
    },
}
KNOWN_MODELS = tuple(MODEL_SPECS)


@dataclass
class ModelHandle:
    model_name: str
    separator: object
    initial_load_seconds: float


@dataclass
class IsolationResult:
    input_path: Path
    output_paths: dict[str, Path]
    model_name: str
    model_initial_load_seconds: float
    model_ready_seconds: float
    process_seconds: float
    total_seconds: float
    samplerate: int = 44100


def bootstrap_product_paths(product_root: Path) -> None:
    third_party_dir = str(product_root / "third_party")
    if third_party_dir not in sys.path:
        sys.path.insert(0, third_party_dir)


def resolve_model_name(raw_value: str | None) -> str:
    name = (raw_value or DEFAULT_MODEL).strip()
    if name not in MODEL_SPECS:
        known = "、".join(KNOWN_MODELS)
        raise ValueError(f"不支持的模型: {name}。可选 {known}。")
    return name


def model_checkpoint_dir(product_root: Path, model_name: str = DEFAULT_MODEL) -> Path:
    return product_root / "models" / resolve_model_name(model_name)


def model_is_available(product_root: Path, model_name: str = DEFAULT_MODEL) -> bool:
    name = resolve_model_name(model_name)
    repo = model_checkpoint_dir(product_root, name)
    yaml_path = repo / f"{name}.yaml"
    if not yaml_path.is_file():
        return False
    return any(path.suffix == ".th" for path in repo.glob("*.th"))


def list_model_status(product_root: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for name, spec in MODEL_SPECS.items():
        items.append(
            {
                "name": name,
                "label": spec["label"],
                "available": model_is_available(product_root, name),
                "checkpoint_dir": f"models/{name}",
                "outputs": list(spec["outputs"]),
                "summary": spec["summary"],
                "note": spec["note"],
            }
        )
    return items


def load_separator(product_root: Path, model_name: str = DEFAULT_MODEL) -> ModelHandle:
    bootstrap_product_paths(product_root)
    name = resolve_model_name(model_name)
    repo = model_checkpoint_dir(product_root, name)
    if not model_is_available(product_root, name):
        raise FileNotFoundError(f"产品模型目录未就绪: {repo}")
    from demucs.api import Separator

    start_time = time.perf_counter()
    separator = Separator(
        model=name,
        repo=repo,
        device=DEVICE,
        shifts=SHIFTS,
        overlap=OVERLAP,
        segment=SEGMENT,
        progress=True,
    )
    return ModelHandle(
        model_name=name,
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


def make_job_output_dir(output_dir: Path, input_path: Path, model_name: str) -> Path:
    safe_name = safe_filename(input_path.name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_dir = output_dir / f"{Path(safe_name).stem}_{model_name}_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def accompaniment_from_stems(stems: dict[str, object]) -> object:
    extras = [wave for name, wave in stems.items() if name != "vocals"]
    if not extras:
        raise ValueError("分轨结果缺少伴奏声部。")
    accompaniment = extras[0]
    for wave in extras[1:]:
        accompaniment = accompaniment + wave
    return accompaniment


def isolate_audio_file(
    model_handle: ModelHandle,
    input_path: Path,
    output_dir: Path,
    model_ready_seconds: float,
    total_start_time: float,
) -> IsolationResult:
    from demucs.api import save_audio

    spec = MODEL_SPECS[model_handle.model_name]
    process_start = time.perf_counter()
    _origin, stems = model_handle.separator.separate_audio_file(input_path)
    if "vocals" not in stems:
        raise FileNotFoundError("分轨结果缺少 vocals")

    waves = {
        "vocals": stems["vocals"],
        "accompaniment": accompaniment_from_stems(stems),
    }
    for name in spec["outputs"]:
        if name in {"vocals", "accompaniment"}:
            continue
        if name not in stems:
            raise FileNotFoundError(f"分轨结果缺少 {name}")
        waves[name] = stems[name]

    job_dir = make_job_output_dir(output_dir, input_path, model_handle.model_name)
    output_paths: dict[str, Path] = {}
    samplerate = int(model_handle.separator.samplerate)
    for name in spec["outputs"]:
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
        model_name=model_handle.model_name,
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
