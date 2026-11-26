import os
import platform
import sys
import time
from pathlib import Path

import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent.parent
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from demucs.api import Separator, save_audio

# 全局关键参数
DEVICE = os.environ.get("LIGHTCLEAR_DEVICE", "auto")  # 可选: "auto" / "cuda" / "cuda:0" / "mps" / "cpu"
MODEL_NAME = "htdemucs"
STEM_NAME = "vocals"
SHIFTS = 1
OVERLAP = 0.25
SEGMENT = 7
MODEL_REPO = MODULE_ROOT / "models" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "music" / "next_station_heaven.mp3"
OUTPUT_BASE = MODULE_ROOT / "outputs"

if not MODEL_REPO.is_dir():
    raise FileNotFoundError(f"缺少模型权重目录: {MODEL_REPO}")
if not INPUT_AUDIO.is_file():
    raise FileNotFoundError(f"缺少测试音频: {INPUT_AUDIO}")

# 设备解析（支持 MPS：auto 优先级 cuda > mps > cpu，目标加速器不可用时回退 cpu）
requested_device = DEVICE.strip().lower()
if requested_device == "auto":
    if torch.cuda.is_available():
        resolved_device = "cuda:0"
    elif torch.backends.mps.is_available():
        resolved_device = "mps"
    else:
        resolved_device = "cpu"
elif requested_device == "cuda" or requested_device.startswith("cuda:"):
    if requested_device == "cuda":
        req_cuda_idx = 0
    else:
        idx_text = requested_device.split(":", 1)[1]
        if not idx_text.isdigit():
            raise ValueError(f"无效的 CUDA 设备索引: {DEVICE!r}")
        req_cuda_idx = int(idx_text)
    if torch.cuda.is_available() and req_cuda_idx < torch.cuda.device_count():
        resolved_device = f"cuda:{req_cuda_idx}"
    else:
        print(f"[警告] 请求设备 {DEVICE!r} 不可用，已自动回退到 cpu")
        resolved_device = "cpu"
elif requested_device == "mps":
    if torch.backends.mps.is_available():
        resolved_device = "mps"
    else:
        print(f"[警告] 请求设备 {DEVICE!r} 不可用，已自动回退到 cpu")
        resolved_device = "cpu"
elif requested_device == "cpu":
    resolved_device = "cpu"
else:
    raise ValueError(f"不支持的 DEVICE 配置: {DEVICE!r}，可选值为 auto / cuda / cuda:<idx> / mps / cpu")

device_type = "cuda" if resolved_device.startswith("cuda") else resolved_device
cuda_index = int(resolved_device.split(":", 1)[1]) if device_type == "cuda" else None

OUTPUT_DIR = OUTPUT_BASE / device_type / MODEL_NAME
OUTPUT_VOCALS = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_{STEM_NAME}.wav"
OUTPUT_ACCOMPANIMENT = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_no_{STEM_NAME}.wav"

# 模型加载
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(cuda_index)
    gpu_total_gib = torch.cuda.get_device_properties(cuda_index).total_memory / (1024 ** 3)
    torch.cuda.reset_peak_memory_stats(cuda_index)

t_load_start = time.perf_counter()
separator = Separator(
    model=MODEL_NAME,
    repo=MODEL_REPO,
    device=resolved_device,
    shifts=SHIFTS,
    overlap=OVERLAP,
    segment=SEGMENT,
    progress=False,
)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
load_elapsed_s = time.perf_counter() - t_load_start

# 推理调用（CUDA/MPS 测冷启动+稳态，CPU 测单次）
if device_type in ("cuda", "mps"):
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t_cold_start = time.perf_counter()
    origin, _ = separator.separate_audio_file(INPUT_AUDIO)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    cold_elapsed_s = time.perf_counter() - t_cold_start

    t_warm_start = time.perf_counter()
    origin, stems = separator.separate_audio_file(INPUT_AUDIO)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    warm_elapsed_s = time.perf_counter() - t_warm_start
else:
    t_infer_start = time.perf_counter()
    origin, stems = separator.separate_audio_file(INPUT_AUDIO)
    infer_elapsed_s = time.perf_counter() - t_infer_start

duration_s = origin.shape[-1] / separator.samplerate
vocals = stems[STEM_NAME].detach().cpu()
accompaniment = (stems["drums"] + stems["bass"] + stems["other"]).detach().cpu()

# 结果输出
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_audio(vocals, str(OUTPUT_VOCALS), samplerate=separator.samplerate)
save_audio(accompaniment, str(OUTPUT_ACCOMPANIMENT), samplerate=separator.samplerate)

if device_type == "cuda":
    peak_alloc_mib = torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 2)
    print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {resolved_device}")
else:
    print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {resolved_device}")
print(f"模型: {MODEL_NAME} (shifts={SHIFTS}, overlap={OVERLAP}, segment={SEGMENT}), 加载耗时: {load_elapsed_s:.3f} s")
print(f"输入音频: {INPUT_AUDIO.name} ({separator.samplerate} Hz, {duration_s:.3f} s)")
if device_type in ("cuda", "mps"):
    print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
    print(f"稳态推理耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
else:
    print(f"推理耗时: {infer_elapsed_s:.3f} s, RTF: {infer_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / infer_elapsed_s:.2f}x")
if device_type == "cuda":
    print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")
print(f"已保存人声: {OUTPUT_VOCALS}")
print(f"已保存伴奏: {OUTPUT_ACCOMPANIMENT}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | CUDA）
# =============================================================================
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0
# 模型: htdemucs (shifts=1, overlap=0.25, segment=7), 加载耗时: 0.533 s
# 输入音频: next_station_heaven.mp3 (44100 Hz, 30.041 s)
# 冷启动推理耗时: 1.699 s, RTF: 0.0566, 实时倍率: 17.68x
# 稳态推理耗时: 0.890 s, RTF: 0.0296, 实时倍率: 33.74x
# 峰值显存: allocated=546.8 MiB, reserved=770.0 MiB
# 已保存人声: /content/LightClear/explore/light_demucs/outputs/cuda/htdemucs/next_station_heaven_vocals.wav
# 已保存伴奏: /content/LightClear/explore/light_demucs/outputs/cuda/htdemucs/next_station_heaven_no_vocals.wav

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.13.0, 设备: mps
# 模型: htdemucs (shifts=1, overlap=0.25, segment=7), 加载耗时: 0.117 s
# 输入音频: next_station_heaven.mp3 (44100 Hz, 30.041 s)
# 冷启动推理耗时: 3.256 s, RTF: 0.1084, 实时倍率: 9.23x
# 稳态推理耗时: 2.427 s, RTF: 0.0808, 实时倍率: 12.38x
# 已保存人声: /Users/boom/workspace/LightClear/explore/light_demucs/outputs/mps/htdemucs/next_station_heaven_vocals.wav
# 已保存伴奏: /Users/boom/workspace/LightClear/explore/light_demucs/outputs/mps/htdemucs/next_station_heaven_no_vocals.wav

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | CPU）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.13.0, 设备: cpu
# 模型: htdemucs (shifts=1, overlap=0.25, segment=7), 加载耗时: 0.152 s
# 输入音频: next_station_heaven.mp3 (44100 Hz, 30.041 s)
# 推理耗时: 7.643 s, RTF: 0.2544, 实时倍率: 3.93x
# 已保存人声: /Users/boom/workspace/LightClear/explore/light_demucs/outputs/cpu/htdemucs/next_station_heaven_vocals.wav
# 已保存伴奏: /Users/boom/workspace/LightClear/explore/light_demucs/outputs/cpu/htdemucs/next_station_heaven_no_vocals.wav
