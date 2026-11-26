import os
import platform
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent.parent
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

# 全局关键参数
DEVICE = os.environ.get("LIGHTCLEAR_DEVICE", "auto")  # 可选: "auto" / "cuda" / "cuda:0" / "mps" / "cpu"
TASK_NAME = "speech_enhancement"
MODEL_NAME = "MossFormer2_SE_48K"
SAMPLE_RATE = 48000
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SE" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "input.wav"
OUTPUT_BASE = MODULE_ROOT / "outputs"

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

OUTPUT_AUDIO = OUTPUT_BASE / device_type / "numpy2numpy" / "MossFormer2_SE_48K_batch.wav"

# 模型加载
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(cuda_index)
    gpu_total_gib = torch.cuda.get_device_properties(cuda_index).total_memory / (1024 ** 3)
    torch.cuda.reset_peak_memory_stats(cuda_index)

t_load_start = time.perf_counter()
myClearVoice = ClearVoice(
    task=TASK_NAME,
    model_names=[MODEL_NAME],
    device=resolved_device,
    checkpoint_dir=str(CHECKPOINT_DIR),
)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
load_elapsed_s = time.perf_counter() - t_load_start

# 输入构造
audio, sample_rate = sf.read(str(INPUT_AUDIO))
if sample_rate != SAMPLE_RATE:
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
if len(audio.shape) < 2:
    audio = np.reshape(audio, [1, audio.shape[0]])
audio = audio.astype(np.float32)
duration_s = audio.shape[-1] / SAMPLE_RATE

# 推理调用（CUDA/MPS 测冷启动+稳态，CPU 测单次）
if device_type in ("cuda", "mps"):
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t_cold_start = time.perf_counter()
    _ = myClearVoice(audio, False)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    cold_elapsed_s = time.perf_counter() - t_cold_start

    t_warm_start = time.perf_counter()
    output_wav = myClearVoice(audio, False)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    warm_elapsed_s = time.perf_counter() - t_warm_start
else:
    t_infer_start = time.perf_counter()
    output_wav = myClearVoice(audio, False)
    infer_elapsed_s = time.perf_counter() - t_infer_start

# 结果输出
OUTPUT_AUDIO.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_AUDIO), output_wav[0, :], SAMPLE_RATE)

if device_type == "cuda":
    peak_alloc_mib = torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 2)
    print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {myClearVoice.models[0].device}")
else:
    print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {myClearVoice.models[0].device}")
print(f"任务: {TASK_NAME}, 模型: {MODEL_NAME}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"NumPy 输入: shape={audio.shape}, sr={SAMPLE_RATE} Hz, 时长={duration_s:.3f} s -> 输出: {type(output_wav).__name__} {output_wav.shape}")
if device_type in ("cuda", "mps"):
    print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
    print(f"稳态推理耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
else:
    print(f"推理耗时: {infer_elapsed_s:.3f} s, RTF: {infer_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / infer_elapsed_s:.2f}x")
if device_type == "cuda":
    print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")
print(f"已保存输出: {OUTPUT_AUDIO}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | CUDA）
# =============================================================================
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0
# 任务: speech_enhancement, 模型: MossFormer2_SE_48K, 加载耗时: 2.475 s
# NumPy 输入: shape=(1, 115089), sr=48000 Hz, 时长=2.398 s -> 输出: ndarray (1, 115089)
# 冷启动推理耗时: 0.563 s, RTF: 0.2349, 实时倍率: 4.26x
# 稳态推理耗时: 0.181 s, RTF: 0.0756, 实时倍率: 13.23x
# 峰值显存: allocated=1046.4 MiB, reserved=1090.0 MiB
# 已保存输出: /content/LightClear/explore/light_clearvoice/outputs/cuda/numpy2numpy/MossFormer2_SE_48K_batch.wav

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.13.0, 设备: mps
# 任务: speech_enhancement, 模型: MossFormer2_SE_48K, 加载耗时: 2.832 s
# NumPy 输入: shape=(1, 115089), sr=48000 Hz, 时长=2.398 s -> 输出: ndarray (1, 115089)
# 冷启动推理耗时: 1.631 s, RTF: 0.6802, 实时倍率: 1.47x
# 稳态推理耗时: 0.218 s, RTF: 0.0911, 实时倍率: 10.97x
# 已保存输出: /Users/boom/workspace/LightClear/explore/light_clearvoice/outputs/mps/numpy2numpy/MossFormer2_SE_48K_batch.wav

# =============================================================================
# 实测输出（2026-11-26 | Colab CPU AMD EPYC 7B12 (2 vCPU) | Linux x86_64 | CPU）
# =============================================================================
# 平台: Linux x86_64, PyTorch: 2.15.0+cpu, 设备: cpu
# 任务: speech_enhancement, 模型: MossFormer2_SE_48K, 加载耗时: 3.117 s
# NumPy 输入: shape=(1, 115089), sr=48000 Hz, 时长=2.398 s -> 输出: ndarray (1, 115089)
# 推理耗时: 1.750 s, RTF: 0.7300, 实时倍率: 1.37x
# 已保存输出: /content/LightClear/explore/light_clearvoice/outputs/cpu/numpy2numpy/MossFormer2_SE_48K_batch.wav
