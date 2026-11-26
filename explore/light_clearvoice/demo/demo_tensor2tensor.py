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
DEVICE = os.environ.get("LIGHTCLEAR_DEVICE", "auto")  # 可选: "auto" / "cuda" / "cuda:0" / "cpu"（不支持 MPS，指定 mps 将回退到 cpu）
TASK_NAME = "speech_enhancement"
MODEL_NAME = "FRCRN_SE_16K"
SAMPLE_RATE = 16000
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT") or (Path.home() / "Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SE" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "input.wav"
OUTPUT_BASE = MODULE_ROOT / "outputs"

# 设备解析（不支持 MPS：auto 优先级 cuda > cpu，CUDA 不可用或指定 mps 时回退 cpu）
requested_device = DEVICE.strip().lower()
if requested_device == "auto":
    resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    print(f"[警告] {MODEL_NAME} 在 MPS 上实测性能退化（复数循环反慢于 CPU），已自动回退到 cpu")
    resolved_device = "cpu"
elif requested_device == "cpu":
    resolved_device = "cpu"
else:
    raise ValueError(f"不支持的 DEVICE 配置: {DEVICE!r}，可选值为 auto / cuda / cuda:<idx> / cpu")

device_type = "cuda" if resolved_device.startswith("cuda") else resolved_device
cuda_index = int(resolved_device.split(":", 1)[1]) if device_type == "cuda" else None

OUTPUT_DIR = OUTPUT_BASE / device_type / "tensor2tensor"
OUTPUT_NUMPY_AUDIO = OUTPUT_DIR / "FRCRN_SE_16K_numpy.wav"
OUTPUT_TORCH_AUDIO = OUTPUT_DIR / "FRCRN_SE_16K_torch.wav"

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
load_elapsed_s = time.perf_counter() - t_load_start

# 输入构造
audio, sample_rate = sf.read(str(INPUT_AUDIO))
if sample_rate != SAMPLE_RATE:
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
if len(audio.shape) > 1:
    audio = audio[:, 0]
audio = np.reshape(audio, [1, audio.shape[0]]).astype(np.float32)
duration_s = audio.shape[-1] / SAMPLE_RATE
torch_audio = torch.from_numpy(audio.copy())

# 推理调用（CUDA 测冷启动+稳态，CPU 测单次）
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
    t_cold_start = time.perf_counter()
    _ = myClearVoice(audio, False)
    torch.cuda.synchronize(cuda_index)
    cold_elapsed_s = time.perf_counter() - t_cold_start

    t_np_start = time.perf_counter()
    output_numpy = myClearVoice(audio, False)
    torch.cuda.synchronize(cuda_index)
    np_elapsed_s = time.perf_counter() - t_np_start

    t_th_start = time.perf_counter()
    output_torch = myClearVoice(torch_audio, False)
    torch.cuda.synchronize(cuda_index)
    th_elapsed_s = time.perf_counter() - t_th_start
else:
    t_np_start = time.perf_counter()
    output_numpy = myClearVoice(audio, False)
    np_elapsed_s = time.perf_counter() - t_np_start

    t_th_start = time.perf_counter()
    output_torch = myClearVoice(torch_audio, False)
    th_elapsed_s = time.perf_counter() - t_th_start

if isinstance(output_torch, torch.Tensor):
    output_torch_np = output_torch.detach().cpu().numpy()
else:
    output_torch_np = output_torch

# 结果输出
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_NUMPY_AUDIO), output_numpy[0, :], SAMPLE_RATE)
sf.write(str(OUTPUT_TORCH_AUDIO), output_torch_np[0, :], SAMPLE_RATE)

max_diff = float(np.max(np.abs(output_numpy - output_torch_np)))
if device_type == "cuda":
    peak_alloc_mib = torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 2)
    print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {myClearVoice.models[0].device}")
else:
    print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {myClearVoice.models[0].device}")
print(f"任务: {TASK_NAME}, 模型: {MODEL_NAME}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"输入时长: {duration_s:.3f} s ({SAMPLE_RATE} Hz, shape={audio.shape})")
if device_type == "cuda":
    print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
    print(f"NumPy 稳态耗时: {np_elapsed_s:.3f} s, RTF: {np_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / np_elapsed_s:.2f}x")
    print(f"Torch 稳态耗时: {th_elapsed_s:.3f} s, RTF: {th_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / th_elapsed_s:.2f}x, max_diff={max_diff:.3e}")
    print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")
else:
    print(f"NumPy 推理耗时: {np_elapsed_s:.3f} s, RTF: {np_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / np_elapsed_s:.2f}x")
    print(f"Torch 推理耗时: {th_elapsed_s:.3f} s, RTF: {th_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / th_elapsed_s:.2f}x, max_diff={max_diff:.3e}")
print(f"已保存输出: {OUTPUT_NUMPY_AUDIO} / {OUTPUT_TORCH_AUDIO}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | CUDA）
# =============================================================================
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0
# 任务: speech_enhancement, 模型: FRCRN_SE_16K, 加载耗时: 2.466 s
# 输入时长: 2.398 s (16000 Hz, shape=(1, 38363))
# 冷启动推理耗时: 0.460 s, RTF: 0.1917, 实时倍率: 5.22x
# NumPy 稳态耗时: 0.112 s, RTF: 0.0469, 实时倍率: 21.32x
# Torch 稳态耗时: 0.106 s, RTF: 0.0441, 实时倍率: 22.67x, max_diff=4.746e-06
# 峰值显存: allocated=2703.1 MiB, reserved=2968.0 MiB
# 已保存输出: /content/LightClear/explore/light_clearvoice/outputs/cuda/tensor2tensor/FRCRN_SE_16K_numpy.wav / /content/LightClear/explore/light_clearvoice/outputs/cuda/tensor2tensor/FRCRN_SE_16K_torch.wav

# =============================================================================
# 实测输出（2026-11-26 | Colab CPU AMD EPYC 7B12 (2 vCPU) | Linux x86_64 | CPU）
# =============================================================================
# 平台: Linux x86_64, PyTorch: 2.15.0+cpu, 设备: cpu
# 任务: speech_enhancement, 模型: FRCRN_SE_16K, 加载耗时: 2.102 s
# 输入时长: 2.398 s (16000 Hz, shape=(1, 38363))
# NumPy 推理耗时: 11.728 s, RTF: 4.8916, 实时倍率: 0.20x
# Torch 推理耗时: 9.108 s, RTF: 3.7988, 实时倍率: 0.26x, max_diff=0.000e+00
# 已保存输出: /content/LightClear/explore/light_clearvoice/outputs/cpu/tensor2tensor/FRCRN_SE_16K_numpy.wav / /content/LightClear/explore/light_clearvoice/outputs/cpu/tensor2tensor/FRCRN_SE_16K_torch.wav
# 注：FRCRN_SE_16K 在 MPS 上实测性能退化（反慢于 CPU），本脚本不启用 MPS。
