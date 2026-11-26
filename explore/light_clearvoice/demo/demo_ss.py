import os
import platform
import sys
import time
from pathlib import Path

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
TASK_NAME = "speech_separation"
MODEL_NAME = "MossFormer2_SS_16K"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT") or (Path.home() / "Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SS" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "mixture" / "input_ss.wav"
INPUT_DIR = REPO_ROOT / "assets" / "audio" / "mixture"
INPUT_SCP = REPO_ROOT / "assets" / "manifests" / "ss.scp"
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
if device_type == "cpu":
    print(f"[警告] 当前运行在 CPU，{MODEL_NAME} 在 CPU 上单文件实测约 39 s（RTF ~8.9），生产建议使用 CUDA 或 MPS")

OUTPUT_ROOT = OUTPUT_BASE / device_type / MODEL_NAME
OUTPUT_FILE = OUTPUT_ROOT / "input_ss.wav"
OUTPUT_DIR = OUTPUT_ROOT / "path_to_output_wavs"
OUTPUT_SCP_DIR = OUTPUT_ROOT / "path_to_output_wavs_scp"

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

# 单文件推理（CUDA/MPS 测冷启动+稳态，CPU 测单次）
audio_info = sf.info(str(INPUT_AUDIO))
duration_s = audio_info.duration
if device_type in ("cuda", "mps"):
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t_cold_start = time.perf_counter()
    _ = myClearVoice(input_path=str(INPUT_AUDIO), online_write=False)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    cold_elapsed_s = time.perf_counter() - t_cold_start

    t_warm_start = time.perf_counter()
    output_wav = myClearVoice(input_path=str(INPUT_AUDIO), online_write=False)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    warm_elapsed_s = time.perf_counter() - t_warm_start
else:
    t_infer_start = time.perf_counter()
    output_wav = myClearVoice(input_path=str(INPUT_AUDIO), online_write=False)
    infer_elapsed_s = time.perf_counter() - t_infer_start
myClearVoice.write(output_wav, output_path=str(OUTPUT_FILE))

# 目录与 SCP 批处理推理
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t_dir_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_DIR), online_write=True, output_path=str(OUTPUT_DIR))
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
dir_elapsed_s = time.perf_counter() - t_dir_start

t_scp_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_SCP), online_write=True, output_path=str(OUTPUT_SCP_DIR))
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
scp_elapsed_s = time.perf_counter() - t_scp_start

# 结果输出
if device_type == "cuda":
    peak_alloc_mib = torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 2)
    print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {myClearVoice.models[0].device}")
else:
    print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {myClearVoice.models[0].device}")
print(f"任务: {TASK_NAME}, 模型: {MODEL_NAME}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"单文件输入: {INPUT_AUDIO.name} ({audio_info.samplerate} Hz, {duration_s:.3f} s)")
if device_type in ("cuda", "mps"):
    print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
    print(f"稳态推理耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
else:
    print(f"单文件推理耗时: {infer_elapsed_s:.3f} s, RTF: {infer_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / infer_elapsed_s:.2f}x")
print(f"目录批处理耗时: {dir_elapsed_s:.3f} s, SCP 批处理耗时: {scp_elapsed_s:.3f} s")
if device_type == "cuda":
    print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")
print(f"单文件输出前缀: {OUTPUT_FILE}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | CUDA）
# =============================================================================
# Running MossFormer2_SS_16K ...
# Running MossFormer2_SS_16K ...
# Running MossFormer2_SS_16K ...
# Running MossFormer2_SS_16K ...
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0
# 任务: speech_separation, 模型: MossFormer2_SS_16K, 加载耗时: 3.151 s
# 单文件输入: input_ss.wav (16000 Hz, 4.409 s)
# 冷启动推理耗时: 1.283 s, RTF: 0.2910, 实时倍率: 3.44x
# 稳态推理耗时: 0.897 s, RTF: 0.2034, 实时倍率: 4.92x
# 目录批处理耗时: 11.413 s, SCP 批处理耗时: 11.319 s
# 峰值显存: allocated=461.4 MiB, reserved=488.0 MiB
# 单文件输出前缀: /content/LightClear/explore/light_clearvoice/outputs/cuda/MossFormer2_SS_16K/input_ss.wav

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# Running MossFormer2_SS_16K ...
# Running MossFormer2_SS_16K ...
# Running MossFormer2_SS_16K ...
# Running MossFormer2_SS_16K ...
# 平台: Darwin arm64, PyTorch: 2.13.0, 设备: mps
# 任务: speech_separation, 模型: MossFormer2_SS_16K, 加载耗时: 1.621 s
# 单文件输入: input_ss.wav (16000 Hz, 4.409 s)
# 冷启动推理耗时: 2.454 s, RTF: 0.5567, 实时倍率: 1.80x
# 稳态推理耗时: 1.806 s, RTF: 0.4096, 实时倍率: 2.44x
# 目录批处理耗时: 20.265 s, SCP 批处理耗时: 20.492 s
# 单文件输出前缀: /Users/boom/workspace/LightClear/explore/light_clearvoice/outputs/mps/MossFormer2_SS_16K/input_ss.wav
# 注：Apple M4 Pro CPU 单文件实测 39.12 s（RTF 8.9），不建议生产在 CPU 运行。
