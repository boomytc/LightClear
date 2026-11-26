import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

MODULE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = MODULE_ROOT.parent.parent
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

# 全局关键参数
DEVICE = "cuda:0"
TASK_NAME = "speech_enhancement"
MODEL_NAME = "FRCRN_SE_16K"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SE" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "input.wav"
INPUT_DIR = REPO_ROOT / "assets" / "audio" / "noisy"
INPUT_SCP = REPO_ROOT / "assets" / "manifests" / "se.scp"
OUTPUT_ROOT = MODULE_ROOT / "outputs" / "cuda" / MODEL_NAME
OUTPUT_FILE = OUTPUT_ROOT / "input.wav"
OUTPUT_DIR = OUTPUT_ROOT / "path_to_output_wavs"
OUTPUT_SCP_DIR = OUTPUT_ROOT / "path_to_output_wavs_scp"

if not torch.cuda.is_available():
    raise RuntimeError("当前环境 CUDA 不可用，请在 NVIDIA GPU 环境运行 demo/cuda/ 脚本")

gpu_name = torch.cuda.get_device_name(0)
gpu_total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

# 模型加载
torch.cuda.reset_peak_memory_stats(0)
t_load_start = time.perf_counter()
myClearVoice = ClearVoice(
    task=TASK_NAME,
    model_names=[MODEL_NAME],
    device=DEVICE,
    checkpoint_dir=str(CHECKPOINT_DIR),
)
torch.cuda.synchronize(0)
load_elapsed_s = time.perf_counter() - t_load_start

# 单文件推理（冷启动 + 稳态二次计时）
audio_info = sf.info(str(INPUT_AUDIO))
duration_s = audio_info.duration
torch.cuda.synchronize(0)
t_cold_start = time.perf_counter()
_ = myClearVoice(input_path=str(INPUT_AUDIO), online_write=False)
torch.cuda.synchronize(0)
cold_elapsed_s = time.perf_counter() - t_cold_start

torch.cuda.synchronize(0)
t_warm_start = time.perf_counter()
output_wav = myClearVoice(input_path=str(INPUT_AUDIO), online_write=False)
torch.cuda.synchronize(0)
warm_elapsed_s = time.perf_counter() - t_warm_start
myClearVoice.write(output_wav, output_path=str(OUTPUT_FILE))

# 目录与 SCP 批处理推理
torch.cuda.synchronize(0)
t_dir_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_DIR), online_write=True, output_path=str(OUTPUT_DIR))
torch.cuda.synchronize(0)
dir_elapsed_s = time.perf_counter() - t_dir_start

torch.cuda.synchronize(0)
t_scp_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_SCP), online_write=True, output_path=str(OUTPUT_SCP_DIR))
torch.cuda.synchronize(0)
scp_elapsed_s = time.perf_counter() - t_scp_start

peak_alloc_mib = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
peak_reserved_mib = torch.cuda.max_memory_reserved(0) / (1024 ** 2)

# 结果输出
print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {myClearVoice.models[0].device}")
print(f"任务: {TASK_NAME}, 模型: {MODEL_NAME}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"单文件输入: {INPUT_AUDIO.name} ({audio_info.samplerate} Hz, {duration_s:.3f} s)")
print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
print(f"稳态推理耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
print(f"目录批处理耗时: {dir_elapsed_s:.3f} s, SCP 批处理耗时: {scp_elapsed_s:.3f} s")
print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")
print(f"单文件输出: {OUTPUT_FILE}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | Linux-6.6.159+-x86_64-with-glibc2.39）
# =============================================================================
# Running FRCRN_SE_16K ...
# Running FRCRN_SE_16K ...
# Running FRCRN_SE_16K ...
# Running FRCRN_SE_16K ...
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0
# 任务: speech_enhancement, 模型: FRCRN_SE_16K, 加载耗时: 2.442 s
# 单文件输入: input.wav (16000 Hz, 2.398 s)
# 冷启动推理耗时: 0.429 s, RTF: 0.1789, 实时倍率: 5.59x
# 稳态推理耗时: 0.103 s, RTF: 0.0428, 实时倍率: 23.35x
# 目录批处理耗时: 14.915 s, SCP 批处理耗时: 13.912 s
# 峰值显存: allocated=1341.7 MiB, reserved=3450.0 MiB
# 单文件输出: /content/LightClear/explore/light_clearvoice/outputs/cuda/FRCRN_SE_16K/input.wav
