import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
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
MODEL_NAME = "MossFormer2_SE_48K"
SAMPLE_RATE = 48000
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SE" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "input.wav"
OUTPUT_AUDIO = MODULE_ROOT / "outputs" / "cuda" / "numpy2numpy" / "MossFormer2_SE_48K_batch.wav"

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

# 输入构造
audio, sample_rate = sf.read(str(INPUT_AUDIO))
if sample_rate != SAMPLE_RATE:
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
if len(audio.shape) < 2:
    audio = np.reshape(audio, [1, audio.shape[0]])
audio = audio.astype(np.float32)
duration_s = audio.shape[-1] / SAMPLE_RATE

# 推理调用（冷启动 + 稳态二次计时）
torch.cuda.synchronize(0)
t_cold_start = time.perf_counter()
_ = myClearVoice(audio, False)
torch.cuda.synchronize(0)
cold_elapsed_s = time.perf_counter() - t_cold_start

torch.cuda.synchronize(0)
t_warm_start = time.perf_counter()
output_wav = myClearVoice(audio, False)
torch.cuda.synchronize(0)
warm_elapsed_s = time.perf_counter() - t_warm_start

peak_alloc_mib = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
peak_reserved_mib = torch.cuda.max_memory_reserved(0) / (1024 ** 2)

# 结果输出
OUTPUT_AUDIO.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_AUDIO), output_wav[0, :], SAMPLE_RATE)

print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {myClearVoice.models[0].device}")
print(f"任务: {TASK_NAME}, 模型: {MODEL_NAME}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"NumPy 输入: shape={audio.shape}, sr={SAMPLE_RATE} Hz, 时长={duration_s:.3f} s -> 输出: {type(output_wav).__name__} {output_wav.shape}")
print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
print(f"稳态推理耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")
print(f"已保存输出: {OUTPUT_AUDIO}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | Linux-6.6.159+-x86_64-with-glibc2.39）
# =============================================================================
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0
# 任务: speech_enhancement, 模型: MossFormer2_SE_48K, 加载耗时: 2.475 s
# NumPy 输入: shape=(1, 115089), sr=48000 Hz, 时长=2.398 s -> 输出: ndarray (1, 115089)
# 冷启动推理耗时: 0.563 s, RTF: 0.2349, 实时倍率: 4.26x
# 稳态推理耗时: 0.181 s, RTF: 0.0756, 实时倍率: 13.23x
# 峰值显存: allocated=1046.4 MiB, reserved=1090.0 MiB
# 已保存输出: /content/LightClear/explore/light_clearvoice/outputs/cuda/numpy2numpy/MossFormer2_SE_48K_batch.wav
