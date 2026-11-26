import os
import platform
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
DEVICE = "cpu"
TASK_NAME = "speech_enhancement"
MODEL_NAME = "FRCRN_SE_16K"
SAMPLE_RATE = 16000
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SE" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "input.wav"
OUTPUT_NUMPY_AUDIO = MODULE_ROOT / "outputs" / "cpu" / "tensor2tensor" / "FRCRN_SE_16K_numpy.wav"
OUTPUT_TORCH_AUDIO = MODULE_ROOT / "outputs" / "cpu" / "tensor2tensor" / "FRCRN_SE_16K_torch.wav"

# 模型加载
t_load_start = time.perf_counter()
myClearVoice = ClearVoice(
    task=TASK_NAME,
    model_names=[MODEL_NAME],
    device=DEVICE,
    checkpoint_dir=str(CHECKPOINT_DIR),
)
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

# 推理调用（CPU 单次计时）
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
OUTPUT_NUMPY_AUDIO.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_NUMPY_AUDIO), output_numpy[0, :], SAMPLE_RATE)
sf.write(str(OUTPUT_TORCH_AUDIO), output_torch_np[0, :], SAMPLE_RATE)

max_diff = float(np.max(np.abs(output_numpy - output_torch_np)))
print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {myClearVoice.models[0].device}")
print(f"任务: {TASK_NAME}, 模型: {MODEL_NAME}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"输入时长: {duration_s:.3f} s ({SAMPLE_RATE} Hz, shape={audio.shape})")
print(f"NumPy 推理耗时: {np_elapsed_s:.3f} s, RTF: {np_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / np_elapsed_s:.2f}x")
print(f"Torch 推理耗时: {th_elapsed_s:.3f} s, RTF: {th_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / th_elapsed_s:.2f}x, max_diff={max_diff:.3e}")
print(f"已保存输出: {OUTPUT_NUMPY_AUDIO} / {OUTPUT_TORCH_AUDIO}")

# =============================================================================
# 实测输出（2026-11-26 | Colab CPU AMD EPYC 7B12 (2 vCPU) | Linux-6.6.159+-x86_64-with-glibc2.39 | CPU）
# =============================================================================
# 平台: Linux x86_64, PyTorch: 2.15.0+cpu, 设备: cpu
# 任务: speech_enhancement, 模型: FRCRN_SE_16K, 加载耗时: 2.102 s
# 输入时长: 2.398 s (16000 Hz, shape=(1, 38363))
# NumPy 推理耗时: 11.728 s, RTF: 4.8916, 实时倍率: 0.20x
# Torch 推理耗时: 9.108 s, RTF: 3.7988, 实时倍率: 0.26x, max_diff=0.000e+00
# 已保存输出: /content/LightClear/explore/light_clearvoice/outputs/cpu/tensor2tensor/FRCRN_SE_16K_numpy.wav / /content/LightClear/explore/light_clearvoice/outputs/cpu/tensor2tensor/FRCRN_SE_16K_torch.wav
