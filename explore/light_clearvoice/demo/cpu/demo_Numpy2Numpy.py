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
MODEL_NAME = "MossFormer2_SE_48K"
SAMPLE_RATE = 48000
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SE" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "input.wav"
OUTPUT_AUDIO = MODULE_ROOT / "outputs" / "cpu" / "numpy2numpy" / "MossFormer2_SE_48K_batch.wav"

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
if len(audio.shape) < 2:
    audio = np.reshape(audio, [1, audio.shape[0]])
audio = audio.astype(np.float32)
duration_s = audio.shape[-1] / SAMPLE_RATE

# 推理调用（CPU 单次计时）
t_infer_start = time.perf_counter()
output_wav = myClearVoice(audio, False)
infer_elapsed_s = time.perf_counter() - t_infer_start

# 结果输出
OUTPUT_AUDIO.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(OUTPUT_AUDIO), output_wav[0, :], SAMPLE_RATE)

print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {myClearVoice.models[0].device}")
print(f"任务: {TASK_NAME}, 模型: {MODEL_NAME}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"NumPy 输入: shape={audio.shape}, sr={SAMPLE_RATE} Hz, 时长={duration_s:.3f} s -> 输出: {type(output_wav).__name__} {output_wav.shape}")
print(f"推理耗时: {infer_elapsed_s:.3f} s, RTF: {infer_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / infer_elapsed_s:.2f}x")
print(f"已保存输出: {OUTPUT_AUDIO}")

# =============================================================================
# 实测输出（2026-11-26 | Colab CPU AMD EPYC 7B12 (2 vCPU) | Linux-6.6.159+-x86_64-with-glibc2.39 | CPU）
# =============================================================================
# 平台: Linux x86_64, PyTorch: 2.15.0+cpu, 设备: cpu
# 任务: speech_enhancement, 模型: MossFormer2_SE_48K, 加载耗时: 3.117 s
# NumPy 输入: shape=(1, 115089), sr=48000 Hz, 时长=2.398 s -> 输出: ndarray (1, 115089)
# 推理耗时: 1.750 s, RTF: 0.7300, 实时倍率: 1.37x
# 已保存输出: /content/LightClear/explore/light_clearvoice/outputs/cpu/numpy2numpy/MossFormer2_SE_48K_batch.wav
