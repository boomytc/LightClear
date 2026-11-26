import os
import platform
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
DEVICE = "cpu"
TASK_NAME = "speech_enhancement"
MODEL_NAME = "FRCRN_SE_16K"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SE" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "input.wav"
INPUT_DIR = REPO_ROOT / "assets" / "audio" / "noisy"
INPUT_SCP = REPO_ROOT / "assets" / "manifests" / "se.scp"
OUTPUT_ROOT = MODULE_ROOT / "outputs" / "cpu" / MODEL_NAME
OUTPUT_FILE = OUTPUT_ROOT / "input.wav"
OUTPUT_DIR = OUTPUT_ROOT / "path_to_output_wavs"
OUTPUT_SCP_DIR = OUTPUT_ROOT / "path_to_output_wavs_scp"

# 模型加载
t_load_start = time.perf_counter()
myClearVoice = ClearVoice(
    task=TASK_NAME,
    model_names=[MODEL_NAME],
    device=DEVICE,
    checkpoint_dir=str(CHECKPOINT_DIR),
)
load_elapsed_s = time.perf_counter() - t_load_start

# 单文件推理（CPU 单次计时）
audio_info = sf.info(str(INPUT_AUDIO))
duration_s = audio_info.duration
t_infer_start = time.perf_counter()
output_wav = myClearVoice(input_path=str(INPUT_AUDIO), online_write=False)
infer_elapsed_s = time.perf_counter() - t_infer_start
myClearVoice.write(output_wav, output_path=str(OUTPUT_FILE))

# 目录与 SCP 批处理推理
t_dir_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_DIR), online_write=True, output_path=str(OUTPUT_DIR))
dir_elapsed_s = time.perf_counter() - t_dir_start

t_scp_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_SCP), online_write=True, output_path=str(OUTPUT_SCP_DIR))
scp_elapsed_s = time.perf_counter() - t_scp_start

# 结果输出
print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {myClearVoice.models[0].device}")
print(f"任务: {TASK_NAME}, 模型: {MODEL_NAME}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"单文件输入: {INPUT_AUDIO.name} ({audio_info.samplerate} Hz, {duration_s:.3f} s)")
print(f"单文件推理耗时: {infer_elapsed_s:.3f} s, RTF: {infer_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / infer_elapsed_s:.2f}x")
print(f"目录批处理耗时: {dir_elapsed_s:.3f} s, SCP 批处理耗时: {scp_elapsed_s:.3f} s")
print(f"单文件输出: {OUTPUT_FILE}")

# =============================================================================
# 实测输出（2026-11-26 | Colab CPU AMD EPYC 7B12 (2 vCPU) | Linux-6.6.159+-x86_64-with-glibc2.39 | CPU）
# =============================================================================
# Running FRCRN_SE_16K ...
# Running FRCRN_SE_16K ...
# Running FRCRN_SE_16K ...
# 平台: Linux x86_64, PyTorch: 2.15.0+cpu, 设备: cpu
# 任务: speech_enhancement, 模型: FRCRN_SE_16K, 加载耗时: 2.513 s
# 单文件输入: input.wav (16000 Hz, 2.398 s)
# 单文件推理耗时: 7.981 s, RTF: 3.3285, 实时倍率: 0.30x
# 目录批处理耗时: 772.419 s, SCP 批处理耗时: 764.783 s
# 单文件输出: /content/LightClear/explore/light_clearvoice/outputs/cpu/FRCRN_SE_16K/input.wav
