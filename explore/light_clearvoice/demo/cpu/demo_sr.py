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
TASK_NAME = "speech_super_resolution"
MODEL_NAME = "MossFormer2_SR_48K"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SR" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "bandlimited" / "input_sr.wav"
INPUT_DIR = REPO_ROOT / "assets" / "audio" / "bandlimited"
INPUT_SCP = REPO_ROOT / "assets" / "manifests" / "sr.scp"
OUTPUT_ROOT = MODULE_ROOT / "outputs" / "cpu" / MODEL_NAME
OUTPUT_FILE = OUTPUT_ROOT / "input_sr.wav"
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
# Running MossFormer2_SR_48K ...
# Running MossFormer2_SR_48K ...
# Running MossFormer2_SR_48K ...
# 平台: Linux x86_64, PyTorch: 2.15.0+cpu, 设备: cpu
# 任务: speech_super_resolution, 模型: MossFormer2_SR_48K, 加载耗时: 6.860 s
# 单文件输入: input_sr.wav (16000 Hz, 2.398 s)
# 单文件推理耗时: 21.008 s, RTF: 8.7617, 实时倍率: 0.11x
# 目录批处理耗时: 646.069 s, SCP 批处理耗时: 642.254 s
# 单文件输出: /content/LightClear/explore/light_clearvoice/outputs/cpu/MossFormer2_SR_48K/input_sr.wav
