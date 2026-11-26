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
DEVICE = "mps"
TASK_NAME = "speech_super_resolution"
MODEL_NAME = "MossFormer2_SR_48K"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SR" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "bandlimited" / "input_sr.wav"
INPUT_DIR = REPO_ROOT / "assets" / "audio" / "bandlimited"
INPUT_SCP = REPO_ROOT / "assets" / "manifests" / "sr.scp"
OUTPUT_ROOT = MODULE_ROOT / "outputs" / "mps" / MODEL_NAME
OUTPUT_FILE = OUTPUT_ROOT / "input_sr.wav"
OUTPUT_DIR = OUTPUT_ROOT / "path_to_output_wavs"
OUTPUT_SCP_DIR = OUTPUT_ROOT / "path_to_output_wavs_scp"

if not torch.backends.mps.is_available():
    raise RuntimeError("当前环境 MPS 不可用，请在 Apple Silicon macOS 上运行 demo/mps/ 脚本")

# 模型加载
t_load_start = time.perf_counter()
myClearVoice = ClearVoice(
    task=TASK_NAME,
    model_names=[MODEL_NAME],
    device=DEVICE,
    checkpoint_dir=str(CHECKPOINT_DIR),
)
torch.mps.synchronize()
load_elapsed_s = time.perf_counter() - t_load_start

# 单文件推理（冷启动 + 稳态二次计时）
audio_info = sf.info(str(INPUT_AUDIO))
duration_s = audio_info.duration
torch.mps.synchronize()
t_cold_start = time.perf_counter()
_ = myClearVoice(input_path=str(INPUT_AUDIO), online_write=False)
torch.mps.synchronize()
cold_elapsed_s = time.perf_counter() - t_cold_start

torch.mps.synchronize()
t_warm_start = time.perf_counter()
output_wav = myClearVoice(input_path=str(INPUT_AUDIO), online_write=False)
torch.mps.synchronize()
warm_elapsed_s = time.perf_counter() - t_warm_start
myClearVoice.write(output_wav, output_path=str(OUTPUT_FILE))

# 目录与 SCP 批处理推理
torch.mps.synchronize()
t_dir_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_DIR), online_write=True, output_path=str(OUTPUT_DIR))
torch.mps.synchronize()
dir_elapsed_s = time.perf_counter() - t_dir_start

torch.mps.synchronize()
t_scp_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_SCP), online_write=True, output_path=str(OUTPUT_SCP_DIR))
torch.mps.synchronize()
scp_elapsed_s = time.perf_counter() - t_scp_start

# 结果输出
print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {myClearVoice.models[0].device}")
print(f"任务: {TASK_NAME}, 模型: {MODEL_NAME}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"单文件输入: {INPUT_AUDIO.name} ({audio_info.samplerate} Hz, {duration_s:.3f} s)")
print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
print(f"稳态推理耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
print(f"目录批处理耗时: {dir_elapsed_s:.3f} s, SCP 批处理耗时: {scp_elapsed_s:.3f} s")
print(f"单文件输出: {OUTPUT_FILE}")

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# Running MossFormer2_SR_48K ...
# Running MossFormer2_SR_48K ...
# Running MossFormer2_SR_48K ...
# Running MossFormer2_SR_48K ...
# 平台: Darwin arm64, PyTorch: 2.13.0, 设备: mps
# 任务: speech_super_resolution, 模型: MossFormer2_SR_48K, 加载耗时: 3.820 s
# 单文件输入: input_sr.wav (16000 Hz, 2.398 s)
# 冷启动推理耗时: 1.304 s, RTF: 0.5437, 实时倍率: 1.84x
# 稳态推理耗时: 0.525 s, RTF: 0.2190, 实时倍率: 4.57x
# 目录批处理耗时: 16.426 s, SCP 批处理耗时: 13.652 s
# 单文件输出: /Users/boom/workspace/LightClear/explore/light_clearvoice/outputs/mps/MossFormer2_SR_48K/input_sr.wav
