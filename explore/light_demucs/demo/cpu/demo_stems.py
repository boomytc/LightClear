import platform
import sys
import time
from pathlib import Path

import torch

MODULE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = MODULE_ROOT.parent.parent
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from demucs.api import Separator, save_audio

# 全局关键参数
MODEL_NAME = "htdemucs"
DEVICE = "cpu"
SHIFTS = 1
OVERLAP = 0.25
SEGMENT = 7
MODEL_REPO = MODULE_ROOT / "models" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "music" / "next_station_heaven.mp3"
OUTPUT_DIR = MODULE_ROOT / "outputs" / "cpu" / MODEL_NAME / INPUT_AUDIO.stem

if not MODEL_REPO.is_dir():
    raise FileNotFoundError(f"缺少模型权重目录: {MODEL_REPO}")
if not INPUT_AUDIO.is_file():
    raise FileNotFoundError(f"缺少测试音频: {INPUT_AUDIO}")

# 模型加载
t_load_start = time.perf_counter()
separator = Separator(
    model=MODEL_NAME,
    repo=MODEL_REPO,
    device=DEVICE,
    shifts=SHIFTS,
    overlap=OVERLAP,
    segment=SEGMENT,
    progress=False,
)
load_elapsed_s = time.perf_counter() - t_load_start

# 推理调用（CPU 单次推理）
t_infer_start = time.perf_counter()
origin, stems = separator.separate_audio_file(INPUT_AUDIO)
infer_elapsed_s = time.perf_counter() - t_infer_start

duration_s = origin.shape[-1] / separator.samplerate
rtf = infer_elapsed_s / duration_s

# 结果输出
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for stem_name, stem_wav in stems.items():
    save_audio(stem_wav.detach().cpu(), str(OUTPUT_DIR / f"{stem_name}.wav"), samplerate=separator.samplerate)

print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {DEVICE}")
print(f"模型: {MODEL_NAME} (stems={list(stems.keys())}, shifts={SHIFTS}, segment={SEGMENT}), 加载耗时: {load_elapsed_s:.3f} s")
print(f"输入音频: {INPUT_AUDIO.name} ({separator.samplerate} Hz, {duration_s:.3f} s)")
print(f"推理耗时: {infer_elapsed_s:.3f} s, RTF: {rtf:.4f}, 实时倍率: {duration_s / infer_elapsed_s:.2f}x")
print(f"已保存四轨目录: {OUTPUT_DIR}")

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | CPU）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.13.0, 设备: cpu
# 模型: htdemucs (stems=['drums', 'bass', 'other', 'vocals'], shifts=1, segment=7), 加载耗时: 0.120 s
# 输入音频: next_station_heaven.mp3 (44100 Hz, 30.041 s)
# 推理耗时: 8.141 s, RTF: 0.2710, 实时倍率: 3.69x
# 已保存四轨目录: /Users/boom/workspace/LightClear/explore/light_demucs/outputs/cpu/htdemucs/next_station_heaven
