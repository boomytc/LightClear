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
DEVICE = "mps"
SHIFTS = 1
OVERLAP = 0.25
SEGMENT = 7
MODEL_REPO = MODULE_ROOT / "models" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "music" / "next_station_heaven.mp3"
OUTPUT_DIR = MODULE_ROOT / "outputs" / "mps" / MODEL_NAME / INPUT_AUDIO.stem

if not torch.backends.mps.is_available():
    raise RuntimeError("当前环境 MPS 不可用，请在 Apple Silicon macOS 上运行 demo/mps/ 脚本")
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
torch.mps.synchronize()
load_elapsed_s = time.perf_counter() - t_load_start

# 推理调用（冷启动 + 稳态二次计时）
torch.mps.synchronize()
t_cold_start = time.perf_counter()
origin, _ = separator.separate_audio_file(INPUT_AUDIO)
torch.mps.synchronize()
cold_elapsed_s = time.perf_counter() - t_cold_start

torch.mps.synchronize()
t_warm_start = time.perf_counter()
origin, stems = separator.separate_audio_file(INPUT_AUDIO)
torch.mps.synchronize()
warm_elapsed_s = time.perf_counter() - t_warm_start

duration_s = origin.shape[-1] / separator.samplerate

# 结果输出
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for stem_name, stem_wav in stems.items():
    save_audio(stem_wav.detach().cpu(), str(OUTPUT_DIR / f"{stem_name}.wav"), samplerate=separator.samplerate)

print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {DEVICE}")
print(f"模型: {MODEL_NAME} (stems={list(stems.keys())}, shifts={SHIFTS}, segment={SEGMENT}), 加载耗时: {load_elapsed_s:.3f} s")
print(f"输入音频: {INPUT_AUDIO.name} ({separator.samplerate} Hz, {duration_s:.3f} s)")
print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
print(f"稳态推理耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
print(f"已保存四轨目录: {OUTPUT_DIR}")

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.13.0, 设备: mps
# 模型: htdemucs (stems=['drums', 'bass', 'other', 'vocals'], shifts=1, segment=7), 加载耗时: 0.117 s
# 输入音频: next_station_heaven.mp3 (44100 Hz, 30.041 s)
# 冷启动推理耗时: 2.747 s, RTF: 0.0914, 实时倍率: 10.94x
# 稳态推理耗时: 2.210 s, RTF: 0.0736, 实时倍率: 13.59x
# 已保存四轨目录: /Users/boom/workspace/LightClear/explore/light_demucs/outputs/mps/htdemucs/next_station_heaven
