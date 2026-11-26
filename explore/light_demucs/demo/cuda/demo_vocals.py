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
STEM_NAME = "vocals"
DEVICE = "cuda:0"
SHIFTS = 1
OVERLAP = 0.25
SEGMENT = 7
MODEL_REPO = MODULE_ROOT / "models" / MODEL_NAME
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "music" / "next_station_heaven.mp3"
OUTPUT_DIR = MODULE_ROOT / "outputs" / "cuda" / MODEL_NAME
OUTPUT_VOCALS = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_{STEM_NAME}.wav"
OUTPUT_ACCOMPANIMENT = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_no_{STEM_NAME}.wav"

if not torch.cuda.is_available():
    raise RuntimeError("当前环境 CUDA 不可用，请在 NVIDIA GPU 环境运行 demo/cuda/ 脚本")
if not MODEL_REPO.is_dir():
    raise FileNotFoundError(f"缺少模型权重目录: {MODEL_REPO}")
if not INPUT_AUDIO.is_file():
    raise FileNotFoundError(f"缺少测试音频: {INPUT_AUDIO}")

gpu_name = torch.cuda.get_device_name(0)
gpu_total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

# 模型加载
torch.cuda.reset_peak_memory_stats(0)
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
torch.cuda.synchronize(0)
load_elapsed_s = time.perf_counter() - t_load_start

# 推理调用（冷启动 + 稳态二次计时）
torch.cuda.synchronize(0)
t_cold_start = time.perf_counter()
origin, _ = separator.separate_audio_file(INPUT_AUDIO)
torch.cuda.synchronize(0)
cold_elapsed_s = time.perf_counter() - t_cold_start

torch.cuda.synchronize(0)
t_warm_start = time.perf_counter()
origin, stems = separator.separate_audio_file(INPUT_AUDIO)
torch.cuda.synchronize(0)
warm_elapsed_s = time.perf_counter() - t_warm_start

peak_alloc_mib = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
peak_reserved_mib = torch.cuda.max_memory_reserved(0) / (1024 ** 2)
duration_s = origin.shape[-1] / separator.samplerate
vocals = stems[STEM_NAME].detach().cpu()
accompaniment = (stems["drums"] + stems["bass"] + stems["other"]).detach().cpu()

# 结果输出
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_audio(vocals, str(OUTPUT_VOCALS), samplerate=separator.samplerate)
save_audio(accompaniment, str(OUTPUT_ACCOMPANIMENT), samplerate=separator.samplerate)

print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {DEVICE}")
print(f"模型: {MODEL_NAME} (shifts={SHIFTS}, overlap={OVERLAP}, segment={SEGMENT}), 加载耗时: {load_elapsed_s:.3f} s")
print(f"输入音频: {INPUT_AUDIO.name} ({separator.samplerate} Hz, {duration_s:.3f} s)")
print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
print(f"稳态推理耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")
print(f"已保存人声: {OUTPUT_VOCALS}")
print(f"已保存伴奏: {OUTPUT_ACCOMPANIMENT}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | Linux-6.6.159+-x86_64-with-glibc2.39）
# =============================================================================
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0
# 模型: htdemucs (shifts=1, overlap=0.25, segment=7), 加载耗时: 0.533 s
# 输入音频: next_station_heaven.mp3 (44100 Hz, 30.041 s)
# 冷启动推理耗时: 1.699 s, RTF: 0.0566, 实时倍率: 17.68x
# 稳态推理耗时: 0.890 s, RTF: 0.0296, 实时倍率: 33.74x
# 峰值显存: allocated=546.8 MiB, reserved=770.0 MiB
# 已保存人声: /content/LightClear/explore/light_demucs/outputs/cuda/htdemucs/next_station_heaven_vocals.wav
# 已保存伴奏: /content/LightClear/explore/light_demucs/outputs/cuda/htdemucs/next_station_heaven_no_vocals.wav
