import os
import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = MODULE_ROOT.parent.parent
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

import torch
from inworld import voice_isolation
from inworld.voice_isolation import audio as audio_lib
from inworld.voice_isolation import scan

# 全局关键参数
DEVICE = "cuda:0"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_PATH = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser.safetensors"
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "p232_005.wav"
OUTPUT_DIR = MODULE_ROOT / "outputs" / "cuda" / "offline_single"
OUTPUT_VOICE = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_voice.wav"
OUTPUT_BACKGROUND = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_background.wav"
OUTPUT_SAMPLE_RATE = 48000

if not torch.cuda.is_available():
    raise RuntimeError("当前环境 CUDA 不可用，请在 NVIDIA GPU 环境运行 demo/cuda/ 脚本")
if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"缺少离线模型权重，请先下载到中心目录: {CHECKPOINT_PATH}")
if not INPUT_AUDIO.is_file():
    raise FileNotFoundError(f"缺少测试音频: {INPUT_AUDIO}")

gpu_name = torch.cuda.get_device_name(0)
gpu_total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
cuda_kernel = scan.kernel()

# 模型加载
torch.cuda.reset_peak_memory_stats(0)
t_load_start = time.perf_counter()
isolator = voice_isolation.VoiceIsolator(
    str(CHECKPOINT_PATH),
    device=DEVICE,
    mode="offline",
)
torch.cuda.synchronize(0)
load_elapsed_s = time.perf_counter() - t_load_start

# 输入构造
wav, input_sr = voice_isolation.load_audio(str(INPUT_AUDIO))
duration_s = wav.shape[-1] / input_sr
aligned_mono = voice_isolation.resample(wav, input_sr, OUTPUT_SAMPLE_RATE)

# 推理调用（冷启动 + 稳态二次计时）
torch.cuda.synchronize(0)
t_cold_start = time.perf_counter()
_ = isolator.isolate(wav, input_sr, output_sample_rate=OUTPUT_SAMPLE_RATE)
torch.cuda.synchronize(0)
cold_elapsed_s = time.perf_counter() - t_cold_start

torch.cuda.synchronize(0)
t_warm_start = time.perf_counter()
voice = isolator.isolate(wav, input_sr, output_sample_rate=OUTPUT_SAMPLE_RATE)
torch.cuda.synchronize(0)
warm_elapsed_s = time.perf_counter() - t_warm_start

peak_alloc_mib = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
peak_reserved_mib = torch.cuda.max_memory_reserved(0) / (1024 ** 2)
aligned_mono = audio_lib.fit_length(aligned_mono, voice.shape[-1])
background = aligned_mono - voice

# 结果输出与数值校验
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
voice_isolation.save_audio(str(OUTPUT_VOICE), voice, OUTPUT_SAMPLE_RATE, subtype="FLOAT")
voice_isolation.save_audio(str(OUTPUT_BACKGROUND), background, OUTPUT_SAMPLE_RATE, subtype="FLOAT")

reconstructed = voice + background
recon_max_err = float((aligned_mono - reconstructed).abs().max())
input_rms = float(aligned_mono.pow(2).mean().sqrt())
voice_rms = float(voice.pow(2).mean().sqrt())
bg_rms = float(background.pow(2).mean().sqrt())
input_peak = float(aligned_mono.abs().max())
voice_peak = float(voice.abs().max())

print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {isolator.device}, 精度: {isolator.precision}")
print(f"CUDA mamba_ssm 内核: {'命中' if cuda_kernel is not None else '未命中 (PyTorch 回退)'}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"输入文件: {INPUT_AUDIO.name} ({input_sr} Hz, {duration_s:.3f} s)")
print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
print(f"稳态推理耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")
print(f"电平保持: input_rms={input_rms:.5f}, voice_rms={voice_rms:.5f}, bg_rms={bg_rms:.5f}")
print(f"峰值统计: input_peak={input_peak:.5f}, voice_peak={voice_peak:.5f}")
print(f"重构最大误差 max(|input - (voice + background)|): {recon_max_err:.3e}")
print(f"已保存人声: {OUTPUT_VOICE}")
print(f"已保存残差: {OUTPUT_BACKGROUND}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | Linux-6.6.159+-x86_64-with-glibc2.39）
# =============================================================================
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0, 精度: bf16
# CUDA mamba_ssm 内核: 未命中 (PyTorch 回退), 加载耗时: 0.773 s
# 输入文件: p232_005.wav (48000 Hz, 6.247 s)
# 冷启动推理耗时: 2.030 s, RTF: 0.3250, 实时倍率: 3.08x
# 稳态推理耗时: 1.384 s, RTF: 0.2216, 实时倍率: 4.51x
# 峰值显存: allocated=5150.1 MiB, reserved=5716.0 MiB
# 电平保持: input_rms=0.10430, voice_rms=0.08070, bg_rms=0.06511
# 峰值统计: input_peak=0.69562, voice_peak=0.50109
# 重构最大误差 max(|input - (voice + background)|): 0.000e+00
# 已保存人声: /content/LightClear/explore/light_voice_isolator/outputs/cuda/offline_single/p232_005_voice.wav
# 已保存残差: /content/LightClear/explore/light_voice_isolator/outputs/cuda/offline_single/p232_005_background.wav
