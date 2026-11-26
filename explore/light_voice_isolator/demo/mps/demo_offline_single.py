import os
import platform
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
DEVICE = "mps"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_PATH = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser.safetensors"
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "p232_005.wav"
OUTPUT_DIR = MODULE_ROOT / "outputs" / "mps" / "offline_single"
OUTPUT_VOICE = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_voice.wav"
OUTPUT_BACKGROUND = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_background.wav"
OUTPUT_SAMPLE_RATE = 48000

if not torch.backends.mps.is_available():
    raise RuntimeError("当前环境 MPS 不可用，请在 Apple Silicon macOS 上运行 demo/mps/ 脚本")
if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"缺少离线模型权重，请先下载到中心目录: {CHECKPOINT_PATH}")
if not INPUT_AUDIO.is_file():
    raise FileNotFoundError(f"缺少测试音频: {INPUT_AUDIO}")

# 模型加载
t_load_start = time.perf_counter()
isolator = voice_isolation.VoiceIsolator(
    str(CHECKPOINT_PATH),
    device=DEVICE,
    mode="offline",
)
torch.mps.synchronize()
load_elapsed_s = time.perf_counter() - t_load_start
metal_kernel = scan.metal_kernel()

# 输入构造
wav, input_sr = voice_isolation.load_audio(str(INPUT_AUDIO))
duration_s = wav.shape[-1] / input_sr
aligned_mono = voice_isolation.resample(wav, input_sr, OUTPUT_SAMPLE_RATE)

# 推理调用（冷启动 + 稳态二次计时）
torch.mps.synchronize()
t_cold_start = time.perf_counter()
_ = isolator.isolate(wav, input_sr, output_sample_rate=OUTPUT_SAMPLE_RATE)
torch.mps.synchronize()
cold_elapsed_s = time.perf_counter() - t_cold_start

torch.mps.synchronize()
t_warm_start = time.perf_counter()
voice = isolator.isolate(wav, input_sr, output_sample_rate=OUTPUT_SAMPLE_RATE)
torch.mps.synchronize()
warm_elapsed_s = time.perf_counter() - t_warm_start

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

print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {isolator.device}, 精度: {isolator.precision}")
print(f"Metal Shader 内核: {'命中 (已编译)' if metal_kernel is not None else '未命中 (PyTorch 回退)'}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"输入文件: {INPUT_AUDIO.name} ({input_sr} Hz, {duration_s:.3f} s)")
print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
print(f"稳态推理耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
print(f"电平保持: input_rms={input_rms:.5f}, voice_rms={voice_rms:.5f}, bg_rms={bg_rms:.5f}")
print(f"峰值统计: input_peak={input_peak:.5f}, voice_peak={voice_peak:.5f}")
print(f"重构最大误差 max(|input - (voice + background)|): {recon_max_err:.3e}")
print(f"已保存人声: {OUTPUT_VOICE}")
print(f"已保存残差: {OUTPUT_BACKGROUND}")

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.15.0, 设备: mps, 精度: fp32
# Metal Shader 内核: 命中 (已编译), 加载耗时: 0.537 s
# 输入文件: p232_005.wav (48000 Hz, 6.247 s)
# 冷启动推理耗时: 1.824 s, RTF: 0.2920, 实时倍率: 3.42x
# 稳态推理耗时: 1.242 s, RTF: 0.1988, 实时倍率: 5.03x
# 电平保持: input_rms=0.10430, voice_rms=0.08065, bg_rms=0.06507
# 峰值统计: input_peak=0.69562, voice_peak=0.50109
# 重构最大误差 max(|input - (voice + background)|): 0.000e+00
# 已保存人声: /Users/boom/workspace/LightClear/explore/light_voice_isolator/outputs/mps/offline_single/p232_005_voice.wav
# 已保存残差: /Users/boom/workspace/LightClear/explore/light_voice_isolator/outputs/mps/offline_single/p232_005_background.wav
