import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

import torch
from inworld import voice_isolation
from inworld.voice_isolation import audio as audio_lib

# 全局关键参数
CHECKPOINT_PATH = Path("/Users/boom/Model/SE/inworld-voice-isolator/denoiser.safetensors")
INPUT_AUDIO = Path("/Users/boom/workspace/LightClear/assets/audio/noisy/p232_005.wav")
OUTPUT_DIR = MODULE_ROOT / "outputs" / "offline_single"
OUTPUT_VOICE = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_voice.wav"
OUTPUT_BACKGROUND = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_background.wav"
OUTPUT_SAMPLE_RATE = 48000
DEVICE = None  # None 自动选择：Apple Silicon (torch>=2.6) -> mps，否则 cpu

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
load_elapsed_s = time.perf_counter() - t_load_start

# 输入构造
wav, input_sr = voice_isolation.load_audio(str(INPUT_AUDIO))
duration_s = wav.shape[-1] / input_sr
aligned_mono = voice_isolation.resample(wav, input_sr, OUTPUT_SAMPLE_RATE)

# 推理调用
if isolator.device.type == "mps":
    torch.mps.synchronize()
t_infer_start = time.perf_counter()
voice = isolator.isolate(wav, input_sr, output_sample_rate=OUTPUT_SAMPLE_RATE)
if isolator.device.type == "mps":
    torch.mps.synchronize()
infer_elapsed_s = time.perf_counter() - t_infer_start

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
rtf = infer_elapsed_s / duration_s

print(f"设备: {isolator.device}, 精度: {isolator.precision}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"输入文件: {INPUT_AUDIO} ({input_sr} Hz, {duration_s:.3f} s)")
print(f"推理耗时: {infer_elapsed_s:.3f} s, RTF: {rtf:.4f}")
print(f"电平保持: input_rms={input_rms:.5f}, voice_rms={voice_rms:.5f}, bg_rms={bg_rms:.5f}")
print(f"峰值统计: input_peak={input_peak:.5f}, voice_peak={voice_peak:.5f}")
print(f"重构最大误差 max(|input - (voice + background)|): {recon_max_err:.3e}")
print(f"已保存人声: {OUTPUT_VOICE}")
print(f"已保存残差: {OUTPUT_BACKGROUND}")
