import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

import numpy as np
import soundfile as sf
import torch
from inworld import voice_isolation

# 全局关键参数
CHECKPOINT_PATH = Path("/Users/boom/Model/SE/inworld-voice-isolator/denoiser.safetensors")
INPUT_16K = Path("/Users/boom/workspace/LightClear/assets/audio/noisy/input.wav")
INPUT_LONG = Path("/Users/boom/workspace/LightClear/assets/audio/noisy/speech1.wav")
INPUT_STEREO = Path("/Users/boom/workspace/LightClear/assets/audio/noisy/speech2_short.wav")
OUTPUT_DIR = MODULE_ROOT / "outputs" / "offline_options"
CHUNK_CUSTOM_S = 4.0
OVERLAP_CUSTOM_S = 1.0

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"缺少离线模型权重: {CHECKPOINT_PATH}")
for audio_path in (INPUT_16K, INPUT_LONG, INPUT_STEREO):
    if not audio_path.is_file():
        raise FileNotFoundError(f"缺少测试音频: {audio_path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型加载（默认自动设备）
isolator = voice_isolation.VoiceIsolator(str(CHECKPOINT_PATH), mode="offline")
print(f"默认离线设备: {isolator.device}, 默认精度: {isolator.precision}")

# 1. 输出采样率：48 kHz vs 保持输入采样率（16 kHz）
wav_16k, sr_16k = voice_isolation.load_audio(str(INPUT_16K))
voice_keep_sr = isolator.isolate(wav_16k, sr_16k, output_sample_rate=None)
voice_48k = isolator.isolate(wav_16k, sr_16k, output_sample_rate=48000)
print(
    f"[1. output_sample_rate] 输入 {sr_16k} Hz ({len(wav_16k)} samples) -> "
    f"保持原率={len(voice_keep_sr)} samples, 输出48k={len(voice_48k)} samples"
)
voice_isolation.save_audio(str(OUTPUT_DIR / "input_keep_16k.wav"), voice_keep_sr, sr_16k, subtype="FLOAT")
voice_isolation.save_audio(str(OUTPUT_DIR / "input_out_48k.wav"), voice_48k, 48000, subtype="FLOAT")

# 2. chunk_seconds / overlap_seconds：默认 8 s / 2 s vs 自定义 4 s / 1 s（拼接至 > 10 s 触发分块）
wav_long, sr_long = voice_isolation.load_audio(str(INPUT_LONG))
wav_12s = torch.cat([wav_long, wav_long, wav_long])[: int(12.0 * sr_long)]
t0 = time.perf_counter()
voice_chunk_default = isolator.isolate(wav_12s, sr_long, output_sample_rate=48000)
if isolator.device.type == "mps":
    torch.mps.synchronize()
t_default_chunk = time.perf_counter() - t0

t0 = time.perf_counter()
voice_chunk_custom = isolator.isolate(
    wav_12s,
    sr_long,
    output_sample_rate=48000,
    chunk_seconds=CHUNK_CUSTOM_S,
    overlap_seconds=OVERLAP_CUSTOM_S,
)
if isolator.device.type == "mps":
    torch.mps.synchronize()
t_custom_chunk = time.perf_counter() - t0
chunk_diff = float((voice_chunk_default - voice_chunk_custom).abs().max())
chunk_rms_diff = float((voice_chunk_default - voice_chunk_custom).pow(2).mean().sqrt())
print(
    f"[2. chunk/overlap] 12 s 音频: 默认(8s/2s) {t_default_chunk:.3f} s vs "
    f"自定义({CHUNK_CUSTOM_S}s/{OVERLAP_CUSTOM_S}s) {t_custom_chunk:.3f} s, "
    f"max_diff={chunk_diff:.5f}, rms_diff={chunk_rms_diff:.5f}"
)

# 3. offline_passes=1 vs 2（二次过模型对比）
isolator_pass2 = voice_isolation.VoiceIsolator(
    str(CHECKPOINT_PATH),
    device=isolator.device,
    mode="offline",
    offline_passes=2,
)
t0 = time.perf_counter()
voice_p1 = isolator.isolate(wav_16k, sr_16k, output_sample_rate=48000)
if isolator.device.type == "mps":
    torch.mps.synchronize()
t_p1 = time.perf_counter() - t0

t0 = time.perf_counter()
voice_p2 = isolator_pass2.isolate(wav_16k, sr_16k, output_sample_rate=48000)
if isolator.device.type == "mps":
    torch.mps.synchronize()
t_p2 = time.perf_counter() - t0
pass_diff_max = float((voice_p1 - voice_p2).abs().max())
pass_diff_rms = float((voice_p1 - voice_p2).pow(2).mean().sqrt())
print(
    f"[3. offline_passes] pass=1 ({t_p1:.3f} s) vs pass=2 ({t_p2:.3f} s), "
    f"max_diff={pass_diff_max:.5f}, rms_diff={pass_diff_rms:.5f}"
)
voice_isolation.save_audio(str(OUTPUT_DIR / "input_pass1.wav"), voice_p1, 48000, subtype="FLOAT")
voice_isolation.save_audio(str(OUTPUT_DIR / "input_pass2.wav"), voice_p2, 48000, subtype="FLOAT")

# 4. precision：fp32 / auto / bf16 / fp16
ref_fp32 = None
for prec in ("fp32", "auto", "bf16", "fp16"):
    iso_prec = voice_isolation.VoiceIsolator(
        str(CHECKPOINT_PATH),
        device=isolator.device,
        precision=prec,
        mode="offline",
    )
    t0 = time.perf_counter()
    try:
        out_prec = iso_prec.isolate(wav_16k, sr_16k, output_sample_rate=48000)
        if isolator.device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        has_nan = bool(torch.isnan(out_prec).any())
        if prec == "fp32":
            ref_fp32 = out_prec
            diff_str = "基准"
        elif has_nan:
            diff_str = "出现 NaN（该精度在当前设备数值溢出）"
        else:
            diff_str = f"vs fp32 max_diff={float((out_prec - ref_fp32).abs().max()):.5e}"
        print(f"[4. precision] 请求={prec} -> 实际={iso_prec.precision}, 耗时={dt:.3f} s, {diff_str}")
    except RuntimeError as err:
        print(f"[4. precision] 请求={prec} 在 {isolator.device} 上不兼容: {err}")

# 5. device：mps vs cpu
iso_cpu = voice_isolation.VoiceIsolator(str(CHECKPOINT_PATH), device="cpu", mode="offline")
t0 = time.perf_counter()
voice_cpu = iso_cpu.isolate(wav_16k, sr_16k, output_sample_rate=48000)
t_cpu = time.perf_counter() - t0
dev_diff = float((voice_p1 - voice_cpu).abs().max())
print(
    f"[5. device] {isolator.device} ({t_p1:.3f} s) vs cpu ({t_cpu:.3f} s), "
    f"max_diff={dev_diff:.5e}"
)

# 6. 立体声降混
stereo_np, sr_stereo = sf.read(str(INPUT_STEREO), dtype="float32", always_2d=True)
voice_from_stereo = isolator.isolate(stereo_np, sr_stereo, output_sample_rate=48000)
manual_mono = stereo_np.mean(axis=1)
voice_from_manual_mono = isolator.isolate(manual_mono, sr_stereo, output_sample_rate=48000)
stereo_diff = float(np.max(np.abs(voice_from_stereo - voice_from_manual_mono)))
print(
    f"[6. 立体声降混] 输入 shape={stereo_np.shape}, sr={sr_stereo}, "
    f"直接传入与手动 mean 降混最大误差={stereo_diff:.5e}"
)

# 7. NumPy / Torch Tensor 类型保持 与 PCM16 防削波保存
np_in = wav_16k.numpy()
np_out = isolator.isolate(np_in, sr_16k, output_sample_rate=48000)
tensor_out = isolator.isolate(wav_16k, sr_16k, output_sample_rate=48000)
print(
    f"[7. 类型保持] np.ndarray -> {type(np_out).__name__} (dtype={np_out.dtype}), "
    f"torch.Tensor -> {type(tensor_out).__name__} (dtype={tensor_out.dtype})"
)

loud_voice = tensor_out / max(float(tensor_out.abs().max()), 1e-6) * 1.35
float_path = OUTPUT_DIR / "peak_guard_float.wav"
pcm16_path = OUTPUT_DIR / "peak_guard_pcm16.wav"
voice_isolation.save_audio(str(float_path), loud_voice, 48000, subtype="FLOAT")
voice_isolation.save_audio(str(pcm16_path), loud_voice, 48000, subtype="PCM_16")
read_float, _ = sf.read(str(float_path), dtype="float32")
read_pcm16, _ = sf.read(str(pcm16_path), dtype="float32")
clipped_ratio = float(np.mean(np.abs(read_pcm16) >= 0.9999))
print(
    f"[8. PCM16 防削波] 原始峰值={float(loud_voice.abs().max()):.3f}, "
    f"FLOAT 读回峰值={float(np.max(np.abs(read_float))):.3f}, "
    f"PCM_16 读回峰值={float(np.max(np.abs(read_pcm16))):.5f} (自动峰值归一化, 满幅采样占比={clipped_ratio:.6f})"
)
