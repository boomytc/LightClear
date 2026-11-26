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

import numpy as np
import soundfile as sf
import torch
from inworld import voice_isolation

# 全局关键参数
DEVICE = "mps"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_PATH = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser.safetensors"
INPUT_16K = REPO_ROOT / "assets" / "audio" / "noisy" / "input.wav"
INPUT_LONG = REPO_ROOT / "assets" / "audio" / "noisy" / "speech1.wav"
INPUT_STEREO = REPO_ROOT / "assets" / "audio" / "noisy" / "speech2_short.wav"
OUTPUT_DIR = MODULE_ROOT / "outputs" / "mps" / "offline_options"
CHUNK_CUSTOM_S = 4.0
OVERLAP_CUSTOM_S = 1.0

if not torch.backends.mps.is_available():
    raise RuntimeError("当前环境 MPS 不可用，请在 Apple Silicon macOS 上运行 demo/mps/ 脚本")
if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"缺少离线模型权重: {CHECKPOINT_PATH}")
for audio_path in (INPUT_16K, INPUT_LONG, INPUT_STEREO):
    if not audio_path.is_file():
        raise FileNotFoundError(f"缺少测试音频: {audio_path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型加载
t_load_start = time.perf_counter()
isolator = voice_isolation.VoiceIsolator(str(CHECKPOINT_PATH), device=DEVICE, mode="offline")
torch.mps.synchronize()
load_elapsed_s = time.perf_counter() - t_load_start
print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {isolator.device}, 默认精度: {isolator.precision}, 加载耗时: {load_elapsed_s:.3f} s")

# 1. 输出采样率：48 kHz vs 保持输入采样率（16 kHz），同时记录冷启动与稳态
wav_16k, sr_16k = voice_isolation.load_audio(str(INPUT_16K))
dur_16k = wav_16k.shape[-1] / sr_16k
torch.mps.synchronize()
t0 = time.perf_counter()
voice_keep_sr = isolator.isolate(wav_16k, sr_16k, output_sample_rate=None)
torch.mps.synchronize()
t_cold = time.perf_counter() - t0

torch.mps.synchronize()
t0 = time.perf_counter()
voice_48k = isolator.isolate(wav_16k, sr_16k, output_sample_rate=48000)
torch.mps.synchronize()
t_warm = time.perf_counter() - t0
print(
    f"[1. output_sample_rate] 输入 {sr_16k} Hz ({len(wav_16k)} samples, {dur_16k:.3f} s) -> "
    f"保持原率={len(voice_keep_sr)} samples (冷启动={t_cold:.3f} s, RTF={t_cold / dur_16k:.4f}), "
    f"输出48k={len(voice_48k)} samples (稳态={t_warm:.3f} s, RTF={t_warm / dur_16k:.4f})"
)
voice_isolation.save_audio(str(OUTPUT_DIR / "input_keep_16k.wav"), voice_keep_sr, sr_16k, subtype="FLOAT")
voice_isolation.save_audio(str(OUTPUT_DIR / "input_out_48k.wav"), voice_48k, 48000, subtype="FLOAT")

# 2. chunk_seconds / overlap_seconds：默认 8 s / 2 s vs 自定义 4 s / 1 s（拼接至 12 s 触发分块）
wav_long, sr_long = voice_isolation.load_audio(str(INPUT_LONG))
wav_12s = torch.cat([wav_long, wav_long, wav_long])[: int(12.0 * sr_long)]
torch.mps.synchronize()
t0 = time.perf_counter()
voice_chunk_default = isolator.isolate(wav_12s, sr_long, output_sample_rate=48000)
torch.mps.synchronize()
t_default_chunk = time.perf_counter() - t0

torch.mps.synchronize()
t0 = time.perf_counter()
voice_chunk_custom = isolator.isolate(
    wav_12s,
    sr_long,
    output_sample_rate=48000,
    chunk_seconds=CHUNK_CUSTOM_S,
    overlap_seconds=OVERLAP_CUSTOM_S,
)
torch.mps.synchronize()
t_custom_chunk = time.perf_counter() - t0
chunk_diff = float((voice_chunk_default - voice_chunk_custom).abs().max())
chunk_rms_diff = float((voice_chunk_default - voice_chunk_custom).pow(2).mean().sqrt())
print(
    f"[2. chunk/overlap] 12.0 s 音频: 默认(8s/2s) {t_default_chunk:.3f} s (RTF={t_default_chunk / 12.0:.4f}) vs "
    f"自定义({CHUNK_CUSTOM_S}s/{OVERLAP_CUSTOM_S}s) {t_custom_chunk:.3f} s (RTF={t_custom_chunk / 12.0:.4f}), "
    f"max_diff={chunk_diff:.5f}, rms_diff={chunk_rms_diff:.5f}"
)

# 3. offline_passes=1 vs 2
isolator_pass2 = voice_isolation.VoiceIsolator(
    str(CHECKPOINT_PATH),
    device=DEVICE,
    mode="offline",
    offline_passes=2,
)
torch.mps.synchronize()
t0 = time.perf_counter()
voice_p1 = isolator.isolate(wav_16k, sr_16k, output_sample_rate=48000)
torch.mps.synchronize()
t_p1 = time.perf_counter() - t0

torch.mps.synchronize()
t0 = time.perf_counter()
voice_p2 = isolator_pass2.isolate(wav_16k, sr_16k, output_sample_rate=48000)
torch.mps.synchronize()
t_p2 = time.perf_counter() - t0
pass_diff_max = float((voice_p1 - voice_p2).abs().max())
pass_diff_rms = float((voice_p1 - voice_p2).pow(2).mean().sqrt())
print(
    f"[3. offline_passes] pass=1 ({t_p1:.3f} s, RTF={t_p1 / dur_16k:.4f}) vs "
    f"pass=2 ({t_p2:.3f} s, RTF={t_p2 / dur_16k:.4f}), "
    f"max_diff={pass_diff_max:.5f}, rms_diff={pass_diff_rms:.5f}"
)
voice_isolation.save_audio(str(OUTPUT_DIR / "input_pass1.wav"), voice_p1, 48000, subtype="FLOAT")
voice_isolation.save_audio(str(OUTPUT_DIR / "input_pass2.wav"), voice_p2, 48000, subtype="FLOAT")

# 4. precision：fp32 / auto / bf16 / fp16
ref_fp32 = None
for prec in ("fp32", "auto", "bf16", "fp16"):
    iso_prec = voice_isolation.VoiceIsolator(
        str(CHECKPOINT_PATH),
        device=DEVICE,
        precision=prec,
        mode="offline",
    )
    torch.mps.synchronize()
    t0 = time.perf_counter()
    try:
        out_prec = iso_prec.isolate(wav_16k, sr_16k, output_sample_rate=48000)
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
        print(f"[4. precision] 请求={prec} -> 实际={iso_prec.precision}, 耗时={dt:.3f} s (RTF={dt / dur_16k:.4f}), {diff_str}")
    except RuntimeError as err:
        print(f"[4. precision] 请求={prec} 在 {DEVICE} 上不兼容: {err}")

# 5. device：mps vs cpu 数值一致性对比
iso_cpu = voice_isolation.VoiceIsolator(str(CHECKPOINT_PATH), device="cpu", mode="offline")
t0 = time.perf_counter()
voice_cpu = iso_cpu.isolate(wav_16k, sr_16k, output_sample_rate=48000)
t_cpu = time.perf_counter() - t0
dev_diff = float((voice_p1 - voice_cpu).abs().max())
print(
    f"[5. device] {DEVICE} ({t_p1:.3f} s) vs cpu ({t_cpu:.3f} s), "
    f"加速比={t_cpu / t_p1:.2f}x, max_diff={dev_diff:.5e}"
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

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.15.0, 设备: mps, 默认精度: fp32, 加载耗时: 0.257 s
# [1. output_sample_rate] 输入 16000 Hz (38363 samples, 2.398 s) -> 保持原率=38363 samples (冷启动=0.560 s, RTF=0.2334), 输出48k=115089 samples (稳态=0.423 s, RTF=0.1766)
# [2. chunk/overlap] 12.0 s 音频: 默认(8s/2s) 3.400 s (RTF=0.2833) vs 自定义(4.0s/1.0s) 2.702 s (RTF=0.2251), max_diff=0.01923, rms_diff=0.00189
# [3. offline_passes] pass=1 (0.397 s, RTF=0.1655) vs pass=2 (0.799 s, RTF=0.3332), max_diff=0.00036, rms_diff=0.00005
# [4. precision] 请求=fp32 -> 实际=fp32, 耗时=0.389 s (RTF=0.1621), 基准
# [4. precision] 请求=auto -> 实际=fp32, 耗时=0.387 s (RTF=0.1615), vs fp32 max_diff=0.00000e+00
# [4. precision] 请求=bf16 -> 实际=bf16, 耗时=0.385 s (RTF=0.1608), vs fp32 max_diff=2.85752e-04
# [4. precision] 请求=fp16 -> 实际=fp16, 耗时=0.399 s (RTF=0.1664), 出现 NaN（该精度在当前设备数值溢出）
# [5. device] mps (0.397 s) vs cpu (44.096 s), 加速比=111.12x, max_diff=4.98444e-06
# [6. 立体声降混] 输入 shape=(84464, 2), sr=44100, 直接传入与手动 mean 降混最大误差=0.00000e+00
# [7. 类型保持] np.ndarray -> ndarray (dtype=float32), torch.Tensor -> Tensor (dtype=torch.float32)
# [8. PCM16 防削波] 原始峰值=1.350, FLOAT 读回峰值=1.350, PCM_16 读回峰值=1.00000 (自动峰值归一化, 满幅采样占比=0.000009)
