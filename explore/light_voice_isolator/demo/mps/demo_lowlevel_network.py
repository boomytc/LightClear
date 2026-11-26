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
from inworld.voice_isolation import scan

# 全局关键参数
DEVICE = "mps"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
OFFLINE_CHECKPOINT = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser.safetensors"
REALTIME_CHECKPOINT = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser_realtime.safetensors"
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "p232_005.wav"
OUTPUT_WAV = MODULE_ROOT / "outputs" / "mps" / "lowlevel_network" / f"{INPUT_AUDIO.stem}_whitebox.wav"

if not torch.backends.mps.is_available():
    raise RuntimeError("当前环境 MPS 不可用，请在 Apple Silicon macOS 上运行 demo/mps/ 脚本")
for ckpt in (OFFLINE_CHECKPOINT, REALTIME_CHECKPOINT):
    if not ckpt.is_file():
        raise FileNotFoundError(f"缺少模型权重: {ckpt}")
if not INPUT_AUDIO.is_file():
    raise FileNotFoundError(f"缺少测试音频: {INPUT_AUDIO}")

# 设备与底层 Mamba 扫描内核检测
device = torch.device(DEVICE)
cuda_kernel = scan.kernel()
metal_library = scan.metal_kernel()
print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {device}")
print(f"scan.kernel() [CUDA mamba_ssm]: {'命中' if cuda_kernel is not None else '未命中 (None)'}")
print(f"scan.metal_kernel() [Apple MPS Metal Shader]: {'命中 (已编译)' if metal_library is not None else '未命中 (None)'}")

# 模型加载
t_load_start = time.perf_counter()
offline_net = voice_isolation.load_model(str(OFFLINE_CHECKPOINT), device=device)
realtime_net = voice_isolation.load_model(str(REALTIME_CHECKPOINT), device=device)
torch.mps.synchronize()
load_elapsed_s = time.perf_counter() - t_load_start
print(f"双底层网络加载耗时: {load_elapsed_s:.3f} s")
print(
    f"离线网络 voice-isolator-v1: num_params={offline_net.num_params:,} ({offline_net.num_params / 1e6:.2f}M), "
    f"bands={len(offline_net.edges)}, width={offline_net.config.width}, "
    f"blocks={offline_net.config.block_types}, causal={offline_net.config.causal}, "
    f"chunk={offline_net.config.chunk_seconds}s / overlap={offline_net.config.overlap_seconds}s"
)
print(
    f"流式网络 voice-isolator-v1-realtime: num_params={realtime_net.num_params:,} ({realtime_net.num_params / 1e6:.2f}M), "
    f"bands={len(realtime_net.edges)}, width={realtime_net.config.width}, "
    f"blocks={realtime_net.config.block_types}, causal={realtime_net.config.causal}, "
    f"lookahead_choices={realtime_net.config.lookahead_frames_choices}, "
    f"attention_window_frames={realtime_net.config.attention_window_frames}"
)

# 输入构造（48 kHz 单声道 [batch=1, samples]）
wav, sr = voice_isolation.load_audio(str(INPUT_AUDIO))
wav_48k = voice_isolation.resample(wav, sr, offline_net.config.sample_rate)
duration_s = wav_48k.shape[-1] / offline_net.config.sample_rate
audio_batch = wav_48k[None].to(device)

# 白盒推理调用（冷启动 + 稳态二次计时）
torch.mps.synchronize()
t0 = time.perf_counter()
with torch.inference_mode():
    _ = offline_net(audio_batch)
torch.mps.synchronize()
cold_elapsed_s = time.perf_counter() - t0

torch.mps.synchronize()
t0 = time.perf_counter()
with torch.inference_mode():
    spectrum = offline_net.stft(audio_batch)
    encoded = offline_net.encode(spectrum)
    refined = offline_net.features(spectrum)
    raw = offline_net.denoise_decoder(refined)
    delta_spec = offline_net.correction(raw, spectrum)
    delta_wav = offline_net.istft(delta_spec, audio_batch.shape[-1])
    voice_whitebox = audio_batch + delta_wav
torch.mps.synchronize()
warm_elapsed_s = time.perf_counter() - t0

with torch.inference_mode():
    voice_forward = offline_net(audio_batch)
torch.mps.synchronize()

max_diff = float((voice_whitebox - voice_forward).abs().max())
print(
    f"张量形状链路:\n"
    f"  audio_batch : {tuple(audio_batch.shape)} ({duration_s:.3f} s)\n"
    f"  stft        : {tuple(spectrum.shape)} ({spectrum.dtype})\n"
    f"  encode      : {tuple(encoded.shape)}\n"
    f"  features    : {tuple(refined.shape)}\n"
    f"  decoder raw : {tuple(raw.shape)}\n"
    f"  correction  : {tuple(delta_spec.shape)}\n"
    f"  istft delta : {tuple(delta_wav.shape)}\n"
    f"冷启动前向耗时: {cold_elapsed_s:.3f} s (RTF={cold_elapsed_s / duration_s:.4f}), "
    f"稳态白盒耗时: {warm_elapsed_s:.3f} s (RTF={warm_elapsed_s / duration_s:.4f}, 实时倍率={duration_s / warm_elapsed_s:.2f}x), "
    f"vs network(audio) 最大误差: {max_diff:.3e}"
)

# 结果输出
OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
voice_isolation.save_audio(str(OUTPUT_WAV), voice_whitebox[0].cpu(), offline_net.config.sample_rate, subtype="FLOAT")
print(f"已保存白盒输出: {OUTPUT_WAV}")

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.15.0, 设备: mps
# scan.kernel() [CUDA mamba_ssm]: 未命中 (None)
# scan.metal_kernel() [Apple MPS Metal Shader]: 命中 (已编译)
# 双底层网络加载耗时: 0.815 s
# 离线网络 voice-isolator-v1: num_params=64,047,112 (64.05M), bands=46, width=512, blocks=('transformer', 'mamba', 'transformer', 'mamba', 'transformer', 'mamba', 'transformer'), causal=False, chunk=8.0s / overlap=2.0s
# 流式网络 voice-isolator-v1-realtime: num_params=58,178,056 (58.18M), bands=46, width=512, blocks=('transformer', 'mamba', 'transformer', 'mamba', 'transformer', 'mamba', 'transformer'), causal=True, lookahead_choices=(0, 2, 4), attention_window_frames=1200
# 张量形状链路:
#   audio_batch : (1, 299838) (6.247 s)
#   stft        : (1, 1025, 625) (torch.complex64)
#   encode      : (1, 46, 625, 512)
#   features    : (1, 46, 625, 512)
#   decoder raw : (1, 1025, 625, 4)
#   correction  : (1, 1025, 625)
#   istft delta : (1, 299838)
# 冷启动前向耗时: 1.279 s (RTF=0.2047), 稳态白盒耗时: 1.201 s (RTF=0.1923, 实时倍率=5.20x), vs network(audio) 最大误差: 0.000e+00
# 已保存白盒输出: /Users/boom/workspace/LightClear/explore/light_voice_isolator/outputs/mps/lowlevel_network/p232_005_whitebox.wav
