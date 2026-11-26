import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

import torch
from inworld import voice_isolation
from inworld.voice_isolation import isolator as isolator_mod
from inworld.voice_isolation import scan

# 全局关键参数
OFFLINE_CHECKPOINT = Path("/Users/boom/Model/SE/inworld-voice-isolator/denoiser.safetensors")
REALTIME_CHECKPOINT = Path("/Users/boom/Model/SE/inworld-voice-isolator/denoiser_realtime.safetensors")
INPUT_AUDIO = Path("/Users/boom/workspace/LightClear/assets/audio/noisy/p232_005.wav")
OUTPUT_WAV = MODULE_ROOT / "outputs" / "lowlevel_network" / f"{INPUT_AUDIO.stem}_whitebox.wav"

for ckpt in (OFFLINE_CHECKPOINT, REALTIME_CHECKPOINT):
    if not ckpt.is_file():
        raise FileNotFoundError(f"缺少模型权重: {ckpt}")
if not INPUT_AUDIO.is_file():
    raise FileNotFoundError(f"缺少测试音频: {INPUT_AUDIO}")

# 设备与底层 Mamba 扫描内核检测
device = isolator_mod.default_device()
cuda_kernel = scan.kernel()
metal_library = scan.metal_kernel() if device.type == "mps" else None
print(f"自动选择设备: {device} (torch={torch.__version__})")
print(f"scan.kernel() [CUDA mamba_ssm]: {'命中' if cuda_kernel is not None else '未命中 (None)'}")
print(f"scan.metal_kernel() [Apple MPS Metal Shader]: {'命中 (已编译)' if metal_library is not None else '未命中 (None)'}")

# 模型加载（直接使用底层 load_model 获取 VoiceIsolatorNetwork）
offline_net = voice_isolation.load_model(str(OFFLINE_CHECKPOINT), device=device)
realtime_net = voice_isolation.load_model(str(REALTIME_CHECKPOINT), device=device)
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
audio_batch = wav_48k[None].to(device)

# 白盒推理调用：显式走 stft -> encode -> features -> denoise_decoder -> correction -> istft
t0 = time.perf_counter()
with torch.inference_mode():
    spectrum = offline_net.stft(audio_batch)
    encoded = offline_net.encode(spectrum)
    refined = offline_net.features(spectrum)
    raw = offline_net.denoise_decoder(refined)
    delta_spec = offline_net.correction(raw, spectrum)
    delta_wav = offline_net.istft(delta_spec, audio_batch.shape[-1])
    voice_whitebox = audio_batch + delta_wav
    voice_forward = offline_net(audio_batch)
if device.type == "mps":
    torch.mps.synchronize()
elapsed_s = time.perf_counter() - t0

max_diff = float((voice_whitebox - voice_forward).abs().max())
print(
    f"张量形状链路:\n"
    f"  audio_batch : {tuple(audio_batch.shape)}\n"
    f"  stft        : {tuple(spectrum.shape)} ({spectrum.dtype})\n"
    f"  encode      : {tuple(encoded.shape)}\n"
    f"  features    : {tuple(refined.shape)}\n"
    f"  decoder raw : {tuple(raw.shape)}\n"
    f"  correction  : {tuple(delta_spec.shape)}\n"
    f"  istft delta : {tuple(delta_wav.shape)}\n"
    f"白盒链路 vs network(audio) 最大误差: {max_diff:.3e}, 耗时: {elapsed_s:.3f} s"
)

# 结果输出
OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
voice_isolation.save_audio(str(OUTPUT_WAV), voice_whitebox[0].cpu(), offline_net.config.sample_rate, subtype="FLOAT")
print(f"已保存白盒输出: {OUTPUT_WAV}")
