import os
import platform
import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent.parent
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

import torch
from inworld import voice_isolation
from inworld.voice_isolation import scan

# 全局关键参数
DEVICE = os.environ.get("LIGHTCLEAR_DEVICE", "auto")  # 可选: "auto" / "cuda" / "cuda:0" / "mps" / "cpu"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
OFFLINE_CHECKPOINT = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser.safetensors"
REALTIME_CHECKPOINT = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser_realtime.safetensors"
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "p232_005.wav"
OUTPUT_BASE = MODULE_ROOT / "outputs"

for ckpt in (OFFLINE_CHECKPOINT, REALTIME_CHECKPOINT):
    if not ckpt.is_file():
        raise FileNotFoundError(f"缺少模型权重: {ckpt}")
if not INPUT_AUDIO.is_file():
    raise FileNotFoundError(f"缺少测试音频: {INPUT_AUDIO}")

# 设备解析（支持 MPS Metal Shader：auto 优先级 cuda > mps > cpu，目标加速器不可用时回退 cpu）
requested_device = DEVICE.strip().lower()
if requested_device == "auto":
    if torch.cuda.is_available():
        resolved_device = "cuda:0"
    elif torch.backends.mps.is_available():
        resolved_device = "mps"
    else:
        resolved_device = "cpu"
elif requested_device == "cuda" or requested_device.startswith("cuda:"):
    if requested_device == "cuda":
        req_cuda_idx = 0
    else:
        idx_text = requested_device.split(":", 1)[1]
        if not idx_text.isdigit():
            raise ValueError(f"无效的 CUDA 设备索引: {DEVICE!r}")
        req_cuda_idx = int(idx_text)
    if torch.cuda.is_available() and req_cuda_idx < torch.cuda.device_count():
        resolved_device = f"cuda:{req_cuda_idx}"
    else:
        print(f"[警告] 请求设备 {DEVICE!r} 不可用，已自动回退到 cpu")
        resolved_device = "cpu"
elif requested_device == "mps":
    if torch.backends.mps.is_available():
        resolved_device = "mps"
    else:
        print(f"[警告] 请求设备 {DEVICE!r} 不可用，已自动回退到 cpu")
        resolved_device = "cpu"
elif requested_device == "cpu":
    resolved_device = "cpu"
else:
    raise ValueError(f"不支持的 DEVICE 配置: {DEVICE!r}，可选值为 auto / cuda / cuda:<idx> / mps / cpu")

device_type = "cuda" if resolved_device.startswith("cuda") else resolved_device
cuda_index = int(resolved_device.split(":", 1)[1]) if device_type == "cuda" else None
if device_type == "cpu":
    if platform.system() == "Darwin":
        torch.set_num_threads(2)
    print("[警告] 当前运行在 CPU，Inworld Voice Isolator 底层前向在 CPU 上耗时较长（生产建议使用 CUDA 或 MPS）")

OUTPUT_WAV = OUTPUT_BASE / device_type / "lowlevel_network" / f"{INPUT_AUDIO.stem}_whitebox.wav"

# 设备与底层 Mamba 扫描内核检测
device = torch.device(resolved_device)
cuda_kernel = scan.kernel()
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(cuda_index)
    gpu_total_gib = torch.cuda.get_device_properties(cuda_index).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {device}")
    print(f"scan.kernel() [CUDA mamba_ssm]: {'命中' if cuda_kernel is not None else '未命中 (None, PyTorch 回退)'}")
else:
    metal_library = scan.metal_kernel() if torch.backends.mps.is_available() else None
    print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {device}")
    print(f"scan.kernel() [CUDA mamba_ssm]: {'命中' if cuda_kernel is not None else '未命中 (None)'}")
    print(f"scan.metal_kernel() [Apple MPS Metal Shader]: {'命中 (已编译)' if metal_library is not None else '未命中 (None)'}")

# 模型加载
if device_type == "cuda":
    torch.cuda.reset_peak_memory_stats(cuda_index)
t_load_start = time.perf_counter()
offline_net = voice_isolation.load_model(str(OFFLINE_CHECKPOINT), device=device)
realtime_net = voice_isolation.load_model(str(REALTIME_CHECKPOINT), device=device)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
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

# 白盒推理调用（CUDA/MPS 测冷启动+稳态，CPU 测单次）
if device_type in ("cuda", "mps"):
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = offline_net(audio_batch)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    cold_elapsed_s = time.perf_counter() - t0

t0 = time.perf_counter()
with torch.inference_mode():
    spectrum = offline_net.stft(audio_batch)
    encoded = offline_net.encode(spectrum)
    refined = offline_net.features(spectrum)
    raw = offline_net.denoise_decoder(refined)
    delta_spec = offline_net.correction(raw, spectrum)
    delta_wav = offline_net.istft(delta_spec, audio_batch.shape[-1])
    voice_whitebox = audio_batch + delta_wav
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
warm_elapsed_s = time.perf_counter() - t0

with torch.inference_mode():
    voice_forward = offline_net(audio_batch)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()

max_diff = float((voice_whitebox - voice_forward).abs().max())
timing_line = (
    f"冷启动前向耗时: {cold_elapsed_s:.3f} s (RTF={cold_elapsed_s / duration_s:.4f}), "
    f"稳态白盒耗时: {warm_elapsed_s:.3f} s (RTF={warm_elapsed_s / duration_s:.4f}, 实时倍率={duration_s / warm_elapsed_s:.2f}x), "
    if device_type in ("cuda", "mps")
    else f"白盒推理耗时: {warm_elapsed_s:.3f} s (RTF={warm_elapsed_s / duration_s:.4f}, 实时倍率={duration_s / warm_elapsed_s:.2f}x), "
)
print(
    f"张量形状链路:\n"
    f"  audio_batch : {tuple(audio_batch.shape)} ({duration_s:.3f} s)\n"
    f"  stft        : {tuple(spectrum.shape)} ({spectrum.dtype})\n"
    f"  encode      : {tuple(encoded.shape)}\n"
    f"  features    : {tuple(refined.shape)}\n"
    f"  decoder raw : {tuple(raw.shape)}\n"
    f"  correction  : {tuple(delta_spec.shape)}\n"
    f"  istft delta : {tuple(delta_wav.shape)}\n"
    f"{timing_line}vs network(audio) 最大误差: {max_diff:.3e}"
)
if device_type == "cuda":
    peak_alloc_mib = torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 2)
    print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")

# 结果输出
OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
voice_isolation.save_audio(str(OUTPUT_WAV), voice_whitebox[0].cpu(), offline_net.config.sample_rate, subtype="FLOAT")
print(f"已保存白盒输出: {OUTPUT_WAV}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | CUDA）
# =============================================================================
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0
# scan.kernel() [CUDA mamba_ssm]: 未命中 (None, PyTorch 回退)
# 双底层网络加载耗时: 1.444 s
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
# 冷启动前向耗时: 1.990 s (RTF=0.3186), 稳态白盒耗时: 1.685 s (RTF=0.2697, 实时倍率=3.71x), vs network(audio) 最大误差: 0.000e+00
# 峰值显存: allocated=5776.2 MiB, reserved=6484.0 MiB
# 已保存白盒输出: /content/LightClear/explore/light_voice_isolator/outputs/cuda/lowlevel_network/p232_005_whitebox.wav

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
# 注：CPU 前向推理 RTF > 4，生产建议使用 CUDA 或 MPS。
