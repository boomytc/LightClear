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
from inworld.voice_isolation import audio as audio_lib
from inworld.voice_isolation import scan

# 全局关键参数
DEVICE = os.environ.get("LIGHTCLEAR_DEVICE", "auto")  # 可选: "auto" / "cuda" / "cuda:0" / "mps" / "cpu"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT") or (Path.home() / "Model"))
OFFLINE_CHECKPOINT = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser.safetensors"
REALTIME_CHECKPOINT = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser_realtime.safetensors"
CLIP_FILES = [
    REPO_ROOT / "assets" / "audio" / "noisy" / "p232_005.wav",
    REPO_ROOT / "assets" / "audio" / "noisy" / "p232_007.wav",
    REPO_ROOT / "assets" / "audio" / "noisy" / "test.wav",
]
OUTPUT_BASE = MODULE_ROOT / "outputs"
SAMPLE_RATE = 48000
EQUAL_LEN_SECONDS = 2.0

for ckpt in (OFFLINE_CHECKPOINT, REALTIME_CHECKPOINT):
    if not ckpt.is_file():
        raise FileNotFoundError(f"缺少模型权重: {ckpt}")
for audio_path in CLIP_FILES:
    if not audio_path.is_file():
        raise FileNotFoundError(f"缺少测试音频: {audio_path}")

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
    print("[警告] 当前运行在 CPU，Inworld Voice Isolator 批处理在 CPU 上耗时极长（生产建议使用 CUDA 或 MPS）")

OUTPUT_DIR = OUTPUT_BASE / device_type / "batch_many"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型加载
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(cuda_index)
    gpu_total_gib = torch.cuda.get_device_properties(cuda_index).total_memory / (1024 ** 3)
    cuda_kernel = scan.kernel()
    torch.cuda.reset_peak_memory_stats(cuda_index)

t_load_start = time.perf_counter()
offline = voice_isolation.VoiceIsolator(str(OFFLINE_CHECKPOINT), device=resolved_device, mode="offline")
streaming = voice_isolation.VoiceIsolator(
    str(REALTIME_CHECKPOINT),
    device=resolved_device,
    mode="streaming",
    lookahead_frames=2,
)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
load_elapsed_s = time.perf_counter() - t_load_start

if device_type == "cuda":
    print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {offline.device}, 精度: {offline.precision}")
    print(f"CUDA mamba_ssm 内核: {'命中' if cuda_kernel is not None else '未命中 (PyTorch 回退)'}, 双模型加载耗时: {load_elapsed_s:.3f} s")
else:
    print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {offline.device}, 精度: {offline.precision}, 双模型加载耗时: {load_elapsed_s:.3f} s")

# 输入构造（统一 48 kHz 单声道）
raw_clips = []
for audio_path in CLIP_FILES:
    wav, sr = voice_isolation.load_audio(str(audio_path))
    raw_clips.append(voice_isolation.resample(wav, sr, SAMPLE_RATE))

equal_n = int(EQUAL_LEN_SECONDS * SAMPLE_RATE)
equal_clips = [audio_lib.fit_length(c, equal_n) for c in raw_clips]
mixed_clips = [raw_clips[0][:48000], raw_clips[1], raw_clips[2][:72000], raw_clips[0][:48000]]
equal_total_s = len(equal_clips) * EQUAL_LEN_SECONDS
mixed_total_s = sum(int(c.shape[-1]) for c in mixed_clips) / SAMPLE_RATE

# 1. 离线同长合批 vs 逐条 isolate()
if device_type in ("cuda", "mps"):
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t0 = time.perf_counter()
    _ = offline.isolate_many(equal_clips, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t_equal_cold = time.perf_counter() - t0

t0 = time.perf_counter()
offline_equal_batch = offline.isolate_many(equal_clips, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t_equal_warm = time.perf_counter() - t0

t0 = time.perf_counter()
offline_equal_single = [
    offline.isolate(c, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE) for c in equal_clips
]
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t_equal_single = time.perf_counter() - t0

equal_errs = [
    float((b - s).abs().max()) for b, s in zip(offline_equal_batch, offline_equal_single)
]
if device_type in ("cuda", "mps"):
    print(
        f"[1. 离线同长合批] {len(equal_clips)} 条 x {EQUAL_LEN_SECONDS} s (总 {equal_total_s:.1f} s): "
        f"冷启动={t_equal_cold:.3f} s, 稳态批处理={t_equal_warm:.3f} s (RTF={t_equal_warm / equal_total_s:.4f}), "
        f"逐条={t_equal_single:.3f} s (RTF={t_equal_single / equal_total_s:.4f}), "
        f"逐条最大误差={[round(e, 7) for e in equal_errs]}"
    )
else:
    print(
        f"[1. 离线同长合批] {len(equal_clips)} 条 x {EQUAL_LEN_SECONDS} s (总 {equal_total_s:.1f} s): "
        f"批处理={t_equal_warm:.3f} s (RTF={t_equal_warm / equal_total_s:.4f}), "
        f"逐条={t_equal_single:.3f} s (RTF={t_equal_single / equal_total_s:.4f}), "
        f"逐条最大误差={[round(e, 7) for e in equal_errs]}"
    )

# 2. 离线异长自动分组并按原顺序返回
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
offline_mixed_batch = offline.isolate_many(mixed_clips, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t_mixed_batch = time.perf_counter() - t0
offline_mixed_single = [
    offline.isolate(c, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE) for c in mixed_clips
]
mixed_lens_in = [int(c.shape[-1]) for c in mixed_clips]
mixed_lens_out = [int(c.shape[-1]) for c in offline_mixed_batch]
mixed_errs = [
    float((b - s).abs().max()) for b, s in zip(offline_mixed_batch, offline_mixed_single)
]
print(
    f"[2. 离线异长分组] 输入长度={mixed_lens_in} -> 输出长度={mixed_lens_out} (总 {mixed_total_s:.2f} s), "
    f"耗时={t_mixed_batch:.3f} s (RTF={t_mixed_batch / mixed_total_s:.4f}), 最大误差={[round(e, 7) for e in mixed_errs]}"
)

# 3. 流式多条音频 lockstep 并发批流式隔离 vs 单条 streaming isolate()
if device_type in ("cuda", "mps"):
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t0 = time.perf_counter()
    _ = streaming.isolate_many(mixed_clips, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t_stream_cold = time.perf_counter() - t0

t0 = time.perf_counter()
stream_batch_out = streaming.isolate_many(mixed_clips, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t_stream_warm = time.perf_counter() - t0

t0 = time.perf_counter()
stream_single_out = [
    streaming.isolate(c, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE) for c in mixed_clips
]
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t_stream_single = time.perf_counter() - t0

stream_errs = [
    float((b - s).abs().max()) for b, s in zip(stream_batch_out, stream_single_out)
]
if device_type in ("cuda", "mps"):
    print(
        f"[3. 流式 lockstep 批处理] {len(mixed_clips)} 条异长音频 (总 {mixed_total_s:.2f} s): "
        f"冷启动={t_stream_cold:.3f} s, 稳态批流式={t_stream_warm:.3f} s (RTF={t_stream_warm / mixed_total_s:.4f}) vs "
        f"逐条流式={t_stream_single:.3f} s (RTF={t_stream_single / mixed_total_s:.4f}), "
        f"逐条最大误差={[round(e, 7) for e in stream_errs]}"
    )
else:
    print(
        f"[3. 流式 lockstep 批处理] {len(mixed_clips)} 条异长音频 (总 {mixed_total_s:.2f} s): "
        f"批流式={t_stream_warm:.3f} s (RTF={t_stream_warm / mixed_total_s:.4f}) vs "
        f"逐条流式={t_stream_single:.3f} s (RTF={t_stream_single / mixed_total_s:.4f}), "
        f"逐条最大误差={[round(e, 7) for e in stream_errs]}"
    )
if device_type == "cuda":
    peak_alloc_mib = torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 2)
    print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")

# 结果输出
for idx, wav_out in enumerate(stream_batch_out):
    out_path = OUTPUT_DIR / f"stream_batch_{idx}.wav"
    voice_isolation.save_audio(str(out_path), wav_out, SAMPLE_RATE, subtype="FLOAT")
print(f"已将流式批处理结果写入: {OUTPUT_DIR}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | CUDA）
# =============================================================================
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0, 精度: bf16
# CUDA mamba_ssm 内核: 未命中 (PyTorch 回退), 双模型加载耗时: 1.433 s
# [1. 离线同长合批] 3 条 x 2.0 s (总 6.0 s): 冷启动=1.748 s, 稳态批处理=1.335 s (RTF=0.2225), 逐条=1.125 s (RTF=0.1876), 逐条最大误差=[0.0006519, 0.0004443, 0.0009649]
# [2. 离线异长分组] 输入长度=[48000, 189883, 72000, 48000] -> 输出长度=[48000, 189883, 72000, 48000] (总 7.46 s), 耗时=1.485 s (RTF=0.1991), 最大误差=[0.0004313, 0.0, 0.0, 0.0004313]
# [3. 流式 lockstep 批处理] 4 条异长音频 (总 7.46 s): 冷启动=4.508 s, 稳态批流式=3.815 s (RTF=0.5117) vs 逐条流式=7.032 s (RTF=0.9432), 逐条最大误差=[0.0004493, 0.0004124, 0.0012384, 0.0004493]
# 峰值显存: allocated=5260.1 MiB, reserved=7188.0 MiB
# 已将流式批处理结果写入: /content/LightClear/explore/light_voice_isolator/outputs/cuda/batch_many

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.15.0, 设备: mps, 精度: fp32, 双模型加载耗时: 0.906 s
# [1. 离线同长合批] 3 条 x 2.0 s (总 6.0 s): 冷启动=1.310 s, 稳态批处理=2.398 s (RTF=0.3996), 逐条=2.695 s (RTF=0.4492), 逐条最大误差=[0.0, 0.0, 0.0]
# [2. 离线异长分组] 输入长度=[48000, 189883, 72000, 48000] -> 输出长度=[48000, 189883, 72000, 48000] (总 7.46 s), 耗时=3.585 s (RTF=0.4808), 最大误差=[1e-07, 0.0, 0.0, 1e-07]
# [3. 流式 lockstep 批处理] 4 条异长音频 (总 7.46 s): 冷启动=4.636 s, 稳态批流式=4.290 s (RTF=0.5754) vs 逐条流式=2.973 s (RTF=0.3987), 逐条最大误差=[1e-07, 1e-07, 3e-07, 1e-07]
# 已将流式批处理结果写入: /Users/boom/workspace/LightClear/explore/light_voice_isolator/outputs/mps/batch_many
# 注：CPU 离线 RTF > 4、流式 RTF > 11，批处理不建议在 CPU 运行。
