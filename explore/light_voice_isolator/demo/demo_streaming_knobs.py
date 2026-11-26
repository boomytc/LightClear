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
REALTIME_CHECKPOINT = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser_realtime.safetensors"
SHORT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "p232_005.wav"
LONG_AUDIO = REPO_ROOT / "assets" / "audio" / "music" / "next_station_heaven.mp3"
OUTPUT_BASE = MODULE_ROOT / "outputs"
PIECE_SAMPLES = 9600  # 200 ms 分片模拟连续推流
LONG_CLIP_SECONDS = 18.0

if not REALTIME_CHECKPOINT.is_file():
    raise FileNotFoundError(f"缺少实时流式模型权重: {REALTIME_CHECKPOINT}")
for audio_path in (SHORT_AUDIO, LONG_AUDIO):
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
    print("[警告] 当前运行在 CPU，Inworld Voice Isolator 流式旋钮全量对比在 CPU 上耗时极长（生产建议使用 MPS 或 CUDA）")

OUTPUT_DIR = OUTPUT_BASE / device_type / "streaming_knobs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型加载
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(cuda_index)
    gpu_total_gib = torch.cuda.get_device_properties(cuda_index).total_memory / (1024 ** 3)
    cuda_kernel = scan.kernel()
    torch.cuda.reset_peak_memory_stats(cuda_index)

t_load_start = time.perf_counter()
live = voice_isolation.VoiceIsolator(
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
    print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {live.device}, 精度: {live.precision}")
    print(f"CUDA mamba_ssm 内核: {'命中' if cuda_kernel is not None else '未命中 (PyTorch 回退)'}, 加载耗时: {load_elapsed_s:.3f} s")
else:
    print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {live.device}, 精度: {live.precision}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"支持的 lookahead_choices={live.lookahead_choices}")

# 输入构造
wav_short, sr_short = voice_isolation.load_audio(str(SHORT_AUDIO))
wav_short = voice_isolation.resample(wav_short, sr_short, live.sample_rate)
short_dur_s = wav_short.shape[-1] / live.sample_rate

wav_long, sr_long = voice_isolation.load_audio(str(LONG_AUDIO))
wav_long = voice_isolation.resample(wav_long, sr_long, live.sample_rate)[: int(LONG_CLIP_SECONDS * live.sample_rate)]
long_dur_s = wav_long.shape[-1] / live.sample_rate

# 预热（仅 CUDA/MPS 执行冷启动预热）
if device_type in ("cuda", "mps"):
    warmup_session = live.stream()
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for i in range(0, wav_short.shape[-1], PIECE_SAMPLES):
        _ = warmup_session.push(wav_short[i : i + PIECE_SAMPLES])
    _ = warmup_session.flush()
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t_cold_warmup = time.perf_counter() - t0
    print(f"[0. 冷启动预热] 短音频 ({short_dur_s:.3f} s) 首次流式耗时={t_cold_warmup:.3f} s (RTF={t_cold_warmup / short_dur_s:.4f})")

# 1. lookahead_frames = 0 / 2 / 4
lookahead_outputs = {}
for la in live.lookahead_choices:
    live.lookahead_frames = la
    session = live.stream()
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    elif device_type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    parts = [session.push(wav_short[i : i + PIECE_SAMPLES]) for i in range(0, wav_short.shape[-1], PIECE_SAMPLES)]
    parts.append(session.flush())
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    elif device_type == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    out_la = torch.cat(parts)
    lookahead_outputs[la] = out_la
    print(
        f"[1. lookahead={la}] 算法延迟={session.latency_samples} samples ({session.latency_seconds * 1000:.2f} ms), "
        f"耗时={dt:.3f} s, RTF={dt / short_dur_s:.4f}"
    )
    voice_isolation.save_audio(str(OUTPUT_DIR / f"lookahead_{la}.wav"), out_la, live.sample_rate, subtype="FLOAT")

diff_0_2 = float((lookahead_outputs[0] - lookahead_outputs[2]).pow(2).mean().sqrt())
diff_2_4 = float((lookahead_outputs[2] - lookahead_outputs[4]).pow(2).mean().sqrt())
print(f"   lookahead RMS 差异: (0 vs 2)={diff_0_2:.5f}, (2 vs 4)={diff_2_4:.5f}")

# 2. block_frames = 1 / 4 / 16
live.lookahead_frames = 2
block_ref = None
for bf in (1, 4, 16):
    session = live.stream(block_frames=bf)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    elif device_type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    parts = [session.push(wav_short[i : i + PIECE_SAMPLES]) for i in range(0, wav_short.shape[-1], PIECE_SAMPLES)]
    parts.append(session.flush())
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    elif device_type == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    out_bf = torch.cat(parts)
    if block_ref is None:
        block_ref = out_bf
    diff_bf = float((out_bf - block_ref).abs().max())
    print(
        f"[2. block_frames={bf:2d}] latency={session.latency_samples} samples, "
        f"耗时={dt:.3f} s, RTF={dt / short_dur_s:.4f}, vs block=1 max_diff={diff_bf:.3e}"
    )

# 3. cache_frames = 100 / 300 / 800
cache_outputs = {}
for cf in (100, 300, 800):
    session = live.stream(cache_frames=cf, window_frames=0)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    elif device_type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    parts = [session.push(wav_short[i : i + PIECE_SAMPLES]) for i in range(0, wav_short.shape[-1], PIECE_SAMPLES)]
    parts.append(session.flush())
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    elif device_type == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    cache_outputs[cf] = torch.cat(parts)
    print(
        f"[3. cache_frames={cf:3d} (window=0)] 实际 cache={session.cache_frames}, "
        f"耗时={dt:.3f} s, RTF={dt / short_dur_s:.4f}"
    )
print(
    f"   cache RMS 差异: (100 vs 800)={float((cache_outputs[100] - cache_outputs[800]).pow(2).mean().sqrt()):.5f}, "
    f"(300 vs 800)={float((cache_outputs[300] - cache_outputs[800]).pow(2).mean().sqrt()):.5f}"
)

# 4. 双轨滑窗 window_frames/crossfade_frames：600/200 vs 0
session_win = live.stream(window_frames=600, crossfade_frames=200)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
parts_win = [session_win.push(wav_long[i : i + PIECE_SAMPLES]) for i in range(0, wav_long.shape[-1], PIECE_SAMPLES)]
parts_win.append(session_win.flush())
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t_win = time.perf_counter() - t0
out_win = torch.cat(parts_win)

session_nowin = live.stream(window_frames=0)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
parts_nowin = [session_nowin.push(wav_long[i : i + PIECE_SAMPLES]) for i in range(0, wav_long.shape[-1], PIECE_SAMPLES)]
parts_nowin.append(session_nowin.flush())
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t_nowin = time.perf_counter() - t0
out_nowin = torch.cat(parts_nowin)

seg_early = slice(0, int(6.0 * live.sample_rate))
seg_late = slice(int(12.0 * live.sample_rate), int(18.0 * live.sample_rate))
early_rms_diff = float((out_win[seg_early] - out_nowin[seg_early]).pow(2).mean().sqrt())
late_rms_diff = float((out_win[seg_late] - out_nowin[seg_late]).pow(2).mean().sqrt())
print(
    f"[4. 双轨滑窗对比] {long_dur_s:.1f} s 长音频:\n"
    f"   默认(600/200): 耗时={t_win:.3f} s, RTF={t_win / long_dur_s:.4f}, 切轨边界帧={session_win.boundaries}\n"
    f"   关闭(window=0): 耗时={t_nowin:.3f} s, RTF={t_nowin / long_dur_s:.4f}, 切轨边界帧={session_nowin.boundaries}\n"
    f"   差异对比: 前6s RMS diff={early_rms_diff:.5f}, 后6s(12-18s) RMS diff={late_rms_diff:.5f}"
)
if device_type == "cuda":
    peak_alloc_mib = torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 2)
    print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")

voice_isolation.save_audio(str(OUTPUT_DIR / "long_windowed_600_200.wav"), out_win, live.sample_rate, subtype="FLOAT")
voice_isolation.save_audio(str(OUTPUT_DIR / "long_unwindowed_0.wav"), out_nowin, live.sample_rate, subtype="FLOAT")
print(f"已保存高阶参数对比音频至: {OUTPUT_DIR}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | CUDA）
# =============================================================================
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0, 精度: bf16
# CUDA mamba_ssm 内核: 未命中 (PyTorch 回退), 加载耗时: 0.713 s
# 支持的 lookahead_choices=(0, 2, 4)
# [0. 冷启动预热] 短音频 (6.247 s) 首次流式耗时=6.083 s (RTF=0.9738)
# [1. lookahead=0] 算法延迟=2048 samples (42.67 ms), 稳态耗时=5.650 s, RTF=0.9044
# [1. lookahead=2] 算法延迟=3008 samples (62.67 ms), 稳态耗时=5.630 s, RTF=0.9012
# [1. lookahead=4] 算法延迟=3968 samples (82.67 ms), 稳态耗时=5.654 s, RTF=0.9052
#    lookahead RMS 差异: (0 vs 2)=0.00224, (2 vs 4)=0.00180
# [2. block_frames= 1] latency=3008 samples, 稳态耗时=21.001 s, RTF=3.3619, vs block=1 max_diff=0.000e+00
# [2. block_frames= 4] latency=3008 samples, 稳态耗时=5.649 s, RTF=0.9043, vs block=1 max_diff=8.103e-04
# [2. block_frames=16] latency=3008 samples, 稳态耗时=2.393 s, RTF=0.3831, vs block=1 max_diff=6.349e-04
# [3. cache_frames=100 (window=0)] 实际 cache=100, 稳态耗时=5.499 s, RTF=0.8804
# [3. cache_frames=300 (window=0)] 实际 cache=300, 稳态耗时=5.450 s, RTF=0.8725
# [3. cache_frames=800 (window=0)] 实际 cache=800, 稳态耗时=5.511 s, RTF=0.8822
#    cache RMS 差异: (100 vs 800)=0.00117, (300 vs 800)=0.00043
# [4. 双轨滑窗对比] 18.0 s 长音频:
#    默认(600/200): 耗时=16.238 s, RTF=0.9021, 切轨边界帧=[600, 1200, 1800]
#    关闭(window=0): 耗时=15.613 s, RTF=0.8674, 切轨边界帧=[]
#    差异对比: 前6s RMS diff=0.00158, 后6s(12-18s) RMS diff=0.05252
# 峰值显存: allocated=2499.3 MiB, reserved=3182.0 MiB
# 已保存高阶参数对比音频至: /content/LightClear/explore/light_voice_isolator/outputs/cuda/streaming_knobs

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.15.0, 设备: mps, 精度: fp32, 加载耗时: 0.441 s
# 支持的 lookahead_choices=(0, 2, 4)
# [0. 冷启动预热] 短音频 (6.247 s) 首次流式耗时=2.883 s (RTF=0.4615)
# [1. lookahead=0] 算法延迟=2048 samples (42.67 ms), 稳态耗时=2.732 s, RTF=0.4373
# [1. lookahead=2] 算法延迟=3008 samples (62.67 ms), 稳态耗时=2.700 s, RTF=0.4322
# [1. lookahead=4] 算法延迟=3968 samples (82.67 ms), 稳态耗时=2.622 s, RTF=0.4197
#    lookahead RMS 差异: (0 vs 2)=0.00224, (2 vs 4)=0.00179
# [2. block_frames= 1] latency=3008 samples, 稳态耗时=6.478 s, RTF=1.0371, vs block=1 max_diff=0.000e+00
# [2. block_frames= 4] latency=3008 samples, 稳态耗时=3.363 s, RTF=0.5383, vs block=1 max_diff=2.682e-07
# [2. block_frames=16] latency=3008 samples, 稳态耗时=2.408 s, RTF=0.3856, vs block=1 max_diff=3.055e-07
# [3. cache_frames=100 (window=0)] 实际 cache=100, 稳态耗时=2.349 s, RTF=0.3761
# [3. cache_frames=300 (window=0)] 实际 cache=300, 稳态耗时=2.350 s, RTF=0.3763
# [3. cache_frames=800 (window=0)] 实际 cache=800, 稳态耗时=2.460 s, RTF=0.3938
#    cache RMS 差异: (100 vs 800)=0.00116, (300 vs 800)=0.00043
# [4. 双轨滑窗对比] 18.0 s 长音频:
#    默认(600/200): 耗时=8.873 s, RTF=0.4929, 切轨边界帧=[600, 1200, 1800]
#    关闭(window=0): 耗时=9.255 s, RTF=0.5142, 切轨边界帧=[]
#    差异对比: 前6s RMS diff=0.00163, 后6s(12-18s) RMS diff=0.05232
# 已保存高阶参数对比音频至: /Users/boom/workspace/LightClear/explore/light_voice_isolator/outputs/mps/streaming_knobs
# 注：CPU 流式 RTF > 11，流式旋钮全量对比不建议在 CPU 运行。
