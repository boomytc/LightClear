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
from inworld.voice_isolation import scan

# 全局关键参数
DEVICE = "cuda:0"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
REALTIME_CHECKPOINT = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser_realtime.safetensors"
SHORT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "p232_005.wav"
LONG_AUDIO = REPO_ROOT / "assets" / "audio" / "music" / "next_station_heaven.mp3"
OUTPUT_DIR = MODULE_ROOT / "outputs" / "cuda" / "streaming_knobs"
PIECE_SAMPLES = 9600  # 200 ms 分片模拟连续推流
LONG_CLIP_SECONDS = 18.0

if not torch.cuda.is_available():
    raise RuntimeError("当前环境 CUDA 不可用，请在 NVIDIA GPU 环境运行 demo/cuda/ 脚本")
if not REALTIME_CHECKPOINT.is_file():
    raise FileNotFoundError(f"缺少实时流式模型权重: {REALTIME_CHECKPOINT}")
for audio_path in (SHORT_AUDIO, LONG_AUDIO):
    if not audio_path.is_file():
        raise FileNotFoundError(f"缺少测试音频: {audio_path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
gpu_name = torch.cuda.get_device_name(0)
gpu_total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
cuda_kernel = scan.kernel()

# 模型加载
torch.cuda.reset_peak_memory_stats(0)
t_load_start = time.perf_counter()
live = voice_isolation.VoiceIsolator(
    str(REALTIME_CHECKPOINT),
    device=DEVICE,
    mode="streaming",
    lookahead_frames=2,
)
torch.cuda.synchronize(0)
load_elapsed_s = time.perf_counter() - t_load_start
print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {live.device}, 精度: {live.precision}")
print(f"CUDA mamba_ssm 内核: {'命中' if cuda_kernel is not None else '未命中 (PyTorch 回退)'}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"支持的 lookahead_choices={live.lookahead_choices}")

# 输入构造
wav_short, sr_short = voice_isolation.load_audio(str(SHORT_AUDIO))
wav_short = voice_isolation.resample(wav_short, sr_short, live.sample_rate)
short_dur_s = wav_short.shape[-1] / live.sample_rate

wav_long, sr_long = voice_isolation.load_audio(str(LONG_AUDIO))
wav_long = voice_isolation.resample(wav_long, sr_long, live.sample_rate)[: int(LONG_CLIP_SECONDS * live.sample_rate)]
long_dur_s = wav_long.shape[-1] / live.sample_rate

# 预热（冷启动计时）
warmup_session = live.stream()
torch.cuda.synchronize(0)
t0 = time.perf_counter()
for i in range(0, wav_short.shape[-1], PIECE_SAMPLES):
    _ = warmup_session.push(wav_short[i : i + PIECE_SAMPLES])
_ = warmup_session.flush()
torch.cuda.synchronize(0)
t_cold_warmup = time.perf_counter() - t0
print(f"[0. 冷启动预热] 短音频 ({short_dur_s:.3f} s) 首次流式耗时={t_cold_warmup:.3f} s (RTF={t_cold_warmup / short_dur_s:.4f})")

# 1. lookahead_frames = 0 / 2 / 4
lookahead_outputs = {}
for la in live.lookahead_choices:
    live.lookahead_frames = la
    session = live.stream()
    torch.cuda.synchronize(0)
    t0 = time.perf_counter()
    parts = [session.push(wav_short[i : i + PIECE_SAMPLES]) for i in range(0, wav_short.shape[-1], PIECE_SAMPLES)]
    parts.append(session.flush())
    torch.cuda.synchronize(0)
    dt = time.perf_counter() - t0
    out_la = torch.cat(parts)
    lookahead_outputs[la] = out_la
    print(
        f"[1. lookahead={la}] 算法延迟={session.latency_samples} samples ({session.latency_seconds * 1000:.2f} ms), "
        f"稳态耗时={dt:.3f} s, RTF={dt / short_dur_s:.4f}"
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
    torch.cuda.synchronize(0)
    t0 = time.perf_counter()
    parts = [session.push(wav_short[i : i + PIECE_SAMPLES]) for i in range(0, wav_short.shape[-1], PIECE_SAMPLES)]
    parts.append(session.flush())
    torch.cuda.synchronize(0)
    dt = time.perf_counter() - t0
    out_bf = torch.cat(parts)
    if block_ref is None:
        block_ref = out_bf
    diff_bf = float((out_bf - block_ref).abs().max())
    print(
        f"[2. block_frames={bf:2d}] latency={session.latency_samples} samples, "
        f"稳态耗时={dt:.3f} s, RTF={dt / short_dur_s:.4f}, vs block=1 max_diff={diff_bf:.3e}"
    )

# 3. cache_frames = 100 / 300 / 800
cache_outputs = {}
for cf in (100, 300, 800):
    session = live.stream(cache_frames=cf, window_frames=0)
    torch.cuda.synchronize(0)
    t0 = time.perf_counter()
    parts = [session.push(wav_short[i : i + PIECE_SAMPLES]) for i in range(0, wav_short.shape[-1], PIECE_SAMPLES)]
    parts.append(session.flush())
    torch.cuda.synchronize(0)
    dt = time.perf_counter() - t0
    cache_outputs[cf] = torch.cat(parts)
    print(
        f"[3. cache_frames={cf:3d} (window=0)] 实际 cache={session.cache_frames}, "
        f"稳态耗时={dt:.3f} s, RTF={dt / short_dur_s:.4f}"
    )
print(
    f"   cache RMS 差异: (100 vs 800)={float((cache_outputs[100] - cache_outputs[800]).pow(2).mean().sqrt()):.5f}, "
    f"(300 vs 800)={float((cache_outputs[300] - cache_outputs[800]).pow(2).mean().sqrt()):.5f}"
)

# 4. 双轨滑窗 window_frames/crossfade_frames：600/200 vs 0
session_win = live.stream(window_frames=600, crossfade_frames=200)
torch.cuda.synchronize(0)
t0 = time.perf_counter()
parts_win = [session_win.push(wav_long[i : i + PIECE_SAMPLES]) for i in range(0, wav_long.shape[-1], PIECE_SAMPLES)]
parts_win.append(session_win.flush())
torch.cuda.synchronize(0)
t_win = time.perf_counter() - t0
out_win = torch.cat(parts_win)

session_nowin = live.stream(window_frames=0)
torch.cuda.synchronize(0)
t0 = time.perf_counter()
parts_nowin = [session_nowin.push(wav_long[i : i + PIECE_SAMPLES]) for i in range(0, wav_long.shape[-1], PIECE_SAMPLES)]
parts_nowin.append(session_nowin.flush())
torch.cuda.synchronize(0)
t_nowin = time.perf_counter() - t0
out_nowin = torch.cat(parts_nowin)

seg_early = slice(0, int(6.0 * live.sample_rate))
seg_late = slice(int(12.0 * live.sample_rate), int(18.0 * live.sample_rate))
early_rms_diff = float((out_win[seg_early] - out_nowin[seg_early]).pow(2).mean().sqrt())
late_rms_diff = float((out_win[seg_late] - out_nowin[seg_late]).pow(2).mean().sqrt())
peak_alloc_mib = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
peak_reserved_mib = torch.cuda.max_memory_reserved(0) / (1024 ** 2)
print(
    f"[4. 双轨滑窗对比] {long_dur_s:.1f} s 长音频:\n"
    f"   默认(600/200): 耗时={t_win:.3f} s, RTF={t_win / long_dur_s:.4f}, 切轨边界帧={session_win.boundaries}\n"
    f"   关闭(window=0): 耗时={t_nowin:.3f} s, RTF={t_nowin / long_dur_s:.4f}, 切轨边界帧={session_nowin.boundaries}\n"
    f"   差异对比: 前6s RMS diff={early_rms_diff:.5f}, 后6s(12-18s) RMS diff={late_rms_diff:.5f}"
)
print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")

voice_isolation.save_audio(str(OUTPUT_DIR / "long_windowed_600_200.wav"), out_win, live.sample_rate, subtype="FLOAT")
voice_isolation.save_audio(str(OUTPUT_DIR / "long_unwindowed_0.wav"), out_nowin, live.sample_rate, subtype="FLOAT")
print(f"已保存高阶参数对比音频至: {OUTPUT_DIR}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | Linux-6.6.159+-x86_64-with-glibc2.39）
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
