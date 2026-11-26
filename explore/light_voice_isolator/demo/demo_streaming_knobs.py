import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

import torch
from inworld import voice_isolation

# 全局关键参数
REALTIME_CHECKPOINT = Path("/Users/boom/Model/SE/inworld-voice-isolator/denoiser_realtime.safetensors")
SHORT_AUDIO = Path("/Users/boom/workspace/LightClear/assets/audio/noisy/p232_005.wav")
LONG_AUDIO = Path("/Users/boom/workspace/LightClear/assets/audio/music/next_station_heaven.mp3")
OUTPUT_DIR = MODULE_ROOT / "outputs" / "streaming_knobs"
PIECE_SAMPLES = 9600  # 200 ms 分片模拟连续推流
LONG_CLIP_SECONDS = 18.0

if not REALTIME_CHECKPOINT.is_file():
    raise FileNotFoundError(f"缺少实时流式模型权重: {REALTIME_CHECKPOINT}")
for audio_path in (SHORT_AUDIO, LONG_AUDIO):
    if not audio_path.is_file():
        raise FileNotFoundError(f"缺少测试音频: {audio_path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型加载（单实例可动态切换 lookahead_frames）
live = voice_isolation.VoiceIsolator(
    str(REALTIME_CHECKPOINT),
    mode="streaming",
    lookahead_frames=2,
)
print(f"设备: {live.device}, 支持的 lookahead_choices={live.lookahead_choices}")

# 输入构造
wav_short, sr_short = voice_isolation.load_audio(str(SHORT_AUDIO))
wav_short = voice_isolation.resample(wav_short, sr_short, live.sample_rate)
short_dur_s = wav_short.shape[-1] / live.sample_rate

wav_long, sr_long = voice_isolation.load_audio(str(LONG_AUDIO))
wav_long = voice_isolation.resample(wav_long, sr_long, live.sample_rate)[: int(LONG_CLIP_SECONDS * live.sample_rate)]
long_dur_s = wav_long.shape[-1] / live.sample_rate

# 1. lookahead_frames = 0 / 2 / 4（0 / 20 / 40 ms）
lookahead_outputs = {}
for la in live.lookahead_choices:
    live.lookahead_frames = la
    session = live.stream()
    t0 = time.perf_counter()
    parts = [session.push(wav_short[i : i + PIECE_SAMPLES]) for i in range(0, wav_short.shape[-1], PIECE_SAMPLES)]
    parts.append(session.flush())
    if live.device.type == "mps":
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

# 2. block_frames = 1 / 4 / 16（吞吐旋钮，不改变算法延迟）
live.lookahead_frames = 2
block_ref = None
for bf in (1, 4, 16):
    session = live.stream(block_frames=bf)
    t0 = time.perf_counter()
    parts = [session.push(wav_short[i : i + PIECE_SAMPLES]) for i in range(0, wav_short.shape[-1], PIECE_SAMPLES)]
    parts.append(session.flush())
    if live.device.type == "mps":
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

# 3. cache_frames = 100 / 300 / 800（在关闭双轨滑窗时观察注意力缓存长度影响）
cache_outputs = {}
for cf in (100, 300, 800):
    session = live.stream(cache_frames=cf, window_frames=0)
    t0 = time.perf_counter()
    parts = [session.push(wav_short[i : i + PIECE_SAMPLES]) for i in range(0, wav_short.shape[-1], PIECE_SAMPLES)]
    parts.append(session.flush())
    if live.device.type == "mps":
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

# 4. 双轨滑窗 window_frames/crossfade_frames：默认 600/200 (6s/2s) vs 0 (关闭) 在 18 s 长音频上的对比
session_win = live.stream(window_frames=600, crossfade_frames=200)
t0 = time.perf_counter()
parts_win = [session_win.push(wav_long[i : i + PIECE_SAMPLES]) for i in range(0, wav_long.shape[-1], PIECE_SAMPLES)]
parts_win.append(session_win.flush())
if live.device.type == "mps":
    torch.mps.synchronize()
t_win = time.perf_counter() - t0
out_win = torch.cat(parts_win)

session_nowin = live.stream(window_frames=0)
t0 = time.perf_counter()
parts_nowin = [session_nowin.push(wav_long[i : i + PIECE_SAMPLES]) for i in range(0, wav_long.shape[-1], PIECE_SAMPLES)]
parts_nowin.append(session_nowin.flush())
if live.device.type == "mps":
    torch.mps.synchronize()
t_nowin = time.perf_counter() - t0
out_nowin = torch.cat(parts_nowin)

# 分段计算前 6 s（第一个窗口内）与后 6 s（12~18 s 跨多窗口后）的差异
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

voice_isolation.save_audio(str(OUTPUT_DIR / "long_windowed_600_200.wav"), out_win, live.sample_rate, subtype="FLOAT")
voice_isolation.save_audio(str(OUTPUT_DIR / "long_unwindowed_0.wav"), out_nowin, live.sample_rate, subtype="FLOAT")
print(f"已保存高阶参数对比音频至: {OUTPUT_DIR}")
