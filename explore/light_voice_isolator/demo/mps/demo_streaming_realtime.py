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
from inworld.voice_isolation import audio as audio_lib

# 全局关键参数
DEVICE = "mps"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
REALTIME_CHECKPOINT = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser_realtime.safetensors"
INPUT_AUDIO_A = REPO_ROOT / "assets" / "audio" / "noisy" / "p232_005.wav"
INPUT_AUDIO_B = REPO_ROOT / "assets" / "audio" / "noisy" / "p232_007.wav"
OUTPUT_DIR = MODULE_ROOT / "outputs" / "mps" / "streaming_realtime"
LOOKAHEAD_FRAMES = 2  # 2 * 10 ms = 20 ms
PIECE_20MS_SAMPLES = 960  # 20 ms @ 48 kHz
PIECE_IRREGULAR_SAMPLES = 333  # 非整帧片长（hop=480）

if not torch.backends.mps.is_available():
    raise RuntimeError("当前环境 MPS 不可用，请在 Apple Silicon macOS 上运行 demo/mps/ 脚本")
if not REALTIME_CHECKPOINT.is_file():
    raise FileNotFoundError(f"缺少实时流式模型权重: {REALTIME_CHECKPOINT}")
for audio_path in (INPUT_AUDIO_A, INPUT_AUDIO_B):
    if not audio_path.is_file():
        raise FileNotFoundError(f"缺少测试音频: {audio_path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型加载
t_load_start = time.perf_counter()
live = voice_isolation.VoiceIsolator(
    str(REALTIME_CHECKPOINT),
    device=DEVICE,
    mode="streaming",
    lookahead_frames=LOOKAHEAD_FRAMES,
)
torch.mps.synchronize()
load_elapsed_s = time.perf_counter() - t_load_start

# 输入构造（48 kHz 单声道）
wav_a, sr_a = voice_isolation.load_audio(str(INPUT_AUDIO_A))
wav_a = voice_isolation.resample(wav_a, sr_a, live.sample_rate)
wav_b, sr_b = voice_isolation.load_audio(str(INPUT_AUDIO_B))
wav_b = voice_isolation.resample(wav_b, sr_b, live.sample_rate)
duration_a_s = wav_a.shape[-1] / live.sample_rate
duration_b_s = wav_b.shape[-1] / live.sample_rate

# 1. 模拟麦克风 20 ms 分片逐块 push + flush（冷启动 + 稳态二次计时）
session_cold = live.stream()
print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 流式设备: {live.device}, 精度: {live.precision}, 加载耗时: {load_elapsed_s:.3f} s")
print(
    f"流式参数: lookahead_frames={live.lookahead_frames}, "
    f"latency_samples={session_cold.latency_samples}, "
    f"latency_seconds={session_cold.latency_seconds * 1000:.2f} ms"
)

torch.mps.synchronize()
t0 = time.perf_counter()
for start in range(0, wav_a.shape[-1], PIECE_20MS_SAMPLES):
    _ = session_cold.push(wav_a[start : start + PIECE_20MS_SAMPLES])
_ = session_cold.flush()
torch.mps.synchronize()
elapsed_20ms_cold = time.perf_counter() - t0

session_20ms = live.stream()
chunks_20ms = []
torch.mps.synchronize()
t0 = time.perf_counter()
for start in range(0, wav_a.shape[-1], PIECE_20MS_SAMPLES):
    chunks_20ms.append(session_20ms.push(wav_a[start : start + PIECE_20MS_SAMPLES]))
chunks_20ms.append(session_20ms.flush())
torch.mps.synchronize()
elapsed_20ms_warm = time.perf_counter() - t0
out_20ms = torch.cat(chunks_20ms)

# 2. 非整帧片长（333 samples）逐块 push + flush，验证输出一致性
session_irreg = live.stream()
chunks_irreg = []
torch.mps.synchronize()
t0 = time.perf_counter()
for start in range(0, wav_a.shape[-1], PIECE_IRREGULAR_SAMPLES):
    chunks_irreg.append(session_irreg.push(wav_a[start : start + PIECE_IRREGULAR_SAMPLES]))
chunks_irreg.append(session_irreg.flush())
torch.mps.synchronize()
elapsed_irreg = time.perf_counter() - t0
out_irreg = torch.cat(chunks_irreg)

piece_diff = float((out_20ms - out_irreg).abs().max())
print(
    f"[1. 分片一致性] 音频时长={duration_a_s:.3f} s: "
    f"20ms(960) 冷启动={elapsed_20ms_cold:.3f} s (RTF={elapsed_20ms_cold / duration_a_s:.4f}), "
    f"稳态={elapsed_20ms_warm:.3f} s (RTF={elapsed_20ms_warm / duration_a_s:.4f}, 实时倍率={duration_a_s / elapsed_20ms_warm:.2f}x), "
    f"非整帧(333) 稳态={elapsed_irreg:.3f} s (RTF={elapsed_irreg / duration_a_s:.4f}), "
    f"max_diff={piece_diff:.3e}"
)

# 3. 带 lengths 的多路并发流式 push([clips, samples])
len_a, len_b = int(wav_a.shape[-1]), int(wav_b.shape[-1])
max_len = max(len_a, len_b)
batch_wav = torch.stack([
    audio_lib.fit_length(wav_a, max_len),
    audio_lib.fit_length(wav_b, max_len),
])
session_batch = live.stream(lengths=[len_a, len_b])
batch_chunks = []
torch.mps.synchronize()
t0 = time.perf_counter()
for start in range(0, max_len, PIECE_20MS_SAMPLES):
    batch_chunks.append(session_batch.push(batch_wav[:, start : start + PIECE_20MS_SAMPLES]))
batch_chunks.append(session_batch.flush())
torch.mps.synchronize()
elapsed_batch = time.perf_counter() - t0
out_batch = torch.cat(batch_chunks, dim=1)

# 对比单路 B 的结果
session_b = live.stream()
chunks_b = []
for start in range(0, len_b, PIECE_20MS_SAMPLES):
    chunks_b.append(session_b.push(wav_b[start : start + PIECE_20MS_SAMPLES]))
chunks_b.append(session_b.flush())
out_b_single = torch.cat(chunks_b)

err_a = float((out_batch[0, :len_a] - out_20ms).abs().max())
err_b = float((out_batch[1, :len_b] - out_b_single).abs().max())
print(
    f"[2. 多路并发流式] lengths={[len_a, len_b]} (总 {duration_a_s + duration_b_s:.3f} s), "
    f"2 路并发耗时={elapsed_batch:.3f} s (总吞吐 RTF={elapsed_batch / (duration_a_s + duration_b_s):.4f}), "
    f"A 误差={err_a:.3e}, B 误差={err_b:.3e}"
)

# 结果输出
voice_isolation.save_audio(str(OUTPUT_DIR / f"{INPUT_AUDIO_A.stem}_stream_20ms.wav"), out_20ms, live.sample_rate, subtype="FLOAT")
voice_isolation.save_audio(str(OUTPUT_DIR / f"{INPUT_AUDIO_B.stem}_stream_batch.wav"), out_batch[1, :len_b], live.sample_rate, subtype="FLOAT")
print(f"已保存流式结果至: {OUTPUT_DIR}")

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.15.0, 流式设备: mps, 精度: fp32, 加载耗时: 0.463 s
# 流式参数: lookahead_frames=2, latency_samples=3008, latency_seconds=62.67 ms
# [1. 分片一致性] 音频时长=6.247 s: 20ms(960) 冷启动=4.452 s (RTF=0.7127), 稳态=4.552 s (RTF=0.7287, 实时倍率=1.37x), 非整帧(333) 稳态=7.611 s (RTF=1.2184), max_diff=2.757e-07
# [2. 多路并发流式] lengths=[299838, 189883] (总 10.203 s), 2 路并发耗时=6.579 s (总吞吐 RTF=0.6448), A 误差=8.941e-08, B 误差=5.960e-08
# 已保存流式结果至: /Users/boom/workspace/LightClear/explore/light_voice_isolator/outputs/mps/streaming_realtime
