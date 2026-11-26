import sys
import time
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

import torch
from inworld import voice_isolation
from inworld.voice_isolation import audio as audio_lib

# 全局关键参数
OFFLINE_CHECKPOINT = Path("/Users/boom/Model/SE/inworld-voice-isolator/denoiser.safetensors")
REALTIME_CHECKPOINT = Path("/Users/boom/Model/SE/inworld-voice-isolator/denoiser_realtime.safetensors")
CLIP_FILES = [
    Path("/Users/boom/workspace/LightClear/assets/audio/noisy/p232_005.wav"),
    Path("/Users/boom/workspace/LightClear/assets/audio/noisy/p232_007.wav"),
    Path("/Users/boom/workspace/LightClear/assets/audio/noisy/test.wav"),
]
OUTPUT_DIR = MODULE_ROOT / "outputs" / "batch_many"
SAMPLE_RATE = 48000
EQUAL_LEN_SECONDS = 2.0

for ckpt in (OFFLINE_CHECKPOINT, REALTIME_CHECKPOINT):
    if not ckpt.is_file():
        raise FileNotFoundError(f"缺少模型权重: {ckpt}")
for audio_path in CLIP_FILES:
    if not audio_path.is_file():
        raise FileNotFoundError(f"缺少测试音频: {audio_path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型加载
offline = voice_isolation.VoiceIsolator(str(OFFLINE_CHECKPOINT), mode="offline")
streaming = voice_isolation.VoiceIsolator(
    str(REALTIME_CHECKPOINT),
    mode="streaming",
    lookahead_frames=2,
)

# 输入构造（统一 48 kHz 单声道）
raw_clips = []
for audio_path in CLIP_FILES:
    wav, sr = voice_isolation.load_audio(str(audio_path))
    raw_clips.append(voice_isolation.resample(wav, sr, SAMPLE_RATE))

equal_n = int(EQUAL_LEN_SECONDS * SAMPLE_RATE)
equal_clips = [audio_lib.fit_length(c, equal_n) for c in raw_clips]
mixed_clips = [raw_clips[0][:48000], raw_clips[1], raw_clips[2][:72000], raw_clips[0][:48000]]

# 1. 离线同长合批 vs 逐条 isolate()
t0 = time.perf_counter()
offline_equal_batch = offline.isolate_many(equal_clips, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
if offline.device.type == "mps":
    torch.mps.synchronize()
t_equal_batch = time.perf_counter() - t0

t0 = time.perf_counter()
offline_equal_single = [
    offline.isolate(c, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE) for c in equal_clips
]
if offline.device.type == "mps":
    torch.mps.synchronize()
t_equal_single = time.perf_counter() - t0

equal_errs = [
    float((b - s).abs().max()) for b, s in zip(offline_equal_batch, offline_equal_single)
]
print(
    f"[1. 离线同长合批] {len(equal_clips)} 条 x {EQUAL_LEN_SECONDS} s: "
    f"批处理={t_equal_batch:.3f} s, 逐条={t_equal_single:.3f} s, "
    f"逐条最大误差={[round(e, 7) for e in equal_errs]}"
)

# 2. 离线异长自动分组并按原顺序返回
t0 = time.perf_counter()
offline_mixed_batch = offline.isolate_many(mixed_clips, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
if offline.device.type == "mps":
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
    f"[2. 离线异长分组] 输入长度={mixed_lens_in} -> 输出长度={mixed_lens_out} (保持原序), "
    f"耗时={t_mixed_batch:.3f} s, 最大误差={[round(e, 7) for e in mixed_errs]}"
)

# 3. 流式多条音频 lockstep 并发批流式隔离 vs 单条 streaming isolate()
t0 = time.perf_counter()
stream_batch_out = streaming.isolate_many(mixed_clips, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE)
if streaming.device.type == "mps":
    torch.mps.synchronize()
t_stream_batch = time.perf_counter() - t0

t0 = time.perf_counter()
stream_single_out = [
    streaming.isolate(c, SAMPLE_RATE, output_sample_rate=SAMPLE_RATE) for c in mixed_clips
]
if streaming.device.type == "mps":
    torch.mps.synchronize()
t_stream_single = time.perf_counter() - t0

stream_errs = [
    float((b - s).abs().max()) for b, s in zip(stream_batch_out, stream_single_out)
]
print(
    f"[3. 流式 lockstep 批处理] {len(mixed_clips)} 条异长音频: "
    f"批流式={t_stream_batch:.3f} s vs 逐条流式={t_stream_single:.3f} s, "
    f"逐条最大误差={[round(e, 7) for e in stream_errs]}"
)

# 结果输出
for idx, wav_out in enumerate(stream_batch_out):
    out_path = OUTPUT_DIR / f"stream_batch_{idx}.wav"
    voice_isolation.save_audio(str(out_path), wav_out, SAMPLE_RATE, subtype="FLOAT")
print(f"已将流式批处理结果写入: {OUTPUT_DIR}")
