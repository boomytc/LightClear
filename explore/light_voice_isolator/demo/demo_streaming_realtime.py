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
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
REALTIME_CHECKPOINT = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser_realtime.safetensors"
INPUT_AUDIO_A = REPO_ROOT / "assets" / "audio" / "noisy" / "p232_005.wav"
INPUT_AUDIO_B = REPO_ROOT / "assets" / "audio" / "noisy" / "p232_007.wav"
OUTPUT_BASE = MODULE_ROOT / "outputs"
LOOKAHEAD_FRAMES = 2  # 2 * 10 ms = 20 ms
PIECE_20MS_SAMPLES = 960  # 20 ms @ 48 kHz
PIECE_IRREGULAR_SAMPLES = 333  # 非整帧片长（hop=480）

if not REALTIME_CHECKPOINT.is_file():
    raise FileNotFoundError(f"缺少实时流式模型权重: {REALTIME_CHECKPOINT}")
for audio_path in (INPUT_AUDIO_A, INPUT_AUDIO_B):
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
    print("[警告] 当前运行在 CPU，Inworld Voice Isolator 20 ms 流式 RTF 约 11.2（不可实时，生产建议使用 MPS 或带 mamba_ssm 的 CUDA）")

OUTPUT_DIR = OUTPUT_BASE / device_type / "streaming_realtime"
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
    lookahead_frames=LOOKAHEAD_FRAMES,
)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
load_elapsed_s = time.perf_counter() - t_load_start

# 输入构造（48 kHz 单声道）
wav_a, sr_a = voice_isolation.load_audio(str(INPUT_AUDIO_A))
wav_a = voice_isolation.resample(wav_a, sr_a, live.sample_rate)
wav_b, sr_b = voice_isolation.load_audio(str(INPUT_AUDIO_B))
wav_b = voice_isolation.resample(wav_b, sr_b, live.sample_rate)
duration_a_s = wav_a.shape[-1] / live.sample_rate
duration_b_s = wav_b.shape[-1] / live.sample_rate

# 1. 模拟麦克风 20 ms 分片逐块 push + flush（CUDA/MPS 测冷启动+稳态，CPU 测单次）
session_info = live.stream()
if device_type == "cuda":
    print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 流式设备: {live.device}, 精度: {live.precision}")
    print(f"CUDA mamba_ssm 内核: {'命中' if cuda_kernel is not None else '未命中 (PyTorch 回退)'}, 加载耗时: {load_elapsed_s:.3f} s")
else:
    print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 流式设备: {live.device}, 精度: {live.precision}, 加载耗时: {load_elapsed_s:.3f} s")
print(
    f"流式参数: lookahead_frames={live.lookahead_frames}, "
    f"latency_samples={session_info.latency_samples}, "
    f"latency_seconds={session_info.latency_seconds * 1000:.2f} ms"
)

if device_type in ("cuda", "mps"):
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for start in range(0, wav_a.shape[-1], PIECE_20MS_SAMPLES):
        _ = session_info.push(wav_a[start : start + PIECE_20MS_SAMPLES])
    _ = session_info.flush()
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    elapsed_20ms_cold = time.perf_counter() - t0

session_20ms = live.stream()
chunks_20ms = []
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
for start in range(0, wav_a.shape[-1], PIECE_20MS_SAMPLES):
    chunks_20ms.append(session_20ms.push(wav_a[start : start + PIECE_20MS_SAMPLES]))
chunks_20ms.append(session_20ms.flush())
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
elapsed_20ms_warm = time.perf_counter() - t0
out_20ms = torch.cat(chunks_20ms)

# 2. 非整帧片长（333 samples）逐块 push + flush，验证输出一致性
session_irreg = live.stream()
chunks_irreg = []
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
for start in range(0, wav_a.shape[-1], PIECE_IRREGULAR_SAMPLES):
    chunks_irreg.append(session_irreg.push(wav_a[start : start + PIECE_IRREGULAR_SAMPLES]))
chunks_irreg.append(session_irreg.flush())
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
elapsed_irreg = time.perf_counter() - t0
out_irreg = torch.cat(chunks_irreg)

piece_diff = float((out_20ms - out_irreg).abs().max())
if device_type in ("cuda", "mps"):
    print(
        f"[1. 分片一致性] 音频时长={duration_a_s:.3f} s: "
        f"20ms(960) 冷启动={elapsed_20ms_cold:.3f} s (RTF={elapsed_20ms_cold / duration_a_s:.4f}), "
        f"稳态={elapsed_20ms_warm:.3f} s (RTF={elapsed_20ms_warm / duration_a_s:.4f}, 实时倍率={duration_a_s / elapsed_20ms_warm:.2f}x), "
        f"非整帧(333) 稳态={elapsed_irreg:.3f} s (RTF={elapsed_irreg / duration_a_s:.4f}), "
        f"max_diff={piece_diff:.3e}"
    )
else:
    print(
        f"[1. 分片一致性] 音频时长={duration_a_s:.3f} s: "
        f"20ms(960) 耗时={elapsed_20ms_warm:.3f} s (RTF={elapsed_20ms_warm / duration_a_s:.4f}, 实时倍率={duration_a_s / elapsed_20ms_warm:.2f}x), "
        f"非整帧(333) 耗时={elapsed_irreg:.3f} s (RTF={elapsed_irreg / duration_a_s:.4f}), "
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
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
t0 = time.perf_counter()
for start in range(0, max_len, PIECE_20MS_SAMPLES):
    batch_chunks.append(session_batch.push(batch_wav[:, start : start + PIECE_20MS_SAMPLES]))
batch_chunks.append(session_batch.flush())
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
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
if device_type == "cuda":
    peak_alloc_mib = torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 2)
    print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")

# 结果输出
voice_isolation.save_audio(str(OUTPUT_DIR / f"{INPUT_AUDIO_A.stem}_stream_20ms.wav"), out_20ms, live.sample_rate, subtype="FLOAT")
voice_isolation.save_audio(str(OUTPUT_DIR / f"{INPUT_AUDIO_B.stem}_stream_batch.wav"), out_batch[1, :len_b], live.sample_rate, subtype="FLOAT")
print(f"已保存流式结果至: {OUTPUT_DIR}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | CUDA）
# =============================================================================
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 流式设备: cuda:0, 精度: bf16
# CUDA mamba_ssm 内核: 未命中 (PyTorch 回退), 加载耗时: 0.710 s
# 流式参数: lookahead_frames=2, latency_samples=3008, latency_seconds=62.67 ms
# [1. 分片一致性] 音频时长=6.247 s: 20ms(960) 冷启动=11.790 s (RTF=1.8875), 稳态=11.423 s (RTF=1.8287, 实时倍率=0.55x), 非整帧(333) 稳态=22.173 s (RTF=3.5496), max_diff=5.611e-04
# [2. 多路并发流式] lengths=[299838, 189883] (总 10.203 s), 2 路并发耗时=11.592 s (总吞吐 RTF=1.1362), A 误差=9.492e-04, B 误差=4.967e-04
# 峰值显存: allocated=2519.5 MiB, reserved=2782.0 MiB
# 已保存流式结果至: /content/LightClear/explore/light_voice_isolator/outputs/cuda/streaming_realtime

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.15.0, 流式设备: mps, 精度: fp32, 加载耗时: 0.463 s
# 流式参数: lookahead_frames=2, latency_samples=3008, latency_seconds=62.67 ms
# [1. 分片一致性] 音频时长=6.247 s: 20ms(960) 冷启动=4.452 s (RTF=0.7127), 稳态=4.552 s (RTF=0.7287, 实时倍率=1.37x), 非整帧(333) 稳态=7.611 s (RTF=1.2184), max_diff=2.757e-07
# [2. 多路并发流式] lengths=[299838, 189883] (总 10.203 s), 2 路并发耗时=6.579 s (总吞吐 RTF=0.6448), A 误差=8.941e-08, B 误差=5.960e-08
# 已保存流式结果至: /Users/boom/workspace/LightClear/explore/light_voice_isolator/outputs/mps/streaming_realtime
# 注：Apple M4 Pro CPU（限 2 线程）20 ms 流式 RTF 约 11.18，不具备实时能力。
