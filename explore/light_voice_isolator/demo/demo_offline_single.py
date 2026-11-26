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
CHECKPOINT_PATH = MODEL_ROOT / "SE" / "inworld-voice-isolator" / "denoiser.safetensors"
INPUT_AUDIO = REPO_ROOT / "assets" / "audio" / "noisy" / "p232_005.wav"
OUTPUT_BASE = MODULE_ROOT / "outputs"
OUTPUT_SAMPLE_RATE = 48000

if not CHECKPOINT_PATH.is_file():
    raise FileNotFoundError(f"缺少离线模型权重，请先下载到中心目录: {CHECKPOINT_PATH}")
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
    print("[警告] 当前运行在 CPU，Inworld Voice Isolator 离线 RTF 约 4.3（生产建议使用 CUDA 或 MPS）")

OUTPUT_DIR = OUTPUT_BASE / device_type / "offline_single"
OUTPUT_VOICE = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_voice.wav"
OUTPUT_BACKGROUND = OUTPUT_DIR / f"{INPUT_AUDIO.stem}_background.wav"

# 模型加载
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(cuda_index)
    gpu_total_gib = torch.cuda.get_device_properties(cuda_index).total_memory / (1024 ** 3)
    cuda_kernel = scan.kernel()
    torch.cuda.reset_peak_memory_stats(cuda_index)

t_load_start = time.perf_counter()
isolator = voice_isolation.VoiceIsolator(
    str(CHECKPOINT_PATH),
    device=resolved_device,
    mode="offline",
)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
elif device_type == "mps":
    torch.mps.synchronize()
    metal_kernel = scan.metal_kernel()
load_elapsed_s = time.perf_counter() - t_load_start

# 输入构造
wav, input_sr = voice_isolation.load_audio(str(INPUT_AUDIO))
duration_s = wav.shape[-1] / input_sr
aligned_mono = voice_isolation.resample(wav, input_sr, OUTPUT_SAMPLE_RATE)

# 推理调用（CUDA/MPS 测冷启动+稳态，CPU 测单次）
if device_type in ("cuda", "mps"):
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    t_cold_start = time.perf_counter()
    _ = isolator.isolate(wav, input_sr, output_sample_rate=OUTPUT_SAMPLE_RATE)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    cold_elapsed_s = time.perf_counter() - t_cold_start

    t_warm_start = time.perf_counter()
    voice = isolator.isolate(wav, input_sr, output_sample_rate=OUTPUT_SAMPLE_RATE)
    if device_type == "cuda":
        torch.cuda.synchronize(cuda_index)
    else:
        torch.mps.synchronize()
    warm_elapsed_s = time.perf_counter() - t_warm_start
else:
    t_infer_start = time.perf_counter()
    voice = isolator.isolate(wav, input_sr, output_sample_rate=OUTPUT_SAMPLE_RATE)
    infer_elapsed_s = time.perf_counter() - t_infer_start

aligned_mono = audio_lib.fit_length(aligned_mono, voice.shape[-1])
background = aligned_mono - voice

# 结果输出与数值校验
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
voice_isolation.save_audio(str(OUTPUT_VOICE), voice, OUTPUT_SAMPLE_RATE, subtype="FLOAT")
voice_isolation.save_audio(str(OUTPUT_BACKGROUND), background, OUTPUT_SAMPLE_RATE, subtype="FLOAT")

reconstructed = voice + background
recon_max_err = float((aligned_mono - reconstructed).abs().max())
input_rms = float(aligned_mono.pow(2).mean().sqrt())
voice_rms = float(voice.pow(2).mean().sqrt())
bg_rms = float(background.pow(2).mean().sqrt())
input_peak = float(aligned_mono.abs().max())
voice_peak = float(voice.abs().max())

if device_type == "cuda":
    peak_alloc_mib = torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 2)
    print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {isolator.device}, 精度: {isolator.precision}")
    print(f"CUDA mamba_ssm 内核: {'命中' if cuda_kernel is not None else '未命中 (PyTorch 回退)'}, 加载耗时: {load_elapsed_s:.3f} s")
elif device_type == "mps":
    print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {isolator.device}, 精度: {isolator.precision}")
    print(f"Metal Shader 内核: {'命中 (已编译)' if metal_kernel is not None else '未命中 (PyTorch 回退)'}, 加载耗时: {load_elapsed_s:.3f} s")
else:
    print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {isolator.device}, 精度: {isolator.precision}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"输入文件: {INPUT_AUDIO.name} ({input_sr} Hz, {duration_s:.3f} s)")
if device_type in ("cuda", "mps"):
    print(f"冷启动推理耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
    print(f"稳态推理耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
else:
    print(f"推理耗时: {infer_elapsed_s:.3f} s, RTF: {infer_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / infer_elapsed_s:.2f}x")
if device_type == "cuda":
    print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")
print(f"电平保持: input_rms={input_rms:.5f}, voice_rms={voice_rms:.5f}, bg_rms={bg_rms:.5f}")
print(f"峰值统计: input_peak={input_peak:.5f}, voice_peak={voice_peak:.5f}")
print(f"重构最大误差 max(|input - (voice + background)|): {recon_max_err:.3e}")
print(f"已保存人声: {OUTPUT_VOICE}")
print(f"已保存残差: {OUTPUT_BACKGROUND}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | CUDA）
# =============================================================================
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0, 精度: bf16
# CUDA mamba_ssm 内核: 未命中 (PyTorch 回退), 加载耗时: 0.773 s
# 输入文件: p232_005.wav (48000 Hz, 6.247 s)
# 冷启动推理耗时: 2.030 s, RTF: 0.3250, 实时倍率: 3.08x
# 稳态推理耗时: 1.384 s, RTF: 0.2216, 实时倍率: 4.51x
# 峰值显存: allocated=5150.1 MiB, reserved=5716.0 MiB
# 电平保持: input_rms=0.10430, voice_rms=0.08070, bg_rms=0.06511
# 峰值统计: input_peak=0.69562, voice_peak=0.50109
# 重构最大误差 max(|input - (voice + background)|): 0.000e+00
# 已保存人声: /content/LightClear/explore/light_voice_isolator/outputs/cuda/offline_single/p232_005_voice.wav
# 已保存残差: /content/LightClear/explore/light_voice_isolator/outputs/cuda/offline_single/p232_005_background.wav

# =============================================================================
# 实测输出（2026-11-26 | Apple M4 Pro | macOS 27.2.1 (arm64) | MPS）
# =============================================================================
# 平台: Darwin arm64, PyTorch: 2.15.0, 设备: mps, 精度: fp32
# Metal Shader 内核: 命中 (已编译), 加载耗时: 0.537 s
# 输入文件: p232_005.wav (48000 Hz, 6.247 s)
# 冷启动推理耗时: 1.824 s, RTF: 0.2920, 实时倍率: 3.42x
# 稳态推理耗时: 1.242 s, RTF: 0.1988, 实时倍率: 5.03x
# 电平保持: input_rms=0.10430, voice_rms=0.08065, bg_rms=0.06507
# 峰值统计: input_peak=0.69562, voice_peak=0.50109
# 重构最大误差 max(|input - (voice + background)|): 0.000e+00
# 已保存人声: /Users/boom/workspace/LightClear/explore/light_voice_isolator/outputs/mps/offline_single/p232_005_voice.wav
# 已保存残差: /Users/boom/workspace/LightClear/explore/light_voice_isolator/outputs/mps/offline_single/p232_005_background.wav
# 注：Apple M4 Pro CPU（限 2 线程）离线实测 ~28.3 s（RTF 4.31），仅作无加速器兜底，不建议生产使用。
