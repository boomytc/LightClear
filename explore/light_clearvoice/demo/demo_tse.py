import os
import platform
import sys
import time
from pathlib import Path

import cv2
import torch

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent.parent
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

# 全局关键参数
DEVICE = os.environ.get("LIGHTCLEAR_DEVICE", "auto")  # 可选: "auto" / "cuda" / "cuda:0" / "cpu"（不支持 MPS，指定 mps 将回退到 cpu）
TASK_NAME = "target_speaker_extraction"
MODEL_NAME = "AV_MossFormer2_TSE_16K"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT") or (Path.home() / "Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SE" / MODEL_NAME
INPUT_VIDEO = REPO_ROOT / "assets" / "video" / "speech" / "001.avi"
INPUT_VIDEO_DIR = REPO_ROOT / "assets" / "video" / "speech"
INPUT_VIDEO_SCP = REPO_ROOT / "assets" / "manifests" / "tse.scp"
OUTPUT_BASE = MODULE_ROOT / "outputs"

# 设备解析（不支持 MPS：auto 优先级 cuda > cpu，CUDA 不可用或指定 mps 时回退 cpu）
requested_device = DEVICE.strip().lower()
if requested_device == "auto":
    resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    print(f"[警告] {MODEL_NAME} 为音视频重链路且未做 MPS 适配，已自动回退到 cpu")
    resolved_device = "cpu"
elif requested_device == "cpu":
    resolved_device = "cpu"
else:
    raise ValueError(f"不支持的 DEVICE 配置: {DEVICE!r}，可选值为 auto / cuda / cuda:<idx> / cpu")

device_type = "cuda" if resolved_device.startswith("cuda") else resolved_device
cuda_index = int(resolved_device.split(":", 1)[1]) if device_type == "cuda" else None
if device_type == "cpu":
    print(f"[警告] 当前运行在 CPU，{MODEL_NAME} 为音视频重链路（含 S3FD 人脸检测与重封装），CPU 耗时较长，生产建议使用 CUDA")

OUTPUT_ROOT = OUTPUT_BASE / device_type / MODEL_NAME
OUTPUT_VIDEO = OUTPUT_ROOT / "single_video"
OUTPUT_VIDEO_DIR = OUTPUT_ROOT / "path_to_output_videos"
OUTPUT_VIDEO_SCP_DIR = OUTPUT_ROOT / "path_to_output_videos_scp"

# 模型加载
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(cuda_index)
    gpu_total_gib = torch.cuda.get_device_properties(cuda_index).total_memory / (1024 ** 3)
    torch.cuda.reset_peak_memory_stats(cuda_index)

t_load_start = time.perf_counter()
myClearVoice = ClearVoice(
    task=TASK_NAME,
    model_names=[MODEL_NAME],
    device=resolved_device,
    checkpoint_dir=str(CHECKPOINT_DIR),
)
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
load_elapsed_s = time.perf_counter() - t_load_start

# 单视频推理（CUDA 测冷启动+稳态，CPU 测单次）
cap = cv2.VideoCapture(str(INPUT_VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
cap.release()
duration_s = frame_count / fps

if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
    t_cold_start = time.perf_counter()
    myClearVoice(input_path=str(INPUT_VIDEO), online_write=True, output_path=str(OUTPUT_VIDEO))
    torch.cuda.synchronize(cuda_index)
    cold_elapsed_s = time.perf_counter() - t_cold_start

    t_warm_start = time.perf_counter()
    myClearVoice(input_path=str(INPUT_VIDEO), online_write=True, output_path=str(OUTPUT_VIDEO))
    torch.cuda.synchronize(cuda_index)
    warm_elapsed_s = time.perf_counter() - t_warm_start
else:
    t_infer_start = time.perf_counter()
    myClearVoice(input_path=str(INPUT_VIDEO), online_write=True, output_path=str(OUTPUT_VIDEO))
    infer_elapsed_s = time.perf_counter() - t_infer_start

# 目录与 SCP 批处理推理
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
t_dir_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_VIDEO_DIR), online_write=True, output_path=str(OUTPUT_VIDEO_DIR))
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
dir_elapsed_s = time.perf_counter() - t_dir_start

t_scp_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_VIDEO_SCP), online_write=True, output_path=str(OUTPUT_VIDEO_SCP_DIR))
if device_type == "cuda":
    torch.cuda.synchronize(cuda_index)
scp_elapsed_s = time.perf_counter() - t_scp_start

# 结果输出
if device_type == "cuda":
    peak_alloc_mib = torch.cuda.max_memory_allocated(cuda_index) / (1024 ** 2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(cuda_index) / (1024 ** 2)
    print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {myClearVoice.models[0].device}")
else:
    print(f"平台: {platform.system()} {platform.machine()}, PyTorch: {torch.__version__}, 设备: {myClearVoice.models[0].device}")
print(f"任务: {TASK_NAME}, 模型: {MODEL_NAME}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"单视频输入: {INPUT_VIDEO.name} ({fps:.2f} fps, {duration_s:.3f} s)")
if device_type == "cuda":
    print(f"冷启动端到端耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
    print(f"稳态端到端耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
else:
    print(f"端到端耗时: {infer_elapsed_s:.3f} s, RTF: {infer_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / infer_elapsed_s:.2f}x")
print(f"目录批处理耗时: {dir_elapsed_s:.3f} s, SCP 批处理耗时: {scp_elapsed_s:.3f} s")
if device_type == "cuda":
    print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")
print(f"单视频输出目录: {OUTPUT_VIDEO}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | CUDA）
# =============================================================================
# Running AV_MossFormer2_TSE_16K ...
# (...略去 S3FD 轨迹中间日志，完整目录共 3 条视频、SCP 共 2 条视频...)
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0
# 任务: target_speaker_extraction, 模型: AV_MossFormer2_TSE_16K, 加载耗时: 3.936 s
# 单视频输入: 001.avi (25.00 fps, 8.040 s)
# 冷启动端到端耗时: 24.888 s, RTF: 3.0955, 实时倍率: 0.32x
# 稳态端到端耗时: 21.805 s, RTF: 2.7121, 实时倍率: 0.37x
# 目录批处理耗时: 92.091 s, SCP 批处理耗时: 38.178 s
# 峰值显存: allocated=661.7 MiB, reserved=754.0 MiB
# 单视频输出目录: /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/single_video
# 注：本模型为 700 MB 音视频重链路（含 ffmpeg 解帧/重封装 + scenedetect + S3FD + AV-MossFormer2），仅建议 CUDA 生产部署。
