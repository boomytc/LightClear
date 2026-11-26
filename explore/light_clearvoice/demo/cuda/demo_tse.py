import os
import sys
import time
from pathlib import Path

import cv2
import torch

MODULE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = MODULE_ROOT.parent.parent
THIRD_PARTY_DIR = MODULE_ROOT / "third_party"
if str(THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_DIR))

from clearvoice import ClearVoice

# 全局关键参数
DEVICE = "cuda:0"
TASK_NAME = "target_speaker_extraction"
MODEL_NAME = "AV_MossFormer2_TSE_16K"
MODEL_ROOT = Path(os.environ.get("LIGHTCLEAR_MODEL_ROOT", "/Users/boom/Model"))
CHECKPOINT_DIR = MODEL_ROOT / "SE" / MODEL_NAME
INPUT_VIDEO = REPO_ROOT / "assets" / "video" / "speech" / "001.avi"
INPUT_VIDEO_DIR = REPO_ROOT / "assets" / "video" / "speech"
INPUT_VIDEO_SCP = REPO_ROOT / "assets" / "manifests" / "tse.scp"
OUTPUT_ROOT = MODULE_ROOT / "outputs" / "cuda" / MODEL_NAME
OUTPUT_VIDEO = OUTPUT_ROOT / "single_video"
OUTPUT_VIDEO_DIR = OUTPUT_ROOT / "path_to_output_videos"
OUTPUT_VIDEO_SCP_DIR = OUTPUT_ROOT / "path_to_output_videos_scp"

if not torch.cuda.is_available():
    raise RuntimeError("当前环境 CUDA 不可用，请在 NVIDIA GPU 环境运行 demo/cuda/ 脚本")

gpu_name = torch.cuda.get_device_name(0)
gpu_total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

# 模型加载
torch.cuda.reset_peak_memory_stats(0)
t_load_start = time.perf_counter()
myClearVoice = ClearVoice(
    task=TASK_NAME,
    model_names=[MODEL_NAME],
    device=DEVICE,
    checkpoint_dir=str(CHECKPOINT_DIR),
)
torch.cuda.synchronize(0)
load_elapsed_s = time.perf_counter() - t_load_start

# 单视频推理（冷启动 + 稳态二次计时）
cap = cv2.VideoCapture(str(INPUT_VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
cap.release()
duration_s = frame_count / fps

torch.cuda.synchronize(0)
t_cold_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_VIDEO), online_write=True, output_path=str(OUTPUT_VIDEO))
torch.cuda.synchronize(0)
cold_elapsed_s = time.perf_counter() - t_cold_start

torch.cuda.synchronize(0)
t_warm_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_VIDEO), online_write=True, output_path=str(OUTPUT_VIDEO))
torch.cuda.synchronize(0)
warm_elapsed_s = time.perf_counter() - t_warm_start

# 目录与 SCP 批处理推理
torch.cuda.synchronize(0)
t_dir_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_VIDEO_DIR), online_write=True, output_path=str(OUTPUT_VIDEO_DIR))
torch.cuda.synchronize(0)
dir_elapsed_s = time.perf_counter() - t_dir_start

torch.cuda.synchronize(0)
t_scp_start = time.perf_counter()
myClearVoice(input_path=str(INPUT_VIDEO_SCP), online_write=True, output_path=str(OUTPUT_VIDEO_SCP_DIR))
torch.cuda.synchronize(0)
scp_elapsed_s = time.perf_counter() - t_scp_start

peak_alloc_mib = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
peak_reserved_mib = torch.cuda.max_memory_reserved(0) / (1024 ** 2)

# 结果输出
print(f"GPU: {gpu_name} ({gpu_total_gib:.2f} GiB), PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, 设备: {myClearVoice.models[0].device}")
print(f"任务: {TASK_NAME}, 模型: {MODEL_NAME}, 加载耗时: {load_elapsed_s:.3f} s")
print(f"单视频输入: {INPUT_VIDEO.name} ({fps:.2f} fps, {duration_s:.3f} s)")
print(f"冷启动端到端耗时: {cold_elapsed_s:.3f} s, RTF: {cold_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / cold_elapsed_s:.2f}x")
print(f"稳态端到端耗时: {warm_elapsed_s:.3f} s, RTF: {warm_elapsed_s / duration_s:.4f}, 实时倍率: {duration_s / warm_elapsed_s:.2f}x")
print(f"目录批处理耗时: {dir_elapsed_s:.3f} s, SCP 批处理耗时: {scp_elapsed_s:.3f} s")
print(f"峰值显存: allocated={peak_alloc_mib:.1f} MiB, reserved={peak_reserved_mib:.1f} MiB")
print(f"单视频输出目录: {OUTPUT_VIDEO}")

# =============================================================================
# 实测输出（2026-11-26 | Colab NVIDIA L4, 23034 MiB, 580.82.07 | Linux-6.6.159+-x86_64-with-glibc2.39）
# =============================================================================
# Running AV_MossFormer2_TSE_16K ...
# torch.Size([1, 128732])
# torch.Size([1, 202, 112, 112])
# ********
# torch.Size([1, 128732])
# torch.Size([1, 202, 112, 112])
# ********
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/single_video/AV_MossFormer2_TSE_16K/001/py_faceTracks/00000.avi
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/single_video/AV_MossFormer2_TSE_16K/001/py_faceTracks/00001.avi
# Running AV_MossFormer2_TSE_16K ...
# torch.Size([1, 128732])
# torch.Size([1, 202, 112, 112])
# ********
# torch.Size([1, 128732])
# torch.Size([1, 202, 112, 112])
# ********
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/single_video/AV_MossFormer2_TSE_16K/001/py_faceTracks/00000.avi
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/single_video/AV_MossFormer2_TSE_16K/001/py_faceTracks/00001.avi
# Running AV_MossFormer2_TSE_16K ...
# torch.Size([1, 128732])
# torch.Size([1, 202, 112, 112])
# ********
# torch.Size([1, 128732])
# torch.Size([1, 202, 112, 112])
# ********
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos/AV_MossFormer2_TSE_16K/001/py_faceTracks/00000.avi
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos/AV_MossFormer2_TSE_16K/001/py_faceTracks/00001.avi
# torch.Size([1, 94720])
# torch.Size([1, 148, 112, 112])
# ********
# torch.Size([1, 94720])
# torch.Size([1, 148, 112, 112])
# ********
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos/AV_MossFormer2_TSE_16K/002/py_faceTracks/00000.avi
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos/AV_MossFormer2_TSE_16K/002/py_faceTracks/00001.avi
# torch.Size([1, 315520])
# torch.Size([1, 493, 112, 112])
# ********
# torch.Size([1, 320993])
# torch.Size([1, 502, 112, 112])
# ********
# torch.Size([1, 320993])
# torch.Size([1, 502, 112, 112])
# ********
# torch.Size([1, 65280])
# torch.Size([1, 102, 112, 112])
# ********
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos/AV_MossFormer2_TSE_16K/003/py_faceTracks/00000.avi
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos/AV_MossFormer2_TSE_16K/003/py_faceTracks/00001.avi
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos/AV_MossFormer2_TSE_16K/003/py_faceTracks/00002.avi
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos/AV_MossFormer2_TSE_16K/003/py_faceTracks/00003.avi
# Running AV_MossFormer2_TSE_16K ...
# torch.Size([1, 128732])
# torch.Size([1, 202, 112, 112])
# ********
# torch.Size([1, 128732])
# torch.Size([1, 202, 112, 112])
# ********
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos_scp/AV_MossFormer2_TSE_16K/001/py_faceTracks/00000.avi
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos_scp/AV_MossFormer2_TSE_16K/001/py_faceTracks/00001.avi
# torch.Size([1, 94720])
# torch.Size([1, 148, 112, 112])
# ********
# torch.Size([1, 94720])
# torch.Size([1, 148, 112, 112])
# ********
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos_scp/AV_MossFormer2_TSE_16K/002/py_faceTracks/00000.avi
# /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/path_to_output_videos_scp/AV_MossFormer2_TSE_16K/002/py_faceTracks/00001.avi
# GPU: NVIDIA L4 (22.03 GiB), PyTorch: 2.15.0+cu132, CUDA: 13.2, 设备: cuda:0
# 任务: target_speaker_extraction, 模型: AV_MossFormer2_TSE_16K, 加载耗时: 3.936 s
# 单视频输入: 001.avi (25.00 fps, 8.040 s)
# 冷启动端到端耗时: 24.888 s, RTF: 3.0955, 实时倍率: 0.32x
# 稳态端到端耗时: 21.805 s, RTF: 2.7121, 实时倍率: 0.37x
# 目录批处理耗时: 92.091 s, SCP 批处理耗时: 38.178 s
# 峰值显存: allocated=661.7 MiB, reserved=754.0 MiB
# 单视频输出目录: /content/LightClear/explore/light_clearvoice/outputs/cuda/AV_MossFormer2_TSE_16K/single_video
