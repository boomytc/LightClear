# AV_MossFormer2_TSE_16K

## 模型简介

ClearVoice 音视频目标说话人提取模型，16 kHz。输入为含人脸的视频。

## 下载来源

https://www.modelscope.cn/models/alibabasglab/AV_MossFormer2_TSE_16K

## 本地路径

`/Users/boom/Model/SE/AV_MossFormer2_TSE_16K`

推理配置：`third_party/clearvoice/config/inference/AV_MossFormer2_TSE_16K.yaml`。

## 运行框架

PyTorch。人脸检测使用 vendored S3FD。

## 音频约束

- 采样率：16000
- 任务：`target_speaker_extraction`
- 输入：视频文件 / 视频目录 / 视频 `.scp`

## 加载方式

```python
from clearvoice import ClearVoice
cv = ClearVoice(task="target_speaker_extraction", model_names=["AV_MossFormer2_TSE_16K"], device="cuda:0")
```

## 硬件实测（`001.avi`，25 fps / 8.040 s）

- 本模型为 700 MB 音视频重链路（含 `ffmpeg` 解帧/重封装 + `scenedetect` + S3FD 人脸检测 + AV-MossFormer2），`demo/demo_tse.py` 默认走 `cuda > cpu`（不启用 MPS，回退 CPU 时打印性能提示；已同步修复上游 `scenedetect >= 0.6.7` 的 `open_video` 兼容以及 `av_mossformer2.py` 中硬编码 `.cuda()`）。
- Colab NVIDIA L4 CUDA（PyTorch 2.15.0+cu132，`cuda:0`）：加载 `3.936 s`，单视频端到端冷启动 `24.888 s` / 稳态 `21.805 s`（RTF `2.7121`，耗时主要在 `ffmpeg`/OpenCV 编解码），峰值显存 `allocated=661.7 MiB, reserved=754.0 MiB`。

## 人脸检测权重

TSE 推理会构造 S3FD，权重文件固定为：

`third_party/clearvoice/models/av_mossformer2_tse/faceDetector/s3fd/sfd_face.pth`

这不是 `AV_MossFormer2_TSE_16K` 的网络 checkpoint，也不进 git。缺文件直接 `FileNotFoundError`，不要用 `gdown` 或家庭缓存补。SE / SS / SR 产品不跑 TSE，不需要这份文件。

## 未验证/待确认

- 官方支持的视频编码、人脸缺失时的行为未在本模块单独核验。
