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
cv = ClearVoice(task="target_speaker_extraction", model_names=["AV_MossFormer2_TSE_16K"])
```

## 人脸检测权重

TSE 推理会构造 S3FD，权重文件固定为：

`third_party/clearvoice/models/av_mossformer2_tse/faceDetector/s3fd/sfd_face.pth`

这不是 `AV_MossFormer2_TSE_16K` 的网络 checkpoint，也不进 git。缺文件直接 `FileNotFoundError`，不要用 `gdown` 或家庭缓存补。SE / SS / SR 产品不跑 TSE，不需要这份文件。

## 未验证/待确认

- 本机中心目录若尚未安装 `AV_MossFormer2_TSE_16K`，`demo/demo_tse.py` 完整推理仍缺网络权重。
- 官方支持的视频编码、人脸缺失时的行为未在本模块单独核验。
