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

`third_party/clearvoice/models/av_mossformer2_tse/faceDetector/s3fd/sfd_face.pth` 不进 git。TSE 推理前该文件必须出现在上述路径。

## 未验证/待确认

- 官方支持的视频编码、人脸缺失时的行为未在本模块单独核验。
