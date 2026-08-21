# MossFormerGAN_SE_16K

## 模型简介

ClearVoice GAN 语音增强模型，16 kHz。

## 下载来源

https://www.modelscope.cn/models/alibabasglab/MossFormerGAN_SE_16K

## 本地路径

`/Users/boom/Model/SE/MossFormerGAN_SE_16K`

推理配置：`third_party/clearvoice/config/inference/MossFormerGAN_SE_16K.yaml`。

## 运行框架

PyTorch。

## 音频约束

- 采样率：16000
- 任务：`speech_enhancement`

## 加载方式

```python
from clearvoice import ClearVoice
cv = ClearVoice(task="speech_enhancement", model_names=["MossFormerGAN_SE_16K"])
```

## 未验证/待确认

- 参数规模、license 未在本模块源码中核验。
