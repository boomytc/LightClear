# MossFormer2_SR_48K

## 模型简介

ClearVoice 语音超分辨率模型，目标 48 kHz。

## 下载来源

https://www.modelscope.cn/models/alibabasglab/MossFormer2_SR_48K

## 本地路径

`/Users/boom/Model/SR/MossFormer2_SR_48K`

推理配置：`third_party/clearvoice/config/inference/MossFormer2_SR_48K.yaml`。

## 运行框架

PyTorch。

## 音频约束

- 输出采样率：48000
- 任务：`speech_super_resolution`

## 加载方式

```python
from clearvoice import ClearVoice
cv = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
```

## 未验证/待确认

- 输入带宽下限、参数规模未在本模块源码中核验。
