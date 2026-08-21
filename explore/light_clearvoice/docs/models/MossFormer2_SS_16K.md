# MossFormer2_SS_16K

## 模型简介

ClearVoice 两路说话人分离模型，16 kHz。

## 下载来源

https://www.modelscope.cn/models/alibabasglab/MossFormer2_SS_16K

## 本地路径

`/Users/boom/Model/SS/MossFormer2_SS_16K`

推理配置：`third_party/clearvoice/config/inference/MossFormer2_SS_16K.yaml`。

## 运行框架

PyTorch。

## 音频约束

- 采样率：16000
- 任务：`speech_separation`
- 输出两路说话人

## 加载方式

```python
from clearvoice import ClearVoice
cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
```

## 未验证/待确认

- 说话人数量是否可配置未在本模块文档中核验；当前产品与 demo 按两路处理。
