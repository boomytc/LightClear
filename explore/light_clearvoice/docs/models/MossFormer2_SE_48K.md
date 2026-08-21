# MossFormer2_SE_48K

## 模型简介

ClearVoice 语音增强模型，48 kHz。

## 下载来源

https://www.modelscope.cn/models/alibabasglab/MossFormer2_SE_48K

## 本地路径

`/Users/boom/Model/SE/MossFormer2_SE_48K`

推理配置：`third_party/clearvoice/config/inference/MossFormer2_SE_48K.yaml`（`checkpoint_dir` 与上路径一致）。

## 运行框架

PyTorch。网络类由 `clearvoice.network_wrapper` 按 task/model 加载。

## 音频约束

- 采样率：48000（配置 `sampling_rate`）
- 任务：`speech_enhancement`

## 加载方式

```python
from clearvoice import ClearVoice
cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
```

## 推理 API

- 文件 / 目录 / `.scp`：`cv(input_path=..., online_write=False|True, output_path=...)`
- numpy / tensor：`cv(array_or_tensor, False)`，输入形状 `[batch, length]`

## 未验证/待确认

- 参数规模、license、官方支持语言列表未在本模块源码中核验。
