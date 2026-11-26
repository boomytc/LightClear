# FRCRN_SE_16K

## 模型简介

ClearVoice 语音增强模型，16 kHz。

## 下载来源

https://www.modelscope.cn/models/alibabasglab/FRCRN_SE_16K

## 本地路径

`/Users/boom/Model/SE/FRCRN_SE_16K`

推理配置：`third_party/clearvoice/config/inference/FRCRN_SE_16K.yaml`。

## 运行框架

PyTorch。

## 音频约束

- 采样率：16000
- 任务：`speech_enhancement`

## 加载方式

```python
from clearvoice import ClearVoice
cv = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"], device="cuda:0")  # 或 "cpu"
```

## 硬件实测（`input.wav`，16 kHz / 2.398 s）

- Colab CPU（AMD EPYC 7B12 2 vCPU，PyTorch 2.15.0+cpu）：单文件 `7.981 s`（RTF `3.3285`，`0.30x`）；Apple M4 Pro CPU 单文件约 `2.66 s`。
- Apple M4 Pro MPS（PyTorch 2.13.0）：实测单文件稳态 `4.20 s`（复数循环算子同步开销大，**反慢于本机 CPU**，批处理超 8 分钟），因此不保留 `demo/mps/` 示例。
- Colab NVIDIA L4 CUDA（PyTorch 2.15.0+cu132，`cuda:0`）：冷启动 `0.429 s` / 稳态 `0.103 s`（RTF `0.0428`，`23.35x`）；峰值显存 `allocated=1341.7 MiB, reserved=3450.0 MiB`。

## 推理 API

文件 / 目录 / `.scp` 或 numpy / tensor，同 `ClearVoice.__call__`。

## 未验证/待确认

- 参数规模、license 未在本模块源码中核验。
