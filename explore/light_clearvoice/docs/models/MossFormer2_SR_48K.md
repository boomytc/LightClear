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
cv = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"], device="cuda:0")  # 或 "mps" / "cpu"
```

## 硬件实测（`input_sr.wav`，16 kHz / 2.398 s → 48 kHz）

- Colab CPU（AMD EPYC 7B12 2 vCPU，PyTorch 2.15.0+cpu）：单文件 `21.008 s`（RTF `8.7617`，`0.11x`）；Apple M4 Pro CPU 单文件约 `6.15 s`。
- Apple M4 Pro MPS（PyTorch 2.13.0）：冷启动 `1.304 s` / 稳态 `0.525 s`（RTF `0.2190`，`4.57x`），与 CPU 最大误差 `2.24e-06`，无 NaN。
- Colab NVIDIA L4 CUDA（PyTorch 2.15.0+cu132，`cuda:0`）：冷启动 `1.181 s` / 稳态 `0.224 s`（RTF `0.0935`，`10.70x`）；峰值显存 `allocated=1111.2 MiB, reserved=1772.0 MiB`。

## 未验证/待确认

- 输入带宽下限、参数规模未在本模块源码中核验。
