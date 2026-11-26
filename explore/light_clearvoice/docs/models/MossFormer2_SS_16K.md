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
cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"], device="cuda:0")  # 或 "mps"
```

## 硬件实测（`input_ss.wav`，16 kHz / 4.409 s）

- CPU：Apple M4 Pro CPU 单文件实测 `39.12 s`（RTF `8.9`），批处理耗时过长，故不单设 `demo/cpu/`。
- Apple M4 Pro MPS（PyTorch 2.13.0）：冷启动 `2.454 s` / 稳态 `1.806 s`（RTF `0.4096`，`2.44x`），与 CPU 最大误差 `2.26e-07`，无 NaN。
- Colab NVIDIA L4 CUDA（PyTorch 2.15.0+cu132，`cuda:0`）：冷启动 `1.283 s` / 稳态 `0.897 s`（RTF `0.2034`，`4.92x`）；峰值显存 `allocated=461.4 MiB, reserved=488.0 MiB`。

## 未验证/待确认

- 说话人数量是否可配置未在本模块文档中核验；当前产品与 demo 按两路处理。
