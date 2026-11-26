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
cv = ClearVoice(task="speech_enhancement", model_names=["MossFormerGAN_SE_16K"], device="cuda:0")
```

## 硬件实测（`test.wav`，48 kHz / 4.003 s）

- CPU / Apple Silicon MPS：实测单文件 CPU 约 `26.5 s`（RTF `6.6`），MPS 虽单文件 `1.18 s` 但全量目录批处理超 9 分钟且显存需求高，均不具备产品部署性价比，故 `demo/demo_mossformergan_se.py` 默认走 `cuda > cpu`（不启用 MPS，落到 CPU 时打印性能提示）。
- Colab NVIDIA L4 CUDA（PyTorch 2.15.0+cu132，`cuda:0`）：冷启动 `1.493 s` / 稳态 `0.604 s`（RTF `0.1509`，`6.63x`）；目录批处理 `69.600 s`；峰值显存 `allocated=4461.1 MiB, reserved=7830.0 MiB`。

## 未验证/待确认

- 参数规模、license 未在本模块源码中核验。
