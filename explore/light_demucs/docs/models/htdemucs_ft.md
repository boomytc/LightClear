# htdemucs_ft

## 模型简介

`htdemucs` 的按声部微调袋。输出仍是 vocals / drums / bass / other，不是新任务。四个专家模型各负责一轨，推理跑四次再按权重拼回，官方大约 4 倍耗时、MUSDB HQ SDR 约 9.20 dB（单模型 `htdemucs` 约 9.00 dB）。

`vocal_isolate_web` 已把本档作为可选质量档接入，任务仍是 `vocal_isolation`。

## 下载来源

https://huggingface.co/adefossez/HTDemucs-ft

上游：https://github.com/adefossez/demucs（v4.1.0）

## 本地路径

`explore/light_demucs/models/htdemucs_ft`

内含 `htdemucs_ft.yaml` 与四个签名 `.th`。安装：

```bash
cd explore/light_demucs
.venv/bin/python scripts/install_model.py htdemucs_ft
```

## 运行框架

PyTorch。包源码在 `third_party/demucs`。`Separator` 通过 `BagOfModels` 加载。

## 音频约束

- 模型采样率：44100
- 输出四轨：vocals / drums / bass / other
- 人声隔离取 `vocals`，伴奏为 drums + bass + other
- CPU 推理建议减小 `segment`（demo 用 7）；耗时大约是 `htdemucs` 的四倍

## 加载方式

```python
from pathlib import Path
from demucs.api import Separator
separator = Separator(model="htdemucs_ft", repo=Path("models/htdemucs_ft"), device="cpu", segment=7)
origin, stems = separator.separate_audio_file("track.mp3")
```

## 硬件实测（`next_station_heaven.mp3`，30.041 s，`shifts=1, overlap=0.25, segment=7`）

- Apple M4 Pro CPU（PyTorch 2.13.0）：`31.885 s`（RTF `1.0614`，`0.94x`）。
- Apple M4 Pro MPS（PyTorch 2.13.0）：冷启动 `9.150 s` / 稳态 `9.470 s`（RTF `0.3152`，`3.17x`）。确定性模式（`shifts=0`）与 CPU 最大误差 `1.08e-05`（RMS `3.32e-07`），无 NaN。
- Colab NVIDIA L4 CUDA（PyTorch 2.15.0+cu132，CUDA 13.2，`cuda:0`）：冷启动 `3.883 s` / 稳态 `3.221 s`（RTF `0.1072`，`9.33x`）；峰值显存 `allocated=546.8 MiB, reserved=770.0 MiB`。

## 未验证/待确认

- 本机未把 ft 与默认模型做旁路听感对比，不作为门禁。
- 默认 demo 使用 `assets/audio/music/next_station_heaven.mp3`（带唱 30 秒，仅供本机演示）。
