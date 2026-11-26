# htdemucs_6s

## 模型简介

实验六轨模型。在四轨之外再拆 `guitar` 与 `piano`。权重里的声部顺序是 drums / bass / other / vocals / guitar / piano。官方说明吉他尚可，钢琴串音和伪影多。

这是乐器分轨能力，不是更好的人声隔离。人声仍取 `vocals`；伴奏是其余五轨相加。`vocal_isolate_web` 把它作为可选六轨，默认仍是四轨 `htdemucs`。

## 下载来源

https://huggingface.co/adefossez/HTDemucs-6s

上游：https://github.com/adefossez/demucs（v4.1.0）

## 本地路径

`explore/light_demucs/models/htdemucs_6s`

内含 `htdemucs_6s.yaml` 与签名 `.th`。安装：

```bash
cd explore/light_demucs
.venv/bin/python scripts/install_model.py htdemucs_6s
```

## 运行框架

PyTorch。包源码在 `third_party/demucs`。

## 音频约束

- 模型采样率：44100
- 输出六轨：drums / bass / other / vocals / guitar / piano
- CPU 推理建议减小 `segment`（demo 用 7）

## 加载方式

```python
from pathlib import Path
from demucs.api import Separator
separator = Separator(model="htdemucs_6s", repo=Path("models/htdemucs_6s"), device="cpu", segment=7)
origin, stems = separator.separate_audio_file("track.mp3")
```

## 硬件实测（`next_station_heaven.mp3`，30.041 s，`shifts=1, overlap=0.25, segment=7`）

- Apple M4 Pro CPU（PyTorch 2.13.0）：`7.338 s`（RTF `0.2443`，`4.09x`）。
- Apple M4 Pro MPS（PyTorch 2.13.0）：冷启动 `2.894 s` / 稳态 `2.670 s`（RTF `0.0889`，`11.25x`）。确定性模式（`shifts=0`）与 CPU 最大误差 `3.23e-06`（RMS `2.09e-07`），无 NaN。
- Colab NVIDIA L4 CUDA（PyTorch 2.15.0+cu132，CUDA 13.2，`cuda:0`）：冷启动 `1.383 s` / 稳态 `0.834 s`（RTF `0.0278`，`36.01x`）；峰值显存 `allocated=583.3 MiB, reserved=752.0 MiB`。

## 未验证/待确认

- 钢琴轨质量按上游说明视为实验性，本模块不把它当可交付指标。
- 共享样例 `next_station_heaven.mp3` 上 guitar / piano 能量极低，只证明六轨合同能跑，不证明这两轨在该曲上可听。
- 默认 demo 使用 `assets/audio/music/next_station_heaven.mp3`（带唱 30 秒，仅供本机演示）。
