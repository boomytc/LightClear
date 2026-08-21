# htdemucs_6s

## 模型简介

实验六轨模型。在四轨之外再拆 `guitar` 与 `piano`。权重里的声部顺序是 drums / bass / other / vocals / guitar / piano。官方说明吉他尚可，钢琴串音和伪影多。

这是乐器分轨能力，不是更好的人声隔离。人声仍取 `vocals`；伴奏是其余五轨相加。不要接到 `vocal_isolate_web` 的默认四轨合同。

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

## 未验证/待确认

- 钢琴轨质量按上游说明视为实验性，本模块不把它当可交付指标。
- 共享样例 `next_station_heaven.mp3` 上 guitar / piano 能量极低，只证明六轨合同能跑，不证明这两轨在该曲上可听。
- MPS / CUDA 速度与数值未在本模块文档中核验。
- 默认 demo 使用 `assets/audio/music/next_station_heaven.mp3`（带唱 30 秒，仅供本机演示）。
