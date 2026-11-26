# htdemucs

## 模型简介

Hybrid Transformer Demucs v4 默认单模型。把混合音频分成 vocals、drums、bass、other 四轨。人声隔离取 `vocals`，伴奏由其余三轨相加得到。`vocal_isolate_web` 的默认档就是这一份。

同任务质量档见 `htdemucs_ft`。吉他 / 钢琴单独一轨见 `htdemucs_6s`。

## 下载来源

https://huggingface.co/adefossez/HTDemucs

上游：https://github.com/adefossez/demucs（v4.1.0）

## 本地路径

`explore/light_demucs/models/htdemucs`

内含 `htdemucs.yaml` 与签名 `.th`。安装：

```bash
cd explore/light_demucs
.venv/bin/python scripts/install_model.py htdemucs
```

## 运行框架

PyTorch。包源码在 `third_party/demucs`。

## 音频约束

- 模型采样率：44100
- 输出四轨：vocals / drums / bass / other
- CPU 推理建议减小 `segment`（demo 用 7）

## 加载方式

```python
from pathlib import Path
from demucs.api import Separator
separator = Separator(model="htdemucs", repo=Path("models/htdemucs"), device="cpu", segment=7)
origin, stems = separator.separate_audio_file("track.mp3")
```

## 硬件实测（`next_station_heaven.mp3`，30.041 s，`shifts=1, overlap=0.25, segment=7`）

- Apple M4 Pro CPU（PyTorch 2.13.0）：人声隔离 `7.643 s`（RTF `0.2544`，`3.93x`），四轨 `8.141 s`（RTF `0.2710`，`3.69x`）。
- Apple M4 Pro MPS（PyTorch 2.13.0）：人声隔离冷启动 `3.256 s` / 稳态 `2.427 s`（RTF `0.0808`，`12.38x`）；四轨冷启动 `2.747 s` / 稳态 `2.210 s`（RTF `0.0736`，`13.59x`）。确定性模式（`shifts=0`）与 CPU 最大误差 `7.39e-06`（RMS `3.76e-07`），无 NaN。
- Colab NVIDIA L4 CUDA（PyTorch 2.15.0+cu132，CUDA 13.2，`cuda:0`）：人声隔离冷启动 `1.699 s` / 稳态 `0.890 s`（RTF `0.0296`，`33.74x`）；四轨稳态 `0.869 s`（RTF `0.0289`，`34.59x`）；峰值显存 `allocated=546.8 MiB, reserved=770.0 MiB`。

## 未验证/待确认

- 本机 CPU 完整推理时长随曲长变化，未作为门禁。
- 仓库样例 `assets/audio/music/test.mp3` 是上游烟雾片段，人声轨能量很低。
- 默认 demo 使用 `assets/audio/music/next_station_heaven.mp3`（带唱 30 秒，仅供本机演示）。
