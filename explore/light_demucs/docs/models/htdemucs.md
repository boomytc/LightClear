# htdemucs

## 模型简介

Hybrid Transformer Demucs v4 默认单模型。把混合音频分成 vocals、drums、bass、other 四轨。人声隔离取 `vocals`，伴奏由其余三轨相加得到。`vocal_isolate_web` 抄的就是这一档。

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

## 未验证/待确认

- 本机 CPU 完整推理时长随曲长变化，未作为门禁。
- MPS / CUDA 速度与数值未在本模块文档中核验。
- 仓库样例 `assets/audio/music/test.mp3` 是上游烟雾片段，人声轨能量很低。
- 默认 demo 使用 `assets/audio/music/next_station_heaven.mp3`（带唱 30 秒，仅供本机演示）。
