# light_demucs

Demucs 探索模块。把混合音频拆成人声或分轨，让目标层听得更清。公共入口：

```python
from demucs.api import Separator
```

默认模型 `htdemucs`（四轨）。另外两档：`htdemucs_ft`（同四轨质量袋）、`htdemucs_6s`（六轨，含 guitar / piano）。权重放在本模块 `models/<model>/`。首次安装：

```bash
.venv/bin/python scripts/install_model.py htdemucs
.venv/bin/python scripts/install_model.py htdemucs_ft
.venv/bin/python scripts/install_model.py htdemucs_6s
```

产品要单独装到它自己的 `models/<model>`，不要运行时读取本目录。`vocal_isolate_web` 目前只装 `htdemucs`。

## 安装

在本模块目录内执行：

```bash
uv venv --python 3.12
uv pip install -e .
```

## Demo

从本模块目录运行：

```bash
.venv/bin/python demo/demo_vocals.py
.venv/bin/python demo/demo_stems.py
.venv/bin/python demo/demo_htdemucs_ft.py
.venv/bin/python demo/demo_htdemucs_6s.py
```

样例来自仓库根 `assets/audio/music/`。默认 demo 用 `next_station_heaven.mp3`（带唱短片段，仅供本机演示）。`test.mp3` 是上游烟雾片，人声能量很低。输出写入本模块 `outputs/`。

## 模型

见 `docs/模型列表.md` 与 `docs/models/`。
