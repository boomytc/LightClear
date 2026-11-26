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

产品要单独装到它自己的 `models/<model>`，不要运行时读取本目录。`vocal_isolate_web` 可装 `htdemucs`、`htdemucs_ft`、`htdemucs_6s`，默认仍是 `htdemucs`。

## 安装

在本模块目录内执行：

```bash
uv venv --python 3.12
uv pip install -e .
```

## Demo

4 个能力脚本平铺在 `demo/` 下，均实测支持 CUDA / MPS / CPU（数值与 CPU 对齐，`shifts=0` 时 MPS vs CPU `max_diff <= 1.1e-5`）。顶部通过 `DEVICE = os.environ.get("LIGHTCLEAR_DEVICE", "auto")` 控制设备（默认 `cuda > mps > cpu`，目标加速器不可用时警告并回退 `cpu`），底部并列附三种硬件实测输出。从本模块目录运行：

```bash
# 默认自动选择最优可用设备（cuda > mps > cpu）
.venv/bin/python demo/demo_vocals.py
.venv/bin/python demo/demo_stems.py
.venv/bin/python demo/demo_htdemucs_ft.py
.venv/bin/python demo/demo_htdemucs_6s.py

# 按需通过环境变量或直接修改脚本顶部 DEVICE 指定设备
LIGHTCLEAR_DEVICE=cpu .venv/bin/python demo/demo_vocals.py
LIGHTCLEAR_DEVICE=mps .venv/bin/python demo/demo_vocals.py
LIGHTCLEAR_DEVICE=cuda:0 .venv/bin/python demo/demo_vocals.py
```

### 实测耗时速览（`next_station_heaven.mp3`，30.041 s，`shifts=1, segment=7`）

| 模型 | Apple M4 Pro CPU | Apple M4 Pro MPS（稳态） | Colab NVIDIA L4 CUDA（稳态） | L4 峰值显存（alloc / reserved） |
| --- | --- | --- | --- | --- |
| `htdemucs`（人声隔离） | 7.643 s（RTF 0.2544 / 3.93x） | 2.427 s（RTF 0.0808 / 12.38x） | 0.890 s（RTF 0.0296 / 33.74x） | 546.8 / 770.0 MiB |
| `htdemucs`（四轨） | 8.141 s（RTF 0.2710 / 3.69x） | 2.210 s（RTF 0.0736 / 13.59x） | 0.869 s（RTF 0.0289 / 34.59x） | 546.8 / 770.0 MiB |
| `htdemucs_ft`（四轨质量档） | 31.885 s（RTF 1.0614 / 0.94x） | 9.470 s（RTF 0.3152 / 3.17x） | 3.221 s（RTF 0.1072 / 9.33x） | 546.8 / 770.0 MiB |
| `htdemucs_6s`（六轨） | 7.338 s（RTF 0.2443 / 4.09x） | 2.670 s（RTF 0.0889 / 11.25x） | 0.834 s（RTF 0.0278 / 36.01x） | 583.3 / 752.0 MiB |

样例来自仓库根 `assets/audio/music/`（通过相对 `MODULE_ROOT` 跨平台解析）。默认 demo 用 `next_station_heaven.mp3`（带唱短片段，仅供本机演示）。`test.mp3` 是上游烟雾片，人声能量很低。输出按硬件写入本模块 `outputs/{cpu,mps,cuda}/`。

## 模型

见 `docs/模型列表.md`、`docs/models/` 与 `docs/参数调优与音频导出.md`。
