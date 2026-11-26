# light_clearvoice

ClearVoice 探索模块。公共入口：

```python
from clearvoice import ClearVoice
```

覆盖语音增强、说话人分离、语音超分辨率和音视频目标说话人提取。模型权重使用中心目录 `/Users/boom/Model/{SE,SS,SR}/`。TSE 人脸检测权重 `sfd_face.pth` 放在 vendored S3FD 目录，不走中心 checkpoint。Demucs 分轨在兄弟模块 `light_demucs`。

## 安装

在本模块目录内执行：

```bash
uv venv --python 3.12
uv pip install -e .
```

## Demo（按实际可用硬件拆分为 `cpu` / `mps` / `cuda`）

为了给产品落地提供真实参考，仅在对应硬件上耗时和内存具备实用价值的模型才建立对应脚本（因此各硬件目录下脚本数不对等）；所有脚本底部均附真实硬件实测输出注释，支持通过 `LIGHTCLEAR_MODEL_ROOT` 指定中心权重根目录（默认 `/Users/boom/Model`）。

| 硬件目录 | 包含脚本 | 裁剪说明 |
| --- | --- | --- |
| `demo/cpu/`（5 个） | `demo_se.py`、`demo_Numpy2Numpy.py`、`demo_frcrn_se.py`、`demo_tensor2tensor.py`、`demo_sr.py` | 剔除 CPU 过慢的重模型 `MossFormer2_SS_16K`（RTF ~8.9）、`MossFormerGAN_SE_16K`（RTF ~6.6）与 `AV_MossFormer2_TSE_16K` |
| `demo/mps/`（4 个） | `demo_se.py`、`demo_Numpy2Numpy.py`、`demo_sr.py`、`demo_ss.py` | 剔除 MPS 反比 CPU 慢的 `FRCRN_SE_16K`、显存/耗时过重的 `MossFormerGAN_SE_16K`（L4 峰值 7.8 GiB）与重视频链路 `AV_MossFormer2_TSE_16K` |
| `demo/cuda/`（8 个） | 全部 8 个脚本（含 `demo_se` / `demo_frcrn_se` / `demo_mossformergan_se` / `demo_ss` / `demo_sr` / `demo_tse` / `demo_Numpy2Numpy` / `demo_tensor2tensor`） | 显式使用 `cuda:0`，含冷启动/稳态计时与峰值显存统计 |

从本模块目录运行示例：

```bash
# CPU（通用基线）
.venv/bin/python demo/cpu/demo_se.py
.venv/bin/python demo/cpu/demo_frcrn_se.py
.venv/bin/python demo/cpu/demo_sr.py
.venv/bin/python demo/cpu/demo_Numpy2Numpy.py
.venv/bin/python demo/cpu/demo_tensor2tensor.py

# Apple Silicon MPS（需 macOS arm64）
.venv/bin/python demo/mps/demo_se.py
.venv/bin/python demo/mps/demo_ss.py
.venv/bin/python demo/mps/demo_sr.py
.venv/bin/python demo/mps/demo_Numpy2Numpy.py

# NVIDIA CUDA（需 NVIDIA GPU）
.venv/bin/python demo/cuda/demo_se.py
.venv/bin/python demo/cuda/demo_frcrn_se.py
.venv/bin/python demo/cuda/demo_mossformergan_se.py
.venv/bin/python demo/cuda/demo_ss.py
.venv/bin/python demo/cuda/demo_sr.py
.venv/bin/python demo/cuda/demo_tse.py
.venv/bin/python demo/cuda/demo_Numpy2Numpy.py
.venv/bin/python demo/cuda/demo_tensor2tensor.py
```

### 实测单文件稳态耗时速览

| 模型（单文件样例 / 时长） | Colab EPYC 2 vCPU | Apple M4 Pro MPS | Colab NVIDIA L4 CUDA | L4 峰值显存（alloc / reserved） |
| --- | --- | --- | --- | --- |
| `MossFormer2_SE_48K`（`input.wav` / 2.40 s） | 1.509 s（RTF 0.63 / 1.59x） | **0.103 s**（RTF 0.043 / 23.34x） | 0.148 s（RTF 0.062 / 16.17x） | 350.5 / 398.0 MiB |
| `FRCRN_SE_16K`（`input.wav` / 2.40 s） | 7.981 s（RTF 3.33 / 0.30x） | 不推荐（复数循环反慢于 CPU） | **0.103 s**（RTF 0.043 / 23.35x） | 1341.7 / 3450.0 MiB |
| `MossFormerGAN_SE_16K`（`test.wav` / 4.00 s） | 不推荐（过慢） | 不推荐（批处理过重） | **0.604 s**（RTF 0.151 / 6.63x） | 4461.1 / 7830.0 MiB |
| `MossFormer2_SS_16K`（`input_ss.wav` / 4.41 s） | 不推荐（过慢） | 1.806 s（RTF 0.410 / 2.44x） | **0.897 s**（RTF 0.203 / 4.92x） | 461.4 / 488.0 MiB |
| `MossFormer2_SR_48K`（`input_sr.wav` / 2.40 s） | 21.008 s（RTF 8.76 / 0.11x） | 0.525 s（RTF 0.219 / 4.57x） | **0.224 s**（RTF 0.094 / 10.70x） | 1111.2 / 1772.0 MiB |
| `AV_MossFormer2_TSE_16K`（`001.avi` / 8.04 s） | 不推荐（重视频链路） | 不推荐（重视频链路） | **21.805 s**（端到端含解帧/封装） | 661.7 / 754.0 MiB |

样例音频/视频来自仓库根 `assets/`（通过相对 `MODULE_ROOT` 跨平台解析）。输出按硬件写入本模块 `outputs/{cpu,mps,cuda}/`。

## 模型

见 `docs/模型列表.md` 与 `docs/models/`。
