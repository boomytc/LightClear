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

## Demo（按能力组织，顶部 `DEVICE` 控制设备并自动回退 CPU）

每个脚本对应一项独立能力，顶部通过全局 `DEVICE = os.environ.get("LIGHTCLEAR_DEVICE", "auto")` 指定目标设备（支持 `auto` / `cuda` / `cuda:0` / `mps` / `cpu`），目标加速器不可用时打印警告并自动回退到 `cpu`；同时支持通过 `LIGHTCLEAR_MODEL_ROOT` 指定中心权重根目录（默认 `~/Model`，本机即 `/Users/boom/Model`）。脚本底部均并列附真实硬件实测输出注释。

| 脚本 | 对应能力 / 模型 | `auto` 优先级与设备说明 |
| --- | --- | --- |
| `demo/demo_se.py` | 48 kHz 语音增强（`MossFormer2_SE_48K`） | `cuda > mps > cpu` |
| `demo/demo_Numpy2Numpy.py` | 内存 NumPy 音频输入输出（`MossFormer2_SE_48K`） | `cuda > mps > cpu` |
| `demo/demo_ss.py` | 16 kHz 双人说话人分离（`MossFormer2_SS_16K`） | `cuda > mps > cpu`（CPU 较慢，落到 CPU 时打印性能提示） |
| `demo/demo_sr.py` | 48 kHz 语音超分辨率（`MossFormer2_SR_48K`） | `cuda > mps > cpu` |
| `demo/demo_frcrn_se.py` | 16 kHz 语音增强（`FRCRN_SE_16K`） | `cuda > cpu`（MPS 实测性能退化，指定 `mps` 自动回退 `cpu`） |
| `demo/demo_tensor2tensor.py` | NumPy / Torch Tensor 输入对比（`FRCRN_SE_16K`） | `cuda > cpu`（MPS 实测性能退化，指定 `mps` 自动回退 `cpu`） |
| `demo/demo_mossformergan_se.py` | 16 kHz GAN 语音增强（`MossFormerGAN_SE_16K`） | `cuda > cpu`（MPS/CPU 过重，落到 CPU 时打印性能提示） |
| `demo/demo_tse.py` | 16 kHz 音视频目标说话人提取（`AV_MossFormer2_TSE_16K`） | `cuda > cpu`（重视频链路，落到 CPU 时打印性能提示） |

从本模块目录运行示例：

```bash
# 默认按能力自动选择最优可用设备（支持 MPS 的脚本走 cuda > mps > cpu，其余走 cuda > cpu）
.venv/bin/python demo/demo_se.py
.venv/bin/python demo/demo_frcrn_se.py
.venv/bin/python demo/demo_mossformergan_se.py
.venv/bin/python demo/demo_ss.py
.venv/bin/python demo/demo_sr.py
.venv/bin/python demo/demo_tse.py
.venv/bin/python demo/demo_Numpy2Numpy.py
.venv/bin/python demo/demo_tensor2tensor.py

# 按需通过环境变量或直接修改脚本顶部 DEVICE 切换设备（目标加速器不可用时自动警告回退 cpu）
LIGHTCLEAR_DEVICE=cpu .venv/bin/python demo/demo_se.py
LIGHTCLEAR_DEVICE=mps .venv/bin/python demo/demo_se.py
LIGHTCLEAR_DEVICE=cuda:0 .venv/bin/python demo/demo_se.py
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
