# light_voice_isolator

上游 **Inworld Voice Isolator**（v1.0.0，Apache-2.0）独立探索模块。覆盖 48 kHz 全频带人声隔离（Denoising，严守不捏造带宽契约），同时包含离线高质量模型 `voice-isolator-v1`（64.05M）与因果低延迟流式模型 `voice-isolator-v1-realtime`（58.18M）。

公共入口：

```python
from inworld import voice_isolation
```

## 安装与环境

在本模块目录内执行（Apple Silicon 上依赖 `torch >= 2.6` 自动启用 MPS + Metal Shader 加速）：

```bash
cd explore/light_voice_isolator
uv venv --python 3.12
uv pip install -e .
```

## 权重准备（中心目录）

模型权重从 ModelScope [`inworld/voice-isolator`](https://www.modelscope.cn/models/inworld/voice-isolator) 下载到中心目录 `/Users/boom/Model/SE/inworld-voice-isolator/`（禁止提交进仓库，demo 不静默联网下载）：

```bash
mkdir -p /Users/boom/Model/SE/inworld-voice-isolator
curl -L --fail -o /Users/boom/Model/SE/inworld-voice-isolator/denoiser.safetensors \
  "https://www.modelscope.cn/models/inworld/voice-isolator/resolve/master/denoiser.safetensors"
curl -L --fail -o /Users/boom/Model/SE/inworld-voice-isolator/denoiser_realtime.safetensors \
  "https://www.modelscope.cn/models/inworld/voice-isolator/resolve/master/denoiser_realtime.safetensors"
shasum -a 256 /Users/boom/Model/SE/inworld-voice-isolator/*.safetensors
```

官方 SHA-256（对照 `third_party/inworld/voice_isolation/registry.py`）：

| 文件 | 模型名 | 大小 | SHA-256 |
| --- | --- | --- | --- |
| `denoiser.safetensors` | `voice-isolator-v1` | 244 MiB (256 MB) | `647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53` |
| `denoiser_realtime.safetensors` | `voice-isolator-v1-realtime` | 222 MiB (233 MB) | `76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f` |

## 6 组聚焦 Demo（仅保留 `mps` / `cuda` 两套实用硬件目录）

所有 demo 均为单文件顶层顺序脚本，样例路径由 `MODULE_ROOT.parent.parent / "assets"` 跨平台推导，中心模型根目录支持 `LIGHTCLEAR_MODEL_ROOT` 环境变量（默认 `/Users/boom/Model`），输出按硬件写入本模块 `outputs/{mps,cuda}/`。

> 说明：本模块不单设 `demo/cpu/` 目录——实测即使在 Apple M4 Pro 上按上游 `cli.py` 限 2 线程，离线 RTF 仍达 `4.31`、20 ms 流式 RTF 达 `11.18`（完全不可实时，无产品部署价值）；与 CPU 的单条速度及数值一致性对比已直接内嵌在 `demo/{mps,cuda}/demo_offline_options.py` 第 5 节与 [`docs/models/inworld-voice-isolator.md`](./docs/models/inworld-voice-isolator.md) 中。

| 脚本 | 覆盖能力与调用路径 |
| --- | --- |
| `demo/{mps,cuda}/demo_offline_single.py` | 离线单文件 `VoiceIsolator(mode="offline").isolate()`、float32 人声 + `aligned_mono - voice` 残差保存、原电平保持与重构误差 `max(abs(input - (voice + background)))` 校验、冷启动/稳态 RTF 与 CUDA 峰值显存输出 |
| `demo/{mps,cuda}/demo_offline_options.py` | 离线全参数覆盖：`output_sample_rate`（48 kHz vs 原率）、`chunk_seconds`/`overlap_seconds`（8s/2s vs 4s/1s）、`offline_passes=1 vs 2`、`precision`（`fp32/auto/bf16/fp16`）、与 CPU 速度/数值一致性、立体声降混、NumPy/Tensor 类型保持与 PCM16 自动防削波 |
| `demo/{mps,cuda}/demo_batch_many.py` | `isolate_many()` 离线同长合批、异长自动分组按原序返回、与单条 `isolate()` 误差校验，以及 `mode="streaming"` 多路 lockstep 并发批流式隔离 |
| `demo/{mps,cuda}/demo_streaming_realtime.py` | 实时流式 `live.stream()` → 20 ms 分片逐块 `push()` → `flush()`、与非整帧片长（333 samples）输出一致性、算法延迟 `latency_samples`/`latency_seconds`，以及带 `lengths` 的多路并发流式 `push([clips, samples])` |
| `demo/{mps,cuda}/demo_streaming_knobs.py` | 流式高阶旋钮：`lookahead_frames`（`0 / 2 / 4`）、`block_frames`（`1 / 4 / 16` 吞吐不增延迟）、`cache_frames`（`100 / 300 / 800`）与双轨滑窗 `window_frames`/`crossfade_frames`（`600/200` vs `0`）在长音频上的漂移与 RTF 对比 |
| `demo/{mps,cuda}/demo_lowlevel_network.py` | 白盒底层调用：`load_model()` 获取 `VoiceIsolatorNetwork`，显式串联 `stft → encode → features → denoise_decoder → correction → istft`，打印设备、`scan.metal_kernel()`/`scan.kernel()` 命中状态与双模型参数量 |

从本模块目录运行（按目标硬件选择子目录）：

```bash
# Apple Silicon MPS（推荐本机运行，Metal Shader 加速，默认 fp32）
.venv/bin/python demo/mps/demo_offline_single.py
.venv/bin/python demo/mps/demo_offline_options.py
.venv/bin/python demo/mps/demo_batch_many.py
.venv/bin/python demo/mps/demo_streaming_realtime.py
.venv/bin/python demo/mps/demo_streaming_knobs.py
.venv/bin/python demo/mps/demo_lowlevel_network.py

# NVIDIA CUDA（需 NVIDIA GPU，显式 cuda:0，默认 bf16；可通过 LIGHTCLEAR_MODEL_ROOT 指定权重根）
.venv/bin/python demo/cuda/demo_offline_single.py
.venv/bin/python demo/cuda/demo_offline_options.py
.venv/bin/python demo/cuda/demo_batch_many.py
.venv/bin/python demo/cuda/demo_streaming_realtime.py
.venv/bin/python demo/cuda/demo_streaming_knobs.py
.venv/bin/python demo/cuda/demo_lowlevel_network.py
```

## CLI 与环境自检

安装可编辑包后，`.venv/bin/` 提供 `voice-isolate-doctor` 与 `voice-isolate` 两个入口：

### 1. 环境与模型自检（`voice-isolate-doctor`）

```bash
# 仅检查本机 Python / PyTorch / 设备（mps/cuda/cpu）与 HF 缓存状态
.venv/bin/voice-isolate-doctor

# 使用中心目录本地权重执行完整冒烟（离线 + 流式双轨各跑 0.5 s 静音，不联网下载）
.venv/bin/voice-isolate-doctor --full \
  --offline-model /Users/boom/Model/SE/inworld-voice-isolator/denoiser.safetensors \
  --online-model /Users/boom/Model/SE/inworld-voice-isolator/denoiser_realtime.safetensors
```

> 注：未传 `--offline-model`/`--online-model` 时，`--full` 会尝试从 Hugging Face 下载并校验缓存。在 LightClear 本地环境中建议显式传入本地权重路径。

### 2. 单文件离线 / 流式 CLI（`voice-isolate`）

```bash
# 离线默认：输出 48 kHz float32 人声 + 同目录 <stem>_background.wav（可用 --background 指定残差路径）
.venv/bin/voice-isolate \
  /Users/boom/workspace/LightClear/assets/audio/noisy/input.wav \
  outputs/cli/input_voice.wav \
  --model /Users/boom/Model/SE/inworld-voice-isolator/denoiser.safetensors \
  --background outputs/cli/input_bg.wav

# 仅输出人声（默认 PCM_16 防削波）；加 --float32 保留 float32；--output-sr 0 保持原采样率（如 16 kHz）
.venv/bin/voice-isolate \
  /Users/boom/workspace/LightClear/assets/audio/noisy/input.wav \
  outputs/cli/input_16k_float.wav \
  --model /Users/boom/Model/SE/inworld-voice-isolator/denoiser.safetensors \
  --no-background --float32 --output-sr 0

# 流式模式（仅写人声，默认 PCM_16，lookahead=2 即 20 ms 前瞻）
.venv/bin/voice-isolate \
  /Users/boom/workspace/LightClear/assets/audio/noisy/input.wav \
  outputs/cli/input_stream.wav \
  --model /Users/boom/Model/SE/inworld-voice-isolator/denoiser_realtime.safetensors \
  --mode streaming --lookahead 2
```

### 3. 目录树批处理、分片与断点续传

`voice-isolate` 传入目录时会递归扫描 `.wav/.flac/.mp3/.ogg/.opus`（保持相对目录层级），输出目录不得位于输入目录内，并自动跳过已完整产出（人声 + 背景均存在）的条目实现断点续传：

```bash
# 2 个分片并发或分批处理
.venv/bin/voice-isolate /path/to/in_dir outputs/cli_batch \
  --model /Users/boom/Model/SE/inworld-voice-isolator/denoiser.safetensors \
  --shard 0 --num-shards 2

.venv/bin/voice-isolate /path/to/in_dir outputs/cli_batch \
  --model /Users/boom/Model/SE/inworld-voice-isolator/denoiser.safetensors \
  --shard 1 --num-shards 2

# 重复运行同一分片会自动跳过已完成文件（打印 "N file(s), 0 to do"）
```

> 实测提示：CLI 离线模式在 macOS CPU 上会自动将 PyTorch 线程限至 2（`_MACOS_CPU_THREADS = 2`），而 Apple Silicon + `torch >= 2.6` 默认自动启用 `mps`。

## 验证与文档

```bash
.venv/bin/python -m tests.test_voice_isolator_import
```

模型索引、完整规格、Apple M4 Pro 实测数据、能力边界与横向对比见：

- [`docs/模型列表.md`](./docs/模型列表.md)
- [`docs/models/inworld-voice-isolator.md`](./docs/models/inworld-voice-isolator.md)
