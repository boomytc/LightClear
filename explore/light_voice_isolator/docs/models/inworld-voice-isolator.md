# Inworld Voice Isolator v1（`voice-isolator-v1` / `voice-isolator-v1-realtime`）

## 1. 模型简介与架构规格

Inworld Voice Isolator v1 是 Apache-2.0 开源的 48 kHz 全频带语音隔离（Denoising / Voice Isolation）模型家族，包含共享同一主干架构的两个独立 checkpoint：

- **上游源码**：[inworld-ai/voice-isolator](https://github.com/inworld-ai/voice-isolator)（v1.0.0，vendor commit `c73b3fab81a870410cbac50d2f05f25edc4d10aa`）
- **权重来源**：[inworld/voice-isolator（ModelScope）](https://www.modelscope.cn/models/inworld/voice-isolator)（与 Hugging Face `inworld/voice-isolator` SHA-256 完全一致）
- **本地权重目录**：`/Users/boom/Model/SE/inworld-voice-isolator/`

| 指标 | `voice-isolator-v1`（离线） | `voice-isolator-v1-realtime`（流式） |
| --- | --- | --- |
| 权重文件 | `denoiser.safetensors`（244 MiB / 256 MB） | `denoiser_realtime.safetensors`（222 MiB / 233 MB） |
| SHA-256 | `647335a6aeb4dc706b895532c209d51e0e84a4849d17b339f00d4392a050ae53` | `76598f05523c0669e76562a8b2f31c9d395fca316e80c907ee11775e7497ae0f` |
| 精确参数量 | `64,047,112`（64.05M） | `58,178,056`（58.18M） |
| 采样率 / STFT | 48 kHz，`n_fft=2048`，`hop_length=480`（10 ms/帧，1025 频点） | 同左 |
| 频带划分 | 46 个非均匀子带（低频细、高频宽），每带线性映射至 `width=512` | 同左 |
| 主干结构 | 7 层交替 Dual-Path：`T-M-T-M-T-M-T`（4 层 RoPE + Exclusive Time/Band Attention，3 层双向 Time/Freq Mamba，state=16） | 7 层交替 Dual-Path：`T-M-T-M-T-M-T`（时间轴因果 Attention + 因果前向 Mamba；频率轴仍双向），带 `lookahead_embedding` |
| 时间上下文 | 双向，默认分块 `chunk_seconds=8.0` / `overlap_seconds=2.0` | 因果，训练 `attention_window_frames=1200`（12 s）；默认双轨滑窗 `window_frames=600` / `crossfade_frames=200`（6 s / 2 s） |
| 前瞻与算法延迟 | 离线无前瞻概念 | `lookahead_frames ∈ {0, 2, 4}`（0 / 20 / 40 ms），对应总算法延迟 `2048 / 3008 / 3968` samples（**42.67 / 62.67 / 82.67 ms**） |
| 解码与输出形式 | 逐带头输出有界复数掩码增量 + 有界残差（`correction_bound=0.5`），iSTFT 后与输入波形残差相加（`audio + delta`） | 同左；流式额外使用 `_PackedBands` 将 46 个子带打包为单次 batched GEMM 降低 GPU dispatch 开销 |

> 注：两份权重通过 safetensors metadata 内嵌 `config` JSON 描述结构，`load_model()` 始终以权重内配置为准，不靠外部猜架构。

---

## 2. 使用方式与关键机制验证

所有接口均已通过 `demo/` 下 6 个单文件顶层脚本和 CLI 实测验证：

1. **离线单文件与精确重构（`demo_offline_single.py`）**：
   - `VoiceIsolator(ckpt, mode="offline").isolate(wav, sr, output_sample_rate=48000)` 保留原始电平。
   - 以 `background = aligned_mono - voice` 导出背景残差并保存为 `FLOAT` WAV，实测重构误差 `max(|aligned_mono - (voice + background)|) = 0.000e+00`。
2. **离线全参数与输入形态（`demo_offline_options.py`）**：
   - `output_sample_rate=None` 保持输入采样率（如 16 kHz），传 `48000` 输出全频带 48 kHz（内部使用 `kaiser_best` sinc 重采样）。
   - `chunk_seconds=4.0, overlap_seconds=1.0` vs 默认 `8.0 / 2.0` 在 12 s 音频上 `rms_diff=0.00189`，自定义更短分块可略降耗时但上下文变短。
   - `offline_passes=2` 会将模型输出再过一次模型，耗时翻倍；对常规带噪语音改动极小（`rms_diff ≈ 5e-5 ~ 1.7e-4`），但在长静音底噪段（如 `speech1.wav`）可进一步压低底噪约 30 dB。
   - **立体声降混**：直接传入 `[T, 2]` 数组与手动 `mean(axis=1)` 单声道输入输出误差 `0.000e+00`。
   - **类型保持**：`np.ndarray -> np.ndarray(float32)`，`torch.Tensor -> torch.Tensor(float32)`。
   - **PCM16 防削波**：`save_audio(..., subtype="PCM_16")` 对峰值 > 1.0 的波形（如 `1.350`）自动按峰值缩放至 `1.00000` 避免硬削波；`subtype="FLOAT"` 原样保留 `1.350`。
3. **批处理与并发流式（`demo_batch_many.py`）**：
   - 离线 `isolate_many()` 对等长音频同批前向，对异长音频按长度自动分组并在返回时严格恢复原输入顺序，与逐条 `isolate()` 最大误差 `<= 1e-7`。
   - 流式 `isolate_many()` 按长度降序打包（每批最多 `STREAM_BATCH=16`）并以 `lengths` 尾部补零执行 lockstep 并发流式，与逐条流式最大误差 `<= 3e-7`。
4. **实时流式与高阶旋钮（`demo_streaming_realtime.py` / `demo_streaming_knobs.py`）**：
   - **片长无关性**：20 ms（960 samples）与非整帧片长（333 samples）推流输出最大误差 `2.76e-7`。
   - **多路并发 `lengths`**：`session.push([2, chunk])` 传入每路真实长度，与各自单路流式最大误差 `<= 8.94e-8`。
   - **`block_frames` 吞吐旋钮**：`block_frames=1 / 4 / 16` 的算法延迟恒为 `3008 samples (62.67 ms)`，数值误差 `<= 3.06e-7`，但 MPS 上将 `block_frames` 从 `1` 提高到 `4` 可使 RTF 从 `1.030` 骤降至 `0.379`。
   - **双轨滑窗 `window_frames=600 / crossfade_frames=200` vs `window_frames=0`**：在 18 s 音乐人声上，切轨边界帧严格落在 `[600, 1200, 1800]`；前 6 s 两种配置几乎一致（RMS diff `0.00163`），到 12–18 s 时关闭滑窗会产生明显长程注意力漂移（RMS diff 扩大至 **`0.05232`**），且默认双轨滑窗在 18 s 上反而略快（RTF `0.423` vs `0.437`），因为 KV 缓存被限制在 `<= 800` 帧内。
5. **白盒底层链路（`demo_lowlevel_network.py`）**：
   - 显式调用 `stft -> encode -> features -> denoise_decoder -> correction -> istft` 再与输入波形相加，与 `network(audio)` 最大误差 `0.000e+00`。
   - Apple M4 Pro（`torch 2.15.0`）上实测 `scan.metal_kernel()` 命中并成功编译（`14.3 ms`），`scan.kernel()`（CUDA `mamba_ssm`）为 `None`。
6. **CLI 与断点续传**：
   - `voice-isolate-doctor` 与 `--full --offline-model ... --online-model ...` 均在 `mps` 上通过。
   - `voice-isolate` 单文件实测覆盖默认双轨 float32、`--background` 自定义残差路径、`--no-background`（默认 `PCM_16`）、`--no-background --float32 --output-sr 0`（保持 16 kHz float32）、`--mode streaming --lookahead 2`。
   - 目录树批处理保持嵌套子目录结构，`--shard 0 --num-shards 2` 与 `--shard 1 --num-shards 2` 正确分片，重跑已完成的 shard 输出 `2 file(s), 0 to do` 实现安全断点续传。

---

## 3. 硬件实测（Apple M4 Pro，单任务串行）

测试环境：macOS arm64，Apple M4 Pro，Python 3.12.13，`torch 2.15.0`，`torchaudio 2.11.1`，`soundfile 0.14.0`。

### 3.1 冷启动与精度实测（`mps`）

- **Metal Shader 编译耗时**（`scan.metal_kernel()`）：`0.0143 s`（14.3 ms）
- **权重加载耗时**（`denoiser.safetensors` → `mps`）：`0.498 s`
- **首帧前向 vs 稳态前向**（`p232_005.wav`，6.247 s 48 kHz）：首次推理 `1.871 s`（含 MPS 算子缓存预热），第二次 `1.024 s`，三次稳态均值 `1.414 s`（**RTF = `0.226`**，最快 `1.294 s` / RTF `0.207`）。
- **MPS 精度对比**（`input.wav`，2.398 s，16 kHz → 48 kHz）：

| 请求 `precision` | 实际精度 | 稳态耗时（s） | vs `fp32` 最大误差 | 结论 |
| --- | --- | --- | --- | --- |
| `auto`（推荐） | `fp32` | `0.361` | `0.00e+00` | Apple Silicon 最快且无数值风险 |
| `fp32` | `fp32` | `0.364` | 基准 | 与 `auto` 一致 |
| `bf16` | `bf16` | `0.376`（首跑含图编译 `1.150`） | `2.86e-04` | 稳态略慢于 `fp32` 且精度略有损失，无收益 |
| `fp16` | `fp16` | `0.379`（首跑 `1.267`） | **`NaN`（数值溢出）** | **禁止在 Apple Silicon MPS 上使用 `fp16`** |

### 3.2 `mps` vs `cpu` 稳态速度与实时性判定

`mps` 在 6.247 s 音频（`p232_005.wav`）上测稳态；`cpu` 在 2.398 s 音频（`input.wav`）上分别测默认 10 线程与 2 线程（`cli.py` 在 macOS CPU 上主动限 2 线程以降低同步开销）：

| 模型与模式 | 设备 / 配置 | 推流片长 / 块 | 算法延迟 | 实测 RTF | 能否实时（RTF < 1） |
| --- | --- | --- | --- | --- | --- |
| 离线 `voice-isolator-v1` | `mps` (`fp32` + Metal Shader) | 整段（<= 8 s） | — | **0.226** | 离线约 4.4x 实时 |
| 离线 `voice-isolator-v1` | `cpu`（默认 10 线程） | 整段 | — | 22.49 | 否（慢 ~100x） |
| 离线 `voice-isolator-v1` | `cpu`（2 线程） | 整段 | — | 24.57 | 否 |
| 流式 `v1-realtime` (`la=0`) | `mps` | 20 ms / 100 ms / 500 ms | 42.67 ms | **0.709 / 0.426 / 0.378** | **稳定满足（< 1）** |
| 流式 `v1-realtime` (`la=2`) | `mps` | 20 ms / 100 ms / 500 ms | 62.67 ms | **0.677 / 0.444 / 0.385** | **稳定满足（< 1）** |
| 流式 `v1-realtime` (`la=4`) | `mps` | 20 ms / 100 ms / 500 ms | 82.67 ms | **0.677 / 0.499 / 0.392** | **稳定满足（< 1）** |
| 流式 `v1-realtime` (`la=2`) | `mps` (`block_frames=1`) | 200 ms | 62.67 ms | 1.030 | 临界超标（每帧一次 dispatch） |
| 流式 `v1-realtime` (`la=2`) | `mps`（极短片 333 采样 ≈ 6.9 ms） | 6.9 ms | 62.67 ms | 1.208 | 超标（dispatch 过密） |
| 流式 `v1-realtime` (`la=2`) | `cpu`（默认 10 线程） | 20 ms / 500 ms | 62.67 ms | 44.80 / 23.60 | 否 |
| 流式 `v1-realtime` (`la=2`) | `cpu`（2 线程） | 20 ms / 500 ms | 62.67 ms | 22.89 / 12.17 | 否（比 10 线程快 ~2x，但仍远大于 1） |

**核心结论**：

- 在 Apple M4 Pro 上，只要安装 `torch >= 2.6` 启用 `mps` + Metal Shader，且保持推流片长 `>= 20 ms`、`block_frames >= 4`，流式模型在 `lookahead=0/2/4` 下均可稳定达到 **`RTF ≈ 0.38–0.73 < 1.0`**，满足真实时流式隔离需求。
- 若退回 CPU，即使按上游 `cli.py` 在 macOS 上将线程限至 2（相比默认多线程快约 4 倍），`p232_005.wav`（6.247 s）离线 RTF 仍为 `4.31`、20 ms 流式 RTF 仍为 `11.18`，仅适合无 GPU 兜底或数值对齐参考，不可用于实时流式。

### 3.3 Colab NVIDIA L4 CUDA 实测（`cuda:0`，`bf16`，无 `mamba_ssm` 融合算子回退）

测试环境：Colab `NVIDIA L4`（22.03 GiB VRAM，驱动 `580.82.07`），Linux x86_64，Python 3.12，`torch 2.15.0+cu132`（CUDA 13.2）。由于 PyTorch 2.15 + cu132 尚无匹配预编译 `mamba_ssm` wheel，本次 CUDA 实测走上游内置的 PyTorch 选择性扫描回退路径（`scan.kernel() == None`）：

| 场景 / 样例 | 精度 | 冷启动耗时 | 稳态耗时 / RTF（实时倍率） | L4 峰值显存（alloc / reserved） |
| --- | --- | --- | --- | --- |
| 离线单文件 `p232_005.wav`（6.247 s） | `bf16`（`auto`） | `2.030 s`（RTF `0.3250`） | **`1.384 s` / RTF `0.2216`（4.51x）** | `5150.1 / 5716.0 MiB` |
| 离线短句 `input.wav`（2.398 s）：`fp32` / `bf16` / `fp16` | 对比 | — | `0.608 s`（`0.2534`） / **`0.449 s`（`0.1872`）** / `fp16` **NaN 溢出** | `6535.9 / 7546.0 MiB`（全项峰值） |
| 离线同长合批 3×2.0 s（6.0 s）vs 逐条 | `bf16` | `1.748 s` | 批 `1.335 s`（`0.2225`） vs 逐条 `1.125 s`（`0.1876`） | `5260.1 / 7188.0 MiB` |
| 流式 20 ms 逐块 `p232_005.wav`（6.247 s，`la=2`） | `bf16` | `11.790 s`（`1.8875`） | `11.423 s` / RTF `1.8287`（`0.55x`，**无 `mamba_ssm` 下不实时**） | `2519.5 / 2782.0 MiB` |
| 流式 200 ms 分片 `block_frames=1 / 4 / 16`（6.247 s） | `bf16` | `6.083 s`（预热） | `21.001 s`（`3.3619`） / `5.649 s`（`0.9043`） / **`2.393 s`（`0.3831`）** | `2499.3 / 3182.0 MiB` |
| 4 路异长流式 lockstep 批处理（总 7.46 s） | `bf16` | `4.508 s` | **`3.815 s` / RTF `0.5117`**（对比逐条 `7.032 s` / `0.9432`） | `5260.1 / 7188.0 MiB` |

**CUDA 工程要点（供产品选型参考）**：

- **禁用 `fp16`**：即使在 NVIDIA L4 上，强开 `fp16` 同样出现 NaN 数值溢出；CUDA 必须保持默认 `precision="auto"`（自动走 `bf16`，相比 `fp32` 提速约 26% 且最大误差仅 `3.35e-4`，需要 Ampere/Ada 及以上原生 BF16 架构）。
- **流式必须关注 `mamba_ssm` 或 `block_frames`**：无 `mamba_ssm` CUDA 融合算子时，20 ms 单路流式（Python/PyTorch 逐帧回退）在 L4 上 `RTF = 1.83`（反而不如带编译 Metal Shader 的 Apple M4 Pro MPS `RTF = 0.73`）；生产 CUDA 流式服务要么固定匹配预编译 `mamba_ssm` wheel 的 Torch/CUDA 版本，要么在可接受分片场景下提高 `block_frames`（如 `16` 时 `RTF = 0.38`）或使用多路 lockstep 合批（4 路总吞吐 `RTF = 0.51`）。

---

## 4. 模型能力与真实边界实测（基于 LightClear 共享样例）

### 4.1 带噪语音去噪与格式兼容性（`assets/audio/noisy/`）

- **音频格式支持**：`soundfile 0.14.0` 可直接读取 `.wav`、`.flac`、`.mp3`、`.ogg`、`.aiff`，但 **`speech2.aac` 无法读取**（libsndfile 报 `Format not recognised`，需外部 `ffmpeg` 转码）。此外，CLI 目录批处理 `_EXTS` 仅扫描 `.wav/.flac/.mp3/.ogg/.opus`（跳过 `.aiff`）。
- **同源多格式一致性（`speech1.{wav,flac,mp3,ogg}`）**：`flac` 与 `wav` 输出 SNR `34.73 dB`，`ogg` 与 `wav` SNR `39.32 dB`；`mp3` 因编码器前置延迟导致时长多出 78 ms（18.469 s vs 18.391 s），逐采样对齐会错位，但能量、底噪降幅（43.14 vs 41.67 dB）和 8–24 kHz 高频能量（`-51.03` vs `-51.02 dB`）几乎完全一致。
- **典型样例离线 vs 流式与二次过模型实测**：

| 样例 | 原采样率 / 时长 | 离线 `p=1` 底噪降幅 / 语音电平变化 / RTF | 离线 `p=2` 底噪降幅 / 与 `p=1` RMS 差 | 流式 `la=0/2/4` vs 离线 SI-SNR（dB） | 观察 |
| --- | --- | --- | --- | --- | --- |
| `p232_005.wav` | 48k / 6.25 s | **39.64 dB** / `-1.37 dB` / `0.215` | 39.65 dB / `9e-5` | `27.13 / 27.72 / 27.61` | 强稳态噪声压制彻底，人声与 8–24 kHz 齿音空气感保持极佳 |
| `p232_007.wav` | 48k / 3.96 s | **32.57 dB** / `-0.15 dB` / `0.166` | 32.68 dB / `1e-4` | `28.28 / 28.57 / 29.12` | 轻中度噪声，人声几乎无电平偏移，`la=4` 最贴近离线 |
| `input.wav` | 16k / 2.40 s | **38.29 dB** / `-0.18 dB` / `0.181` | 38.29 dB / `5e-5` | `30.00 / 30.48 / 30.57` | 16 kHz 带噪语音高度一致，流式与离线 SI-SNR > 30 dB |
| `speech1.wav` | 44.1k 双声道 / 18.39 s | **41.67 dB** / `-2.13 dB` / `0.265` | **71.68 dB** / `1.7e-4` | `17.96 / 18.61 / 18.71` | 长语音场景下 `la=2/4` 明显优于 `la=0`；`p=2` 仅把极弱停顿底噪再压 30 dB |
| `aviation_glideslope_too_low.wav` | 16k / 2.35 s | **38.63 dB** / `+0.44 dB` / `0.266` | 40.47 dB / `1.94e-3` | **`-33.0 / -24.2 / -21.9`（告警被误删！）** | **重要边界发现**：重驾舱噪声下的合成电子 GPWS 语音告警，离线模型保留良好（`-19.91 dBFS`），但因果流式模型将其误判为非人声告警全部抹除（跌至 `-42 ~ -53 dBFS`） |
| `aviation_decision_height.wav` | 16k / 8.08 s | **44.93 dB** / `-7.98 dB` / `0.360` | — | — | 驾舱强持续告警音中部分电子语音被衰减 |

### 4.2 频带忠实度与非修复边界（`assets/audio/bandlimited/`）

对照上游 VIBE 榜单区分 Denoising 与 Restoration 的硬判据（输出是否在原输入奈奎斯特频率以上凭空增加 > 10 dB 相对能量），在 48 kHz 输出下同台实测 `voice-isolator-v1` 与 ClearVoice `MossFormer2_SR_48K`：

| 样例 | 原采样率 / 截止频率 | 输入带外能量 | `voice-isolator-v1` 带外能量 | `voice-isolator-v1` 带内 SNR / SI-SNR | `MossFormer2_SR_48K` 带外能量（对比） |
| --- | --- | --- | --- | --- | --- |
| `input_sr.wav` | 16 kHz / 8 kHz | `-146.24 dB` | **`-141.86 dB`**（仍低于带内 122 dB） | `58.08 / 60.55 dB` | **`-37.02 dB`**（相对抬升 **`+109.74 dB`**） |
| `LJ001-0001.wav` | 22.05 kHz / 11 kHz | `-125.56 dB` | **`-123.53 dB`**（相对变化 `+2.03 dB`） | `56.62 / 56.89 dB` | — |
| `LJ001-0002.wav` | 22.05 kHz / 11 kHz | `-120.46 dB` | **`-126.52 dB`**（相对变化 `-6.06 dB`） | `50.98 / 51.35 dB` | — |
| 合成 4 kHz 砖墙低通 `LJ001` | 48 kHz / 4 kHz | `-154.14 dB` | **`-135.96 dB`**（仍低于带内 124 dB，仅 STFT 数值底噪） | `27.24 / 27.45 dB` | — |

**结论**：Voice Isolator v1 严格遵守 Denoising 契约——**绝不在原输入奈奎斯特频率以上捏造高频谐波**。需要带宽扩展/超分时必须使用 `MossFormer2_SR_48K`（或等待上游 v2）。

### 4.3 竞争说话人边界（`assets/audio/mixture/` + 对称音量差扫描）

1. **共享混叠样例（双说话人响度接近）**：
   - `050a0502_1.3461_440o030j_-1.3461.wav`：输出保留 **99.3%** 总能量（残差仅 3.1%，仅去底噪）。
   - `input_ss.wav` / `speech_mixure1.wav`：输出保留 **97.2%** 总能量，双说话人均完整保留，**不会拆成多轨**。
2. **可控对称 SIR 扫描**（两条同等 16 kHz→48 kHz 带宽的单人干净语音 A / B 按不同相对响度混合，最小二乘投影测各说话人保留增益）：

| 输入 `SIR(A/B)` | 说话人 A 保留增益 | 说话人 B 保留增益 | 实际行为 |
| --- | --- | --- | --- |
| `-15 dB`（B 明显更响） | `0.215`（`-13.35 dB`） | `0.996`（`-0.03 dB`） | 剔除弱者 A，保前景 B |
| `-10 dB` | `0.365`（`-8.75 dB`） | `0.994`（`-0.05 dB`） | 明显压制 A，保前景 B |
| `-6 dB` | `0.541`（`-5.34 dB`） | `0.991`（`-0.08 dB`） | 过渡带，开始衰减弱者 A |
| `-3 dB` | `0.979`（`-0.18 dB`） | `0.998`（`-0.02 dB`） | **响度接近：双双完整保留** |
| `0 dB` | `0.999`（`-0.01 dB`） | `1.000`（`0.00 dB`） | **响度相等：双双完整保留** |
| `+3 dB` | `0.999`（`-0.01 dB`） | `0.999`（`-0.01 dB`） | **响度接近：双双完整保留** |
| `+6 dB` | `0.998`（`-0.01 dB`） | `0.661`（`-3.59 dB`） | 过渡带，开始衰减弱者 B |
| `+10 dB` | `0.993`（`-0.06 dB`） | `0.199`（`-14.02 dB`） | 明显压制 B，保前景 A |
| `+15 dB`（A 明显更响） | `1.000`（`0.00 dB`） | `0.240`（`-12.41 dB`） | 剔除弱者 B，保前景 A |

> 额外观察：若两位说话人录制距离/频带显著不同（如 48 kHz 近讲男声 `p232_005` 混合 22 kHz 播音女声 `LJ001`），即使 RMS 相等（`0 dB`），模型也会优先保留近讲全频带前景声（`-0.31 dB`）并将另一位压低 `-22.54 dB`。因此本模型适合「保主讲人、去远场背景人声」，**不能替代 `MossFormer2_SS_16K` 做双人独立分轨**。

### 4.4 音乐 / 人声残差分离（`assets/audio/music/`）vs `htdemucs`

| 样例 | 模型与设备 | 输出声道 / 采样率 | 实测 RTF | 人声轨 RMS | 伴奏/残差 RMS | 边界观察与滞后对齐相关性 |
| --- | --- | --- | --- | --- | --- | --- |
| `next_station_heaven.mp3`（30 s DJ 混音带唱） | `voice-isolator-v1`（MPS） | 单声道 / 48 kHz | `0.247` | `-19.45 dBFS` | `-7.52 dBFS` | 120 Hz 以下低频鼓点/贝斯削减 **48.99 dB**；把演唱混响、和声与电子叠层剥入残差，人声偏干且比 Demucs 低 4 dB；与 `htdemucs` 伴奏相关性 `0.712`（声部相关性 `0.253`） |
| `next_station_heaven.mp3` | `htdemucs`（CPU） | 立体声 4 轨 / 44.1 kHz | `0.217` | `-15.41 dBFS` | `-7.94 dBFS` | 完整保留立体声主唱质感与音乐混响，并独立拆分 `drums/bass/other` |
| `test.mp3`（20 s 极低人声烟雾片） | `voice-isolator-v1`（MPS） | 单声道 / 48 kHz | `0.245` | `-53.28 dBFS` | `-16.31 dBFS` | 将纯器乐压低 37 dB，仅留极弱残余 |
| `test.mp3` | `htdemucs`（CPU） | 立体声 4 轨 / 44.1 kHz | `0.203` | `-78.38 dBFS` | `-16.34 dBFS` | 对无人声器乐的静音抑制更彻底（低 25 dB） |

**结论**：Voice Isolator 的 `aligned_mono - voice` 适合「语音节目去背景音乐 / 快速获取单声道背景残差」，但**不保留立体声声场、不拆多乐器、且会按“去混响干人声”标准剥离歌唱混响**；音乐人声分离与分轨应优先选用 `explore/light_demucs`（`htdemucs` / `htdemucs_ft` / `htdemucs_6s`）。

### 4.5 极端边界探针（干净过水、纯噪/静音、长音频滑窗）

1. **干净语音过水无损性**：
   - `LJ001-0001.wav`（离线）：**SNR `56.62 dB` / SI-SNR `56.89 dB`**，RMS 偏移 `+0.003 dB`，4–10 kHz 频段偏移 `-0.01 dB`，活跃帧衰减 > 6 dB 比例 **`0.0%`**（零吞字）；流式 `la=2` 同样达 `57.68 / 57.69 dB`。
   - 48 kHz 干净语音二次过水：SNR **`59.50 dB`**，8–20 kHz 高频偏移仅 `-0.01 dB`（近乎完全透明幂等）。
2. **纯静音与非语音输入（幻觉检测）**：
   - **纯静音（3 s 全零）**：离线与流式输出均为 **精确 `0.000000`**（`-120 dB`，零幻觉）。
   - **白噪声（-20 dBFS）**：离线压制 **52.16 dB**（峰值 `0.0019`），流式压制 **63.99 dB**（峰值 `0.0003`）。
   - **1 kHz 纯音（-20 dBFS）**：离线压制 **70.98 dB**，流式压制 **62.54 dB**。
   - **粉红噪声（-20 dBFS，1/f 类语音谱斜率）**：离线压制 **40.49 dB**（峰值 `0.0047`）；但 **流式 `la=2` 仅压制 `13.42 dB`（偶发瞬态峰值 `0.141`）**，说明强低频类语音噪声下因果流式模式存在少量漏过风险。
3. **24 s 长音频滑窗接缝稳定性**：
   - 离线（8 s / 2 s 线性淡入淡出）与流式默认双轨滑窗（600 / 200 帧）在 6 s / 12 s / 18 s 边界处的最大相邻采样跳变几乎一致（`0.022–0.071`，无爆音台阶）。
   - 逐 6 s 分段与离线基准对比：开启双轨滑窗（`600/200`）全程 SI-SNR 稳定在 `27.34–28.19 dB`，关闭滑窗（`window_frames=0`）在 6–18 s 区间下降约 `1.1 dB`（26.27–26.40 dB）；而在强音乐背景 18 s 音频上，关闭滑窗在 12–18 s 的漂移 RMS 差异高达 `0.05232`，**生产长流必须保持默认双轨滑窗开启**。

---

## 5. 与 LightClear 现有模型家族的横向选型对照

### 5.1 同台实测：Inworld `voice-isolator-v1` vs ClearVoice `MossFormer2_SE_48K`

| 对比维度 | Inworld `voice-isolator-v1`（离线） | Inworld `v1-realtime`（流式 `la=2`） | ClearVoice `MossFormer2_SE_48K` |
| --- | --- | --- | --- |
| 实测底噪压制（`p232_005` / `p232_007`） | **39.64 / 32.57 dB** | **48.05 / 30.83 dB** | 32.06 / 22.71 dB |
| 48 kHz 高频保留（`p232_005` / `p232_007` 8–24 kHz） | **-47.68 / -54.62 dB**（保留更好） | 与离线高度接近 | -48.48 / -56.60 dB（略偏暗） |
| 干净语音过水保真（`LJ001-0001` SNR / SI-SNR） | **`56.62 / 56.89 dB`** | **`57.68 / 57.69 dB`** | `45.18 / 45.54 dB`（低 ~11.4 dB） |
| 航空电子合成语音告警（`aviation_glideslope`） | 语音保留佳（`+0.44 dB`） | **误删告警（-52.6 dBFS）** | 保留但略衰减（`-2.68 dB`） |
| 低延迟因果流式支持 | 不支持 | **原生支持（62.67 ms 延迟）** | 不支持 |
| Apple M4 Pro `mps` RTF | **0.17–0.26**（Metal Shader） | **0.38–0.71**（20–500 ms 分片） | 当前 explore 按 CPU 链路运行 |
| Apple M4 Pro `cpu` RTF | 22.5–24.6（不建议纯 CPU） | 12.2–44.8（不可实时） | **0.41–0.78**（纯 CPU 显著更轻快） |

### 5.2 全工作区选型矩阵

| 场景需求 | 首选模型家族 | 原因 |
| --- | --- | --- |
| **Apple Silicon MPS / GPU 高保真 48 kHz 录音降噪、原音色与呼吸/笑声保留、干净段零损伤** | **`explore/light_voice_isolator`（`voice-isolator-v1`）** | 干净语音过水 56.6 dB SNR、电平完全保持、底噪压制比 `MossFormer2_SE_48K` 深 7–10 dB、精准残差可重构 |
| **实时通话 / 麦克风推流低延迟降噪（需 `RTF < 1`）** | **`explore/light_voice_isolator`（`voice-isolator-v1-realtime`）** | 原生因果流式、62.67 ms 算法延迟、M4 Pro MPS 20 ms 分片 RTF `0.68`（注意避开强合成电子告警音） |
| **纯 CPU 机器（无 GPU / 无 MPS）离线 48 kHz 语音增强** | **`explore/light_clearvoice`（`MossFormer2_SE_48K`）** | CPU RTF `0.4–0.8` 远轻于 Voice Isolator CPU（RTF > 22） |
| **低采样率 / 电话窄带语音高频修复（带宽扩展）** | **`explore/light_clearvoice`（`MossFormer2_SR_48K`）** | 实测对 16 kHz 输入可在 8.5–23.5 kHz 重建 `+109.7 dB` 高频能量；Voice Isolator v1 严格不做修复 |
| **双说话人等响度混叠独立拆轨** | **`explore/light_clearvoice`（`MossFormer2_SS_16K`）** | Voice Isolator 在 `SIR ∈ [-3, +3] dB` 时双留不拆轨，仅适合压制弱背景人声 |
| **歌曲立体声人声提取与多乐器分轨** | **`explore/light_demucs`（`htdemucs` / `htdemucs_ft` / `htdemucs_6s`）** | 保留立体声唱腔与混响、支持四轨/六轨且 CPU RTF 仅 `0.21` |
