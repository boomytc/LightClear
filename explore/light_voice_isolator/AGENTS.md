# light_voice_isolator Instructions

## Scope

- 本文件只适用于 `explore/light_voice_isolator` 目录内的工作。
- 将本目录视为独立 Python 项目根目录。在本模块目录内编辑、运行、测试、安装；从仓库根进入则 `cd explore/light_voice_isolator`。
- 不要使用仓库根目录作为本项目 Python 根目录。
- 本模块是上游 Inworld Voice Isolator 运行时的独立探索家族，覆盖 48 kHz 离线语音隔离（`voice-isolator-v1`）与因果实时流式语音隔离（`voice-isolator-v1-realtime`）。不要折进 `light_clearvoice` 或 `light_demucs`，也不要在本模块内新增 `products/` 运行时代码。

## Environment

- 创建并使用本目录内的 `.venv`：`uv venv --python 3.12`。
- 运行优先使用 `.venv/bin/python`；CLI 入口使用 `.venv/bin/voice-isolate` 与 `.venv/bin/voice-isolate-doctor`。
- 安装：`uv pip install -e .`。
- Apple Silicon（macOS arm64）必须安装 `torch >= 2.6`，以启用上游内置的 MPS + 自定义 Metal Shader（`inworld.voice_isolation.scan.metal_kernel`）Mamba 选择性扫描加速；若退回 CPU，实时与离线都会明显变慢（无法满足 `RTF < 1`）。

## Layout

- Python 包源码位于 `third_party/inworld/voice_isolation`（隐式命名空间包 `inworld`），公共入口是 `from inworld import voice_isolation`。
- `third_party/inworld` 是上游 `inworld-ai/voice-isolator` v1.0.0（commit `c73b3fab81a870410cbac50d2f05f25edc4d10aa`，Apache-2.0）的 vendor 副本，并保留 `third_party/LICENSE` 与 `third_party/NOTICE`。允许在本模块副本上做必要修改，官方更新后单独判断是否同步，不要自动覆盖。
- 中心模型权重目录：`/Users/boom/Model/SE/inworld-voice-isolator/`，不纳入版本库，也不在 demo 中静默联网自动下载：
  - `denoiser.safetensors`（离线 `voice-isolator-v1`，64.05M）
  - `denoiser_realtime.safetensors`（流式 `voice-isolator-v1-realtime`，58.18M）
- 模型索引见 `docs/模型列表.md`；完整模型规格、能力边界与实测数据见 `docs/models/inworld-voice-isolator.md`。
- 共享样例使用仓库根 `assets/audio/`（demo 通过 `MODULE_ROOT.parent.parent / "assets"` 跨平台解析），不要在本模块再拷一份共享音频。
- Demo 按能力平铺在 `demo/` 下（6 个单文件顶层脚本），顶部通过 `DEVICE = os.environ.get("LIGHTCLEAR_DEVICE", "auto")` 控制设备（默认 `cuda > mps > cpu`，目标加速器不可用时警告并回退 `cpu`，落到 CPU 时提示性能限制并在 macOS 限 2 线程）；中心权重根支持 `LIGHTCLEAR_MODEL_ROOT`（默认 `/Users/boom/Model`）；输出按实际设备写入本模块 `outputs/{cuda,mps,cpu}/`。

## Runtime Rules

- 入口脚本如果直接 `import inworld`，先注入本模块 `third_party/`。
- 所有 demo 与本地调用必须显式传入 `/Users/boom/Model/SE/inworld-voice-isolator/` 下的本地 `.safetensors` 路径，不要传入 `"inworld/voice-isolator"` 触发静默 Hugging Face 联网下载。
- 缺依赖、缺权重或路径错误必须显式暴露直接错误，不做隐式回退。
- Apple Silicon `mps` 上保持默认 `precision="auto"`（自动走 `fp32`）；不要在 `mps` 上强开 `fp16`（实测出现 NaN 溢出），`bf16` 稳态略慢于 `fp32` 且无收益。

## Validation

- `.venv/bin/python -c "from inworld import voice_isolation; print(voice_isolation.VoiceIsolator.__name__)"`
- `.venv/bin/python -m py_compile demo/demo_*.py`
- `.venv/bin/python -m tests.test_voice_isolator_import`
- `.venv/bin/voice-isolate-doctor`
- `.venv/bin/voice-isolate-doctor --full --offline-model /Users/boom/Model/SE/inworld-voice-isolator/denoiser.safetensors --online-model /Users/boom/Model/SE/inworld-voice-isolator/denoiser_realtime.safetensors`

## Cleanup

- 可清理 `__pycache__`、`.pytest_cache`、`outputs/`、一次性验证产物。
- 不要清理 `demo/`、`docs/`、`third_party/` 或 `tests/` 中的保留内容。

## Demo 脚本规范

- `demo` 下 `.py` 脚本保持单文件独立实现，不封装任何函数，包括 `main`。
- 顺序：import → 全局关键参数 → 模型加载 → 输入构造 → 推理调用 → 结果输出。
- 关键路径、权重路径、样例路径和输出路径使用全局变量放在文件顶部。
