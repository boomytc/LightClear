# light_clearvoice Instructions

## Scope

- 本文件只适用于 `explore/light_clearvoice` 目录内的工作。
- 将本目录视为独立 Python 项目根目录。在本模块目录内编辑、运行、测试、安装；从仓库根进入则 `cd explore/light_clearvoice`。
- 不要使用仓库根目录作为本项目 Python 根目录。
- 本模块是 ClearVoice 这一份上游运行时的探索家族，同时覆盖语音增强、说话人分离、语音超分辨率和音视频目标说话人提取。不要再拆成多个 `light_*` 任务模块。Demucs 在兄弟目录 `light_demucs`，不要折进来。

## Environment

- 创建并使用本目录内的 `.venv`：`uv venv --python 3.12`。
- 运行优先使用 `.venv/bin/python`。
- 安装：`uv pip install -e .`。
- macOS 或无 NVIDIA GPU 按 CPU 链路验证；不要默认假设 CUDA 可用。

## Layout

- Python 包源码位于 `third_party/clearvoice`，公共入口是 `from clearvoice import ClearVoice`。
- 模型索引见 `docs/模型列表.md`；各模型文档在 `docs/models/`。本模块不保留本地 `models/` 权重目录。
- 中心模型目录：
  - SE / TSE：`/Users/boom/Model/SE/<ModelName>`
  - SS：`/Users/boom/Model/SS/<ModelName>`
  - SR：`/Users/boom/Model/SR/<ModelName>`
- 推理配置在 `third_party/clearvoice/config/inference/`，其中 `checkpoint_dir` 必须与中心目录约定一致。yaml 里的 `input_path` / `output_dir` 只是配置占位，真正输入输出由 `ClearVoice(...)` 调用参数决定。
- 共享样例使用仓库根 `assets/` 的绝对路径。不要在本模块再拷一份共享 wav。
- Demo 位于 `demo/`。输出写入本模块 `outputs/`。
- 稳定产品入口在 `../../products/<product_name>/`。不要在本模块内新增产品运行时代码。
- TSE 人脸检测权重 `third_party/clearvoice/models/av_mossformer2_tse/faceDetector/s3fd/sfd_face.pth` 不纳入版本控制。上游 S3FD 从该文件旁的 `__file__` 加载，不在 `checkpoint_dir` 里。缺文件时 `ClearVoice` 仍可导入，构造人脸检测器会直接失败，不要 `gdown`。SE / SS / SR 单工具产品与 `speech_clarity_web` 不跑 TSE，不必拷这份权重。TSE 网络权重仍走 `/Users/boom/Model/SE/AV_MossFormer2_TSE_16K`。

## Runtime Rules

- 入口脚本如果直接 `import clearvoice`，先注入本模块 `third_party/`。
- 任务名与模型名：
  - `speech_enhancement`：`MossFormer2_SE_48K`、`FRCRN_SE_16K`、`MossFormerGAN_SE_16K`
  - `speech_separation`：`MossFormer2_SS_16K`
  - `speech_super_resolution`：`MossFormer2_SR_48K`
  - `target_speaker_extraction`：`AV_MossFormer2_TSE_16K`
- 缺依赖、缺模型或路径错误应显式暴露直接错误，不要回退到仓库根或产品目录。

## Validation

- `.venv/bin/python -c "from clearvoice import ClearVoice; print(ClearVoice.__name__)"`
- `.venv/bin/python -m py_compile demo/*.py`
- `.venv/bin/python -m tests.test_clearvoice_import`
- 完整推理前确认对应中心模型目录存在。缺权重是环境限制，不是模块缺陷。

## Cleanup

- 可清理 `__pycache__`、`.pytest_cache`、`outputs/`、一次性验证产物。
- 不要清理 `demo/`、`docs/` 或 `third_party/` 中的保留内容。

## Demo 脚本规范

- `demo` 下 `.py` 脚本保持单文件独立实现，不封装任何函数，包括 `main`。
- 顺序：import → 全局关键参数 → 模型加载 → 输入构造 → 推理调用 → 结果输出。
- 关键路径使用全局变量放在文件顶部。
