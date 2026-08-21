# light_demucs Instructions

## Scope

- 本文件只适用于 `explore/light_demucs` 目录内的工作。
- 将本目录视为独立 Python 项目根目录。在本模块目录内编辑、运行、测试、安装；从仓库根进入则 `cd explore/light_demucs`。
- 不要使用仓库根目录作为本项目 Python 根目录。
- 本模块是 Demucs 这一份上游运行时的探索家族，覆盖人声隔离与音乐分轨。不要折进 `light_clearvoice`，也不要按声部再拆模块。

## Environment

- 创建并使用本目录内的 `.venv`：`uv venv --python 3.12`。
- 运行优先使用 `.venv/bin/python`。
- 安装：`uv pip install -e .`。
- macOS 或无 NVIDIA GPU 按 CPU 链路验证；不要默认假设 CUDA 可用。

## Layout

- Python 包源码位于 `third_party/demucs`，公共入口是 `from demucs.api import Separator`。
- `third_party/demucs` 是 `/Users/boom/workspace/demucs/demucs` 的 vendor 副本（上游 `adefossez/demucs` v4.1.0，`eeac1d15891af95b1288d2884b95baa3e5baa96c`）。允许在这份副本上做本模块需要的上游修改。官方仓库更新后单独判断是否值得同步，不要自动覆盖本地改动。产品 `vocal_isolate_web` 从本模块 `third_party/demucs` 再拷一份，不要从 `workspace/demucs` 直拷。
- 模型索引见 `docs/模型列表.md`；各模型文档在 `docs/models/`。
- 权重目录：本模块 `models/htdemucs/`，不进版本库。安装：`.venv/bin/python scripts/install_htdemucs.py`。缺目录直接失败。不要写家庭缓存，不要安装时自动拷进产品。产品要权重时自己指定目标目录再装一份。
- 共享样例使用仓库根 `assets/audio/music/` 的绝对路径。不要在本模块再拷一份共享音频。
- Demo 位于 `demo/`。输出写入本模块 `outputs/`。
- 稳定产品入口是 `../../products/vocal_isolate_web/`。不要在本模块内新增产品运行时代码。不要接到 `speech_separation_web`（那是说话人分离，不是分轨）。

## Runtime Rules

- 入口脚本如果直接 `import demucs`，先注入本模块 `third_party/`。
- 默认模型：`htdemucs`。清晰主路径是人声 / 伴奏两路；四轨是同一模型的完整输出。
- 缺依赖、缺模型或路径错误应显式暴露直接错误，不要回退到仓库根、`light_clearvoice` 或产品目录。

## Validation

- `.venv/bin/python -c "from demucs.api import Separator; print(Separator.__name__)"`
- `.venv/bin/python -m py_compile demo/*.py`
- `.venv/bin/python -m tests.test_demucs_import`
- 完整推理前确认本模块 `models/htdemucs/` 存在。缺权重是环境限制，不是模块缺陷。

## Cleanup

- 可清理 `__pycache__`、`.pytest_cache`、`outputs/`、一次性验证产物。
- 不要清理 `demo/`、`docs/`、`third_party/` 或 `models/README.md`。`models/htdemucs/` 下的权重可按需删掉后重装。

## Demo 脚本规范

- `demo` 下 `.py` 脚本保持单文件独立实现，不封装任何函数，包括 `main`。
- 顺序：import → 全局关键参数 → 模型加载 → 输入构造 → 推理调用 → 结果输出。
- 关键路径使用全局变量放在文件顶部。
