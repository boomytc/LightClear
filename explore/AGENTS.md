# LightClear Explore Instructions

## Scope

- 本文件适用于 `explore/` 下的语音增强 / 分离 / 超分 / 目标说话人提取探索模块。
- 将每个 `explore/light_*` 目录视为独立 Python 项目根目录。
- 当前探索家族只有 `light_clearvoice`（上游 ClearVoice，覆盖 SE / SS / SR / TSE）。不要按任务再拆成 `light_se` / `light_ss` / `light_sr` / `light_tse`。
- 编辑、运行、测试、安装或调试前，先 `cd` 到对应模块目录并遵循该模块自己的 `AGENTS.md`。
- `explore/` 不是共享 Python 包、共享运行时根目录或稳定产品根目录。
- 新上游家族开新的 `explore/light_*`，不要折进 `light_clearvoice`。

## Module Boundaries

- 每个探索模块拥有自己的 `AGENTS.md`、`pyproject.toml`、`third_party/`、`demo/`、文档和运行产物目录。
- 跨模块共用的样例只放在仓库根 `assets/`。模块不要再拷一份共享 wav。
- 模块运行时代码从模块目录解析路径；`sys.path` 只允许注入本模块 `third_party/`。
- 稳定 WebUI 放在根目录 `products/<product_name>/`。产品运行时不得 import explore，不得读取本目录的 `third_party/`、`.venv` 或依赖元数据。

## Environment

- 每个探索模块使用模块本地 `.venv`。
- 创建环境：在目标模块目录内执行 `uv venv --python 3.12`。
- 运行优先使用 `.venv/bin/python`。
- 依赖只安装到当前模块：`uv pip install -e .`。

## Layout

- `third_party/`：模块本地 vendored runtime。
- `demo/`：直接能力示例，单文件顶层脚本。
- `docs/models/`：按模型维护的文档。
- `outputs/`：运行产物，可清理。

## Demo 脚本规范

- `demo` 下 `.py` 脚本保持单文件独立实现，不封装任何函数，包括 `main`。
- 顺序：import → 全局关键参数 → 模型加载 → 输入构造 → 推理调用 → 结果输出。
- 关键路径、模型名、任务名、输入文件、输出路径、采样率使用全局变量放在文件顶部。
- 要体现 `from clearvoice import ClearVoice` 的真实使用方式，不把核心逻辑藏到包装层。
