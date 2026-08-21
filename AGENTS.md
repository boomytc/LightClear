# LightClear Workspace Instructions

## Scope

- 使用与用户提问相同的语言回复；修改前先看目标模块或产品的现有代码、README 和本地 `AGENTS.md`。
- 仓库根只做路由与规范，不是可安装 Python 项目，不承担统一依赖或共享运行时。
- 将每个 `explore/light_*` 目录视为独立 Python 项目根。编辑、运行、测试、安装或调试前，先 `cd` 到该目录并遵循其 `AGENTS.md`。
- 将每个 `products/<product_name>/` 视为独立产品根。编辑、运行、测试、安装或调试前，先 `cd` 到该目录并遵循其 `AGENTS.md`。
- `explore/` 是音频变清晰探索层。当前家族：`explore/light_clearvoice`（ClearVoice：SE / SS / SR / TSE）与 `explore/light_demucs`（Demucs：人声隔离 / 分轨）。新上游开新的 `explore/light_*`，不要按任务拆 ClearVoice，也不要折进已有家族。
- 根目录 `products/` 是稳定应用层。产品运行时代码必须自包含，不得 import explore，不得读取仓库根或兄弟目录的 `third_party/`、`.venv`。
- 不要在仓库根重新引入 `pyproject.toml`、`third_party/`、`models/` 或 `demo/`。
- 根目录 `assets/` 是 explore 共享样例池（见 `assets/README.md`）。产品运行时不得读取；需要样例时拷进产品自己的 `assets/`。

## Module Guides

- 根目录 `explore/` 保留 `AGENTS.md`，定义探索模块共享边界。
- 每个 `explore/light_*` 模块保留自己的 `AGENTS.md`。
- 根指南只放共享边界；模型路径、安装命令、入口脚本放到模块自己的指南。

## Product Guides

- 根目录 `products/` 保留 `AGENTS.md`。
- 每个产品保留自己的 `AGENTS.md`、`README.md`、`pyproject.toml`、`third_party/`、样例和启动说明。

## Python Environment

- 每个探索模块和每个产品使用各自目录内的 `.venv`。
- 创建环境：在目标目录执行 `uv venv --python 3.12`。
- 依赖只安装到当前目录：`uv pip install -e .`。
- 不要使用仓库根 `.venv` 运行模块或产品代码。
- macOS 或无 NVIDIA GPU 按 CPU 链路验证。

## Refactor Rules

- 保持 vendored runtime 在模块或产品本地 `third_party/`。
- ClearVoice 中心模型目录：`/Users/boom/Model/{SE,SS,SR}/`。Demucs 权重各放在使用方自己的 `models/htdemucs/`（explore 一份、产品一份），运行时互不读取。缺权重直接失败，不要回退家庭缓存。以后若有第二个产品要用，再单独做指向。
- 允许在当前模块 `third_party/` 上做该家族需要的上游修改。官方更新后单独判断是否同步，不要自动覆盖，也不要改兄弟模块的副本。优先在 demo 或产品层适配。
- 缺依赖、缺模型或路径错误应显式暴露直接错误，不要回退到旧根路径。
- 不要添加共享 SDK 或把各产品 WebUI 合成全模型管理台。

## Demo 脚本规范

- `demo` 下 `.py` 脚本保持单文件独立实现，不封装任何函数，包括 `main`。
- 顺序：import → 全局关键参数 → 模型加载 → 输入构造 → 推理调用 → 结果输出。
- 公共入口以该模块 `AGENTS.md` 为准：`light_clearvoice` 用 `from clearvoice import ClearVoice`；`light_demucs` 用 `from demucs.api import Separator`。

## Cleanup

- 可清理 `__pycache__`、`.pytest_cache`、`outputs/`、`workspace/outputs/`、一次性验证产物。
- 不要清理 `demo/`、`docs/`、`third_party/`、产品 `frontend/` 或共享 `assets/` 中的保留内容。
