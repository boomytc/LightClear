# LightClear Products Instructions

## Scope

- 将每个 `products/<product_name>/` 视为独立产品根目录。
- 编辑、运行、测试、安装或调试前，先 `cd` 到对应产品目录并遵循该产品自己的 `AGENTS.md`。
- `products/` 不是共享 Python 包或共享运行时根目录。
- 产品可以参考对应 explore 家族的 demo 形状，但运行时必须自包含。

## Product Boundaries

- 每个产品拥有自己的 `pyproject.toml`、`AGENTS.md`、`README.md`、`third_party/`、`assets/`、前端资源和启动方式。
- 产品运行时代码只从产品目录解析路径。
- 产品不得读取任何 `explore/light_*` 的 `third_party/`、`.venv`、`demo/` 或依赖元数据。
- 产品不得依赖仓库根 `assets/`；需要样例时拷进产品自己的 `assets/`。
- 按场景拆入口。同一场景内可以有多件工具（`speech_clarity_web` 的增强 / 分离 / 超分），也可以有同任务质量档或输出档（`vocal_isolate_web` 的 `htdemucs` / `htdemucs_ft` / `htdemucs_6s`）。
- 三个单工具 `speech_enhance_web` / `speech_separation_web` / `speech_super_resolution_web` 继续合法存在，工作台不替换它们的 API。
- 禁止跨场景引擎超市（ClearVoice 与 Demucs 同一目录页）。禁止服务端万能管道（不要 `POST /api/pipeline`）。人声隔离不要接到任何 speech_* 产品，也不要在该入口加 mdx 等其它 Demucs 袋。

## Environment

- 每个产品使用产品本地 `.venv`：`uv venv --python 3.12`。
- 安装：在产品目录执行 `uv pip install -e .`。
- 运行优先使用 `.venv/bin/python` 与 `.venv/bin/uvicorn`。
- 不要复用 explore 或仓库根的 `.venv`。

## Web Product Architecture

- FastAPI JSON API + 静态 HTML + 原生 JS。
- 主页面由 `frontend/index.html` 承载。
- 从产品目录启动：`.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860`。
