# speech_clarity_web Product Instructions

## Scope

- 本文件只适用于 `products/speech_clarity_web`。
- 将本目录视为独立产品根目录。在本产品目录内编辑、运行、测试、安装；从仓库根进入则 `cd products/speech_clarity_web`。
- 场景是 **语音变清晰**。三件工具：`enhance`、`separate`、`super_resolve`。工具可独立调用，场景组合只在客户端。
- 运行时不得 import explore，不得读取仓库根或任何兄弟产品的 `third_party/`、`.venv`。不要接到 `vocal_isolate_web`，不要跑 TSE。

## Product Constitution

1. 场景产品，不是模型目录。不摊 FRCRN / MossFormerGAN / TSE / Demucs。
2. 一任务一工具，可独立调用。只要去噪就只调 `enhance`。
3. 组合在客户端。服务端每次 run 只吃一份输入音频。禁止服务端万能管道，禁止 `POST /api/pipeline`。
4. 缺件只禁用。权重不齐的工具 `available=false`，对该工具的 run 返回 503，不整站锁死。
5. 契约优先。工具名、产物 id、`input_ref`、task/run JSON 是 SSOT。
6. 个人本机。单操作者，无鉴权。
7. 产物在 `workspace/tasks/<task_id>/`。
8. 三兄弟 WebUI 与 explore demo 只证明怎么调 ClearVoice。本产品自写 runtime / 任务 / UI。

## Environment

- `uv venv --python 3.12`
- `uv pip install -e .`
- 运行：`.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860`

## Layout

- `backend/app.py`：FastAPI 入口。
- `backend/runtime.py`：三工具 ClearVoice 懒加载与单次推理。
- `backend/tasks.py`：磁盘 task 与产物。
- `backend/analysis.py`：按工具分析。
- `third_party/clearvoice`：产品本地 vendored 运行时，来自 `explore/light_clearvoice/third_party/clearvoice`。
- `assets/`：产品自有样例（带噪 / 叠人 / 带限）。
- `workspace/tasks/`：任务输入与逐步 wav。
- 中心模型目录：
  - 增强 `/Users/boom/Model/SE/MossFormer2_SE_48K`
  - 分离 `/Users/boom/Model/SS/MossFormer2_SS_16K`
  - 超分 `/Users/boom/Model/SR/MossFormer2_SR_48K`

三模型懒加载后常驻，v1 不卸载。全局一把推理锁。本机按 CPU 验证。

客户端默认串联顺序：分离 → 增强 → 超分（未勾选的跳过）。切换样例时预勾该样例的 `suggested_tools`，禁止默认三件全开。

## Validation

- `.venv/bin/python -m py_compile backend/*.py`
- `.venv/bin/python -m tests.test_health`
- `.venv/bin/python -m tests.test_task_contract`
