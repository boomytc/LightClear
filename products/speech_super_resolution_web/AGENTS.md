# speech_super_resolution_web Product Instructions

## Scope

- 本文件只适用于 `products/speech_super_resolution_web`。
- 将本目录视为独立产品根目录；先 `cd products/speech_super_resolution_web`。
- 默认模型 `MossFormer2_SR_48K`，任务 `speech_super_resolution`。
- 运行时不得 import explore，不得读取仓库根或 explore 的 `third_party/`。

## Environment

- `uv venv --python 3.12`
- `uv pip install -e .`
- 运行：`.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860`

## Layout

- `backend/app.py`：FastAPI 入口。
- `backend/runtime.py`：ClearVoice 加载与超分辨率推理。
- `third_party/clearvoice`：产品本地 vendored 运行时。
- `assets/input_sr.wav`：产品自有样例。
- `workspace/uploads/`、`workspace/outputs/`：运行产物。
- 中心模型目录：`/Users/boom/Model/SR/MossFormer2_SR_48K`

## Validation

- `.venv/bin/python -m py_compile backend/*.py`
- `.venv/bin/python -m tests.test_health`
