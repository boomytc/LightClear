# speech_separation_web Product Instructions

## Scope

- 本文件只适用于 `products/speech_separation_web`。
- 将本目录视为独立产品根目录；先 `cd products/speech_separation_web`。
- 默认模型 `MossFormer2_SS_16K`，任务 `speech_separation`。
- 运行时不得 import explore，不得读取仓库根或 explore 的 `third_party/`。

## Environment

- `uv venv --python 3.12`
- `uv pip install -e .`
- 运行：`.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860`

## Layout

- `backend/app.py`：FastAPI 入口。
- `backend/runtime.py`：ClearVoice 加载与两路分离。
- `third_party/clearvoice`：产品本地 vendored 运行时。
- `assets/input_ss.wav`：产品自有样例。
- `workspace/uploads/`、`workspace/outputs/`：运行产物。
- 中心模型目录：`/Users/boom/Model/SS/MossFormer2_SS_16K`

## Validation

- `.venv/bin/python -m py_compile backend/*.py`
- `.venv/bin/python -m tests.test_health`
