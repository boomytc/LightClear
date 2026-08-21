# vocal_isolate_web Product Instructions

## Scope

- 本文件只适用于 `products/vocal_isolate_web`。
- 将本目录视为独立产品根目录。在本产品目录内编辑、运行、测试、安装；从仓库根进入则 `cd products/vocal_isolate_web`。
- 默认模型 `htdemucs`，任务 `vocal_isolation`。输出人声 / 伴奏，并附带 drums / bass / other。
- 这不是说话人分离。不要接到 `speech_separation_web`，不要 import explore。

## Environment

- `uv venv --python 3.12`
- `uv pip install -e .`
- 运行：`.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860`

## Layout

- `backend/app.py`：FastAPI 入口。
- `backend/runtime.py`：Demucs `Separator` 加载与一次分轨。
- `third_party/demucs`：产品本地 vendored 运行时，来自 `explore/light_demucs/third_party/demucs`。
- `assets/next_station_heaven.mp3`：产品自有带唱短样例。
- `workspace/uploads/`、`workspace/outputs/`：运行产物。
- 权重只读本产品 `models/htdemucs/`。缺失时 health 为未就绪，推理直接失败。安装：在 `explore/light_demucs` 运行 `scripts/install_htdemucs.py`（会拷一份到本产品）。不读 explore 运行时，不写 `~/.cache/huggingface`。

## Validation

- `.venv/bin/python -m py_compile backend/*.py`
- `.venv/bin/python -m tests.test_health`
