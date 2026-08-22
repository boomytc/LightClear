# vocal_isolate_web Product Instructions

## Scope

- 本文件只适用于 `products/vocal_isolate_web`。
- 将本目录视为独立产品根目录。在本产品目录内编辑、运行、测试、安装；从仓库根进入则 `cd products/vocal_isolate_web`。
- 任务 `vocal_isolation`。可选模型：`htdemucs`（默认四轨）、`htdemucs_ft`（同四轨质量档）、`htdemucs_6s`（六轨，含 guitar / piano）。
- 一次推理得到人声 / 伴奏；四轨模型附带 drums / bass / other，六轨再附 guitar / piano。
- 这不是说话人分离。不要接到 `speech_separation_web`，不要 import explore。不要加 mdx 等其它 Demucs 袋。

## Environment

- `uv venv --python 3.12`
- `uv pip install -e .`
- 运行：`.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860`

## Layout

- `backend/app.py`：FastAPI 入口。
- `backend/runtime.py`：Demucs `Separator` 按所选模型加载与一次分轨。
- `third_party/demucs`：产品本地 vendored 运行时，来自 `explore/light_demucs/third_party/demucs`。
- `assets/next_station_heaven.mp3`：产品自有带唱短样例。
- `workspace/uploads/`、`workspace/outputs/`：运行产物。
- 权重只读本产品 `models/<model>/`。所选模型缺失时 health 标记未就绪，推理 503。安装：

```bash
cd ../../explore/light_demucs
.venv/bin/python scripts/install_model.py htdemucs ../../products/vocal_isolate_web/models/htdemucs
.venv/bin/python scripts/install_model.py htdemucs_ft ../../products/vocal_isolate_web/models/htdemucs_ft
.venv/bin/python scripts/install_model.py htdemucs_6s ../../products/vocal_isolate_web/models/htdemucs_6s
```

不读 explore 的 `models/`。

## Validation

- `.venv/bin/python -m py_compile backend/*.py`
- `.venv/bin/python -m tests.test_health`
