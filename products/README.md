# products/

稳定产品层。每个子目录是独立产品根，运行时不 import explore、不读取仓库根。

| 产品 | 任务 | 启动 |
| --- | --- | --- |
| `speech_enhance_web` | speech_enhancement | `cd products/speech_enhance_web && .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860` |
| `speech_separation_web` | speech_separation | `cd products/speech_separation_web && .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860` |
| `speech_super_resolution_web` | speech_super_resolution | `cd products/speech_super_resolution_web && .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860` |

安装：在对应产品目录执行 `uv venv --python 3.12` 与 `uv pip install -e .`。
