# products/

稳定产品层。每个子目录是独立产品根，运行时不 import explore、不读取仓库根。

从仓库根目录进入产品后再安装、启动：

| 产品 | 场景 / 任务 | 启动 |
| --- | --- | --- |
| `speech_clarity_web` | 语音清晰工作台（增强 / 分离 / 超分，客户端编排） | `cd products/speech_clarity_web && .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860` |
| `speech_enhance_web` | 语音增强单工具（`speech_enhancement`） | `cd products/speech_enhance_web && .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860` |
| `speech_separation_web` | 说话人分离单工具（`speech_separation`） | `cd products/speech_separation_web && .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860` |
| `speech_super_resolution_web` | 语音超分辨率单工具（`speech_super_resolution`） | `cd products/speech_super_resolution_web && .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860` |
| `vocal_isolate_web` | 人声隔离（`vocal_isolation`，默认 `htdemucs`，可选 `htdemucs_ft` / `htdemucs_6s`） | `cd products/vocal_isolate_web && .venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860` |

安装：进入对应产品目录后执行 `uv venv --python 3.12` 与 `uv pip install -e .`。产品 README 里的命令默认已经在该产品目录内。人声隔离不要接到任何 speech_* 产品。三个单工具入口与工作台并存，不要合并成模型台。
