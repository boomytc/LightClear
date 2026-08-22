# Vocal Isolate FastAPI WebUI

独立的 LightClear 人声隔离 WebUI。默认模型 `htdemucs`，也可选 `htdemucs_ft`（同四轨质量档）和 `htdemucs_6s`（六轨）。一次推理得到人声、伴奏，以及对应分轨。不是说话人分离。

默认示例 `assets/next_station_heaven.mp3` 来自本机《下一站天后》混音的 30 秒片段，仅供本机演示。

## 运行

在本产品目录内执行：

```bash
uv venv --python 3.12
uv pip install -e .
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860`。

## 接口

- `GET /api/health`：默认模型、三档可选模型及各自是否就绪。`model_name` 与 `default_model` 都是默认档 `htdemucs`。
- `GET /api/samples`：列出产品 `assets/` 下的示例音频。
- `POST /api/isolate`：上传或选择示例，按 `model_name` 执行人声隔离。
- `GET /api/jobs/{job_id}/audio/{original|vocals|accompaniment|drums|bass|other|guitar|piano}`：播放。六轨模型才有 guitar / piano。
- `GET /api/jobs/{job_id}/download/{vocals|accompaniment|drums|bass|other|guitar|piano}`：下载。

默认输出写入 `workspace/outputs/`，上传缓存写入 `workspace/uploads/`。权重只读本产品 `models/<model>/`。缺失时：

```bash
cd ../../explore/light_demucs
.venv/bin/python scripts/install_model.py htdemucs ../../products/vocal_isolate_web/models/htdemucs
.venv/bin/python scripts/install_model.py htdemucs_ft ../../products/vocal_isolate_web/models/htdemucs_ft
.venv/bin/python scripts/install_model.py htdemucs_6s ../../products/vocal_isolate_web/models/htdemucs_6s
```
