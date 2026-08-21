# Vocal Isolate FastAPI WebUI

独立的 LightClear 人声隔离 WebUI。默认模型 `htdemucs`。一次推理得到人声、伴奏，以及 drums / bass / other。不是说话人分离。

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

- `GET /api/health`：模型、示例数量和默认输出目录。
- `GET /api/samples`：列出产品 `assets/` 下的示例音频。
- `POST /api/isolate`：上传或选择示例并执行人声隔离。
- `GET /api/jobs/{job_id}/audio/{original|vocals|accompaniment|drums|bass|other}`：播放。
- `GET /api/jobs/{job_id}/download/{vocals|accompaniment|drums|bass|other}`：下载。

默认输出写入 `workspace/outputs/`，上传缓存写入 `workspace/uploads/`。权重只读本产品 `models/htdemucs/`。缺失时在 `explore/light_demucs` 运行 `scripts/install_htdemucs.py`。
