# Speech Separation FastAPI WebUI

前后端分离的 LightClear 语音分离 WebUI。当前默认使用 `MossFormer2_SS_16K`，后端使用 FastAPI 暴露 `/api/*`，前端是 `frontend/` 下的静态 HTML/CSS/JS/Tailwind 页面。

## 运行

```bash
cd /Users/boom/workspace/LightClear
uv venv --python 3.12
uv pip install -e ".[web]"
.venv/bin/uvicorn products.speech_separation_web.backend.app:app --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860`。

## 接口

- `GET /api/health`：模型目录、示例数量和默认输出目录。
- `GET /api/samples`：列出 `assets/clearvoice_samples/` 下的示例音频。
- `POST /api/separate`：上传混合音频或选择示例音频并执行两路说话人分离。
- `GET /api/jobs/{job_id}/audio/{original|speaker-1|speaker-2}`：播放任务音频。
- `GET /api/jobs/{job_id}/download/{speaker-1|speaker-2}`：下载分离音频。

默认输出写入 `outputs/speech_separation_web/separated/`，上传缓存写入 `outputs/speech_separation_web/uploads/`。
