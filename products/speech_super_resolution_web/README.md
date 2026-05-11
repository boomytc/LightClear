# Speech Super Resolution FastAPI WebUI

前后端分离的 LightClear 语音超分辨率 WebUI。当前默认使用 `MossFormer2_SR_48K`，后端使用 FastAPI 暴露 `/api/*`，前端是 `frontend/` 下的静态 HTML/CSS/JS/Tailwind 页面。

## 运行

```bash
cd /Users/boom/workspace/LightClear
uv venv --python 3.12
uv pip install -e ".[web]"
.venv/bin/uvicorn products.speech_super_resolution_web.backend.app:app --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860`。

## 接口

- `GET /api/health`：模型目录、示例数量和默认输出目录。
- `GET /api/samples`：列出 `assets/clearvoice_samples/` 下的示例音频。
- `POST /api/super-resolve`：上传音频或选择示例音频并执行语音超分辨率。
- `GET /api/jobs/{job_id}/audio/{original|super-resolved}`：播放任务音频。
- `GET /api/jobs/{job_id}/download`：下载超分辨率音频。

默认输出写入 `outputs/speech_super_resolution_web/super_resolved/`，上传缓存写入 `outputs/speech_super_resolution_web/uploads/`。
