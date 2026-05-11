# Speech Enhance FastAPI WebUI

前后端分离的 LightClear 语音增强 WebUI。当前默认使用 `MossFormer2_SE_48K`，后端使用 FastAPI 暴露 `/api/*`，前端是 `frontend/` 下的静态 HTML/CSS/JS/Tailwind 页面。

该目录按产品能力命名，不绑定单一模型名；后续可以在 `products/` 下继续增加语音分离和语音超分辨率产品入口。

## 运行

```bash
cd /Users/boom/workspace/LightClear
uv venv --python 3.12
uv pip install -e ".[web]"
.venv/bin/uvicorn products.speech_enhance_web.backend.app:app --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860`。

## 接口

- `GET /api/health`：模型目录、示例数量和默认输出目录。
- `GET /api/samples`：列出 `assets/clearvoice_samples/` 下的示例音频。
- `POST /api/enhance`：上传音频或选择示例音频并执行增强。
- `GET /api/jobs/{job_id}/audio/{original|enhanced}`：播放任务音频。
- `GET /api/jobs/{job_id}/download`：下载增强音频。

默认输出写入 `outputs/speech_enhance_web/enhanced/`，上传缓存写入 `outputs/speech_enhance_web/uploads/`。
