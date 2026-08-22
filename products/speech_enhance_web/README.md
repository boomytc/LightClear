# Speech Enhance FastAPI WebUI

独立的 LightClear 语音增强 WebUI。默认模型 `MossFormer2_SE_48K`。后端 FastAPI 暴露 `/api/*`，前端是 `frontend/` 下的静态 HTML/CSS/JS。

## 运行

在本产品目录内执行：

```bash
uv venv --python 3.12
uv pip install -e .
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860`。

## 接口

- `GET /api/health`：模型目录、示例数量和默认输出目录。
- `GET /api/samples`：列出产品 `assets/` 下的示例音频。
- `POST /api/enhance`：上传音频或选择示例音频并执行增强。
- `GET /api/jobs/{job_id}/audio/{original|enhanced}`：播放任务音频。
- `GET /api/jobs/{job_id}/download`：下载增强音频。

默认输出写入 `workspace/outputs/`，上传缓存写入 `workspace/uploads/`。

语音多工具工作台见 `products/speech_clarity_web`。本入口仍是单工具增强。
