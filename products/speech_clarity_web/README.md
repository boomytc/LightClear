# Speech Clarity FastAPI WebUI

独立的 LightClear **语音清晰工作台**。场景是把带噪、叠人或带限的语音变清楚。增强、分离、超分是三件可独立调用的工具，由页面按需串联，不是固定流水线，也不是 ClearVoice 模型台。

默认模型：

- 增强 `MossFormer2_SE_48K`
- 分离 `MossFormer2_SS_16K`
- 超分 `MossFormer2_SR_48K`

人声隔离在 `vocal_isolate_web`，不要接到这里。三个单工具入口 `speech_enhance_web` / `speech_separation_web` / `speech_super_resolution_web` 仍然独立存在。

## 运行

在本产品目录内执行：

```bash
uv venv --python 3.12
uv pip install -e .
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860`。

## 接口

- `GET /api/health`：场景、三件工具的模型与 `available`（不加载权重）。
- `GET /api/tools`：与 health 中的 `tools` 相同。
- `GET /api/samples`：产品 `assets/` 下的三类示例，含 `suggested_tools`。
- `POST /api/tasks`：从示例或上传创建任务，拷入 `workspace/tasks/<task_id>/`。
- `GET /api/tasks`：扫描磁盘上的任务列表（刷新后仍可打开）。
- `POST /api/tasks/{task_id}/runs`：对一份 `input_ref` 跑 **一个** 工具。`input_ref` 为 `input` 或 `step:<run_id>:<output_id>`。
- `GET /api/tasks/{task_id}`：任务与步骤。
- `GET /api/tasks/{task_id}/runs/{run_id}`：按步骤重算分析。
- `GET /api/tasks/{task_id}/audio/{artifact_id}`：播放。
- `GET /api/tasks/{task_id}/download/{artifact_id}`：下载。

没有 `POST /api/pipeline`。缺某一工具权重时只对该工具 503。

客户端默认顺序：分离 → 增强 → 超分。勾选分离后再勾增强或超分时，会对分离出的每一路各跑一次。叠人样例默认只勾分离；带噪只勾增强；带限只勾超分。选中某一步产物会刷新该步分析，并可对这一路再跑另一件工具。

## 样例

| 文件 | 问题 | 默认勾选 |
| --- | --- | --- |
| `assets/noisy_input.wav` | 带噪 | 增强 |
| `assets/mixture_input.wav` | 叠人 | 分离 |
| `assets/bandlimited_input.wav` | 带限 | 超分 |

产物写入 `workspace/tasks/`。
