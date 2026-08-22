# explore/

能力探索层。每个 `light_*` 是独立项目根，运行时互不 import。
稳定应用在仓库根 `products/`，不要把可交付 UI 做进这里。

## 家族（不要合并）

| 模块 | 上游家族 | 学什么 | 产品可抄形状 | 不要当 |
|------|----------|--------|--------------|--------|
| [`light_clearvoice/`](./light_clearvoice/) | ClearVoice | SE / SS / SR / TSE 加载与推理 | 三个单工具 `speech_*_web` 与 `speech_clarity_web` 工作台各自自持一份 ClearVoice | Demucs 运行时、把三个 WebUI 合成引擎超市 |
| [`light_demucs/`](./light_demucs/) | Demucs | 人声隔离、四轨、`htdemucs_ft` 质量档、六轨；`Separator` API | `vocal_isolate_web` 自持一份 Demucs，默认四轨，可选质量档与六轨 | ClearVoice 说话人分离、`speech_separation_web` |

新上游家族开新的 `light_*`，不要按任务拆 ClearVoice，也不要折进兄弟目录。

## 和产品怎么对应

| 产品 | 主要参考的 explore | 注意 |
|------|--------------------|------|
| `products/speech_clarity_web` | `light_clearvoice` 的 SE / SS / SR 用法 | 语音清晰场景工作台；工具可独立调用，组合在客户端。不得 import explore，不得合并三个单工具包 |
| `products/speech_enhance_web` | `light_clearvoice` | 单工具增强；产品自持 ClearVoice；不得 import explore |
| `products/speech_separation_web` | `light_clearvoice` | 单工具说话人分离，不是 Demucs 分轨 |
| `products/speech_super_resolution_web` | `light_clearvoice` | 单工具超分 |
| `products/vocal_isolate_web` | `light_demucs` 的 `htdemucs` / `htdemucs_ft` / `htdemucs_6s` | 人声隔离；默认四轨，可选质量档与六轨。不得 import explore |

## 从哪进

从仓库根目录：

```bash
cd explore/light_clearvoice
uv venv --python 3.12
uv pip install -e .
.venv/bin/python demo/demo_se.py

cd ../light_demucs
uv venv --python 3.12
uv pip install -e .
.venv/bin/python demo/demo_vocals.py
```
