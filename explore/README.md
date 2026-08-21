# explore/

能力探索层。每个 `light_*` 是独立项目根，运行时互不 import。
稳定应用在仓库根 `products/`，不要把可交付 UI 做进这里。

## 家族（不要合并）

| 模块 | 上游家族 | 学什么 | 产品可抄形状 | 不要当 |
|------|----------|--------|--------------|--------|
| [`light_clearvoice/`](./light_clearvoice/) | ClearVoice | SE / SS / SR / TSE 加载与推理 | 三个 `speech_*_web` 产品自持一份 ClearVoice | 第二个上游、全模型管理台 |

新上游家族开新的 `light_*`，不要按任务拆 ClearVoice，也不要折进兄弟目录。

## 和产品怎么对应

| 产品 | 主要参考的 explore | 注意 |
|------|--------------------|------|
| `products/speech_enhance_web` | `light_clearvoice` | 产品自持 ClearVoice；不得 import explore |
| `products/speech_separation_web` | `light_clearvoice` | 同上 |
| `products/speech_super_resolution_web` | `light_clearvoice` | 同上 |

## 从哪进

```bash
cd explore/light_clearvoice
uv venv --python 3.12
uv pip install -e .
.venv/bin/python demo/demo_se.py
```
