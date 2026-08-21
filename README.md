# LightClear

LightClear 是 ClearVoice 本地推理工作区，面向语音增强、语音分离、语音超分辨率和音视频目标说话人提取。

仓库根只做导航。ClearVoice 探索家族在 `explore/light_clearvoice`；稳定 WebUI 在 `products/`，各自独立安装和运行。

公共入口：

```python
from clearvoice import ClearVoice
```

## 目录

```text
LightClear/
├── assets/                         # explore 共享样例池
├── explore/light_clearvoice/       # 唯一上游家族，独立 Python 项目
├── products/
│   ├── speech_enhance_web/
│   ├── speech_separation_web/
│   └── speech_super_resolution_web/
├── AGENTS.md
├── Makefile
└── README.md
```

| 路径 | 说明 |
| --- | --- |
| `explore/light_clearvoice/` | ClearVoice 运行时、demo、模型文档。 |
| `assets/` | explore 共享样例；产品运行时不得读取。 |
| `products/speech_enhance_web/` | 语音增强 WebUI，默认 `MossFormer2_SE_48K`。 |
| `products/speech_separation_web/` | 语音分离 WebUI，默认 `MossFormer2_SS_16K`。 |
| `products/speech_super_resolution_web/` | 语音超分辨率 WebUI，默认 `MossFormer2_SR_48K`。 |

## 探索模块

```bash
cd explore/light_clearvoice
uv venv --python 3.12
uv pip install -e .
.venv/bin/python demo/demo_se.py
.venv/bin/python demo/demo_ss.py
.venv/bin/python demo/demo_sr.py
.venv/bin/python demo/demo_tse.py
```

模型权重使用中心目录 `/Users/boom/Model/{SE,SS,SR}/`。详见 `explore/light_clearvoice/README.md`。

## 产品

每个产品在自己的目录安装和启动：

```bash
cd products/speech_enhance_web
uv venv --python 3.12
uv pip install -e .
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860
```

分离与超分产品把目录换成 `speech_separation_web` 或 `speech_super_resolution_web`。不要从仓库根安装，也不要用 `products.<name>` 作为 uvicorn 目标。

## 验证

```bash
cd explore/light_clearvoice
.venv/bin/python -c "from clearvoice import ClearVoice; print(ClearVoice.__name__)"
.venv/bin/python -m tests.test_clearvoice_import

cd ../../products/speech_enhance_web
.venv/bin/python -m tests.test_health

cd ../..
python3 scripts/check_layout.py
```

完整推理前确认中心模型目录存在。缺权重是环境限制。
