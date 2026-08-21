# LightClear

LightClear 是音频变清晰工作区。把带噪、混合、带限或叠乐的声音拆成听得清的目标层。

仓库根只做导航。探索家族各自独立安装；稳定 WebUI 在 `products/`，运行时不读取 explore。

| 家族 | 公共入口 | 做什么 |
| --- | --- | --- |
| `explore/light_clearvoice` | `from clearvoice import ClearVoice` | 语音增强、说话人分离、超分、目标说话人提取 |
| `explore/light_demucs` | `from demucs.api import Separator` | 人声隔离、音乐分轨 |

## 目录

```text
LightClear/
├── assets/                         # explore 共享样例池
├── explore/
│   ├── light_clearvoice/           # ClearVoice 家族
│   └── light_demucs/               # Demucs 家族
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
| `explore/light_demucs/` | Demucs 运行时、demo、模型文档。vendor 在模块 `third_party/demucs`。 |
| `assets/` | explore 共享样例；产品运行时不得读取。 |
| `products/speech_enhance_web/` | 语音增强 WebUI，默认 `MossFormer2_SE_48K`。 |
| `products/speech_separation_web/` | 说话人分离 WebUI，默认 `MossFormer2_SS_16K`。 |
| `products/speech_super_resolution_web/` | 语音超分辨率 WebUI，默认 `MossFormer2_SR_48K`。 |

## 探索模块

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

ClearVoice 权重使用 `/Users/boom/Model/{SE,SS,SR}/`。Demucs 默认从 Hugging Face 加载，可选 `/Users/boom/Model/MSS/`。详见各模块 README。

## 产品

从仓库根目录进入产品目录后安装和启动：

```bash
cd products/speech_enhance_web
uv venv --python 3.12
uv pip install -e .
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860
```

分离与超分产品把目录换成 `speech_separation_web` 或 `speech_super_resolution_web`。不要从仓库根安装，也不要用 `products.<name>` 作为 uvicorn 目标。Demucs 尚未产品化，不要接到 `speech_separation_web`。

## 验证

```bash
cd explore/light_clearvoice
.venv/bin/python -c "from clearvoice import ClearVoice; print(ClearVoice.__name__)"
.venv/bin/python -m tests.test_clearvoice_import

cd ../light_demucs
.venv/bin/python -c "from demucs.api import Separator; print(Separator.__name__)"
.venv/bin/python -m tests.test_demucs_import

cd ../../products/speech_enhance_web
.venv/bin/python -m tests.test_health

cd ../..
python3 scripts/check_layout.py
```

完整推理前确认对应权重可用。缺权重是环境限制。
