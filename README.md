# LightClear

LightClear 是音频变清晰工作区。把带噪、混合、带限或叠乐的声音拆成听得清的目标层。

仓库根只做导航。探索家族各自独立安装；稳定 WebUI 在 `products/`，运行时不读取 explore。

| 家族 | 公共入口 | 做什么 |
| --- | --- | --- |
| `explore/light_clearvoice` | `from clearvoice import ClearVoice` | 语音增强、说话人分离、超分、目标说话人提取 |
| `explore/light_demucs` | `from demucs.api import Separator` | 人声隔离、四轨 / 六轨分轨 |

## 目录

```text
LightClear/
├── assets/                         # explore 共享样例池
├── explore/
│   ├── light_clearvoice/           # ClearVoice 家族
│   └── light_demucs/               # Demucs 家族
├── products/
│   ├── speech_clarity_web/
│   ├── speech_enhance_web/
│   ├── speech_separation_web/
│   ├── speech_super_resolution_web/
│   └── vocal_isolate_web/
├── AGENTS.md
├── Makefile
└── README.md
```

| 路径 | 说明 |
| --- | --- |
| `explore/light_clearvoice/` | ClearVoice 运行时、demo、模型文档。 |
| `explore/light_demucs/` | Demucs 运行时、demo、模型文档。vendor 在模块 `third_party/demucs`。 |
| `assets/` | explore 共享样例；产品运行时不得读取。 |
| `products/speech_clarity_web/` | 语音清晰工作台：增强 / 分离 / 超分可独立调用，客户端按需串联。 |
| `products/speech_enhance_web/` | 语音增强单工具 WebUI，默认 `MossFormer2_SE_48K`。 |
| `products/speech_separation_web/` | 说话人分离单工具 WebUI，默认 `MossFormer2_SS_16K`。 |
| `products/speech_super_resolution_web/` | 语音超分辨率单工具 WebUI，默认 `MossFormer2_SR_48K`。 |
| `products/vocal_isolate_web/` | 人声隔离 WebUI，默认 `htdemucs`，可选 `htdemucs_ft` 与 `htdemucs_6s`。 |

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

ClearVoice 权重使用 `/Users/boom/Model/{SE,SS,SR}/`。Demucs 权重放在各自模块/产品的 `models/<model>/`；产品 `vocal_isolate_web` 可装 `htdemucs`、`htdemucs_ft`、`htdemucs_6s`，默认 `htdemucs`。详见各模块 README。

## 产品

从仓库根目录进入产品目录后安装和启动：

```bash
cd products/speech_enhance_web
uv venv --python 3.12
uv pip install -e .
.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 7860
```

其他产品把目录换成 `speech_clarity_web`、`speech_separation_web`、`speech_super_resolution_web` 或 `vocal_isolate_web`。不要从仓库根安装，也不要用 `products.<name>` 作为 uvicorn 目标。人声隔离不要接到任何 speech_* 产品。三个单工具 WebUI 仍然独立存在，不是把它们合并进工作台。

## 验证

```bash
cd explore/light_clearvoice
.venv/bin/python -c "from clearvoice import ClearVoice; print(ClearVoice.__name__)"
.venv/bin/python -m tests.test_clearvoice_import

cd ../light_demucs
.venv/bin/python -c "from demucs.api import Separator; print(Separator.__name__)"
.venv/bin/python -m tests.test_demucs_import

cd ../../products/speech_clarity_web
.venv/bin/python -m tests.test_health
.venv/bin/python -m tests.test_task_contract

cd ../speech_enhance_web
.venv/bin/python -m tests.test_health

cd ../vocal_isolate_web
.venv/bin/python -m tests.test_health

cd ../..
python3 scripts/check_layout.py
```

完整推理前确认对应权重可用。缺权重是环境限制。
