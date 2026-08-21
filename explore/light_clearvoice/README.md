# light_clearvoice

ClearVoice 探索模块。公共入口：

```python
from clearvoice import ClearVoice
```

覆盖语音增强、语音分离、语音超分辨率和音视频目标说话人提取。模型权重使用中心目录 `/Users/boom/Model/{SE,SS,SR}/`。

## 安装

```bash
cd explore/light_clearvoice
uv venv --python 3.12
uv pip install -e .
```

## Demo

从本模块目录运行：

```bash
.venv/bin/python demo/demo_se.py
.venv/bin/python demo/demo_ss.py
.venv/bin/python demo/demo_sr.py
.venv/bin/python demo/demo_tse.py
.venv/bin/python demo/demo_frcrn_se.py
.venv/bin/python demo/demo_mossformergan_se.py
.venv/bin/python demo/demo_Numpy2Numpy.py
.venv/bin/python demo/demo_tensor2tensor.py
```

样例音频/视频来自仓库根 `assets/`。输出写入本模块 `outputs/`。

## 模型

见 `docs/模型列表.md` 与 `docs/models/`。
