# LightClear

LightClear 是一个独立的 ClearVoice 本地推理项目，面向语音增强、语音分离、语音超分辨率和音视频目标说话人提取。项目公共入口是：

```python
from clearvoice import ClearVoice
```

本仓库自带 vendored `third_party/clearvoice` 运行时、示例素材、模型目录索引、公开 demo，以及按任务拆分的 FastAPI WebUI 产品入口。

## 项目结构

| 路径 | 说明 |
| --- | --- |
| `pyproject.toml` | 项目依赖和打包配置；通过 `third_party/` 发现并安装 `clearvoice*` 包。 |
| `third_party/clearvoice/` | ClearVoice 实际运行时、模型网络、数据处理、音视频工具和推理配置。 |
| `third_party/clearvoice/config/inference/` | 各模型推理配置；其中 `checkpoint_dir` 默认指向 `/Users/boom/Model/{SE,SS,SR}/` 下的中心模型目录。 |
| `models/se_models/` | 语音增强和目标说话人提取模型下载索引。 |
| `models/ss_models/` | 语音分离模型下载索引。 |
| `models/sr_models/` | 语音超分辨率模型下载索引。 |
| `assets/clearvoice_samples/` | demo 和 WebUI 使用的示例音频、视频、目录输入和 `.scp` 输入。 |
| `demo/` | 公开能力 demo，展示真实模型加载、输入构造、推理调用和输出路径。 |
| `products/speech_enhance_web/` | 语音增强 WebUI，默认模型 `MossFormer2_SE_48K`。 |
| `products/speech_separation_web/` | 语音分离 WebUI，默认模型 `MossFormer2_SS_16K`。 |
| `products/speech_super_resolution_web/` | 语音超分辨率 WebUI，默认模型 `MossFormer2_SR_48K`。 |
| `outputs/` | demo 和 WebUI 默认输出目录，可清理，不作为源码能力的一部分。 |

## 环境安装

建议在项目根目录创建本项目自己的 Python 3.12 环境：

```bash
cd /Users/boom/workspace/LightClear
uv venv --python 3.12
uv pip install -e .
```

如需运行 WebUI，安装 Web 额外依赖：

```bash
uv pip install -e ".[web]"
```

说明：

- 运行本项目代码时优先使用 `.venv/bin/python`。
- macOS 或无 NVIDIA GPU 机器默认按 CPU 链路验证。
- 非 `.wav` 音频格式和视频处理通常需要系统已安装 FFmpeg。

## 模型目录

`models/*/models.txt` 是模型下载和中心目录索引，不存放真实权重，也不等于权重一定已经完整存在。完整推理前应先确认对应权重目录存在，并和 `third_party/clearvoice/config/inference/*.yaml` 中的 `checkpoint_dir` 一致。

当前任务和模型边界：

| 任务 | `task` | 模型 | 中心目录约定 |
| --- | --- | --- | --- |
| 语音增强 | `speech_enhancement` | `MossFormer2_SE_48K` | `/Users/boom/Model/SE/MossFormer2_SE_48K` |
| 语音增强 | `speech_enhancement` | `FRCRN_SE_16K` | `/Users/boom/Model/SE/FRCRN_SE_16K` |
| 语音增强 | `speech_enhancement` | `MossFormerGAN_SE_16K` | `/Users/boom/Model/SE/MossFormerGAN_SE_16K` |
| 语音分离 | `speech_separation` | `MossFormer2_SS_16K` | `/Users/boom/Model/SS/MossFormer2_SS_16K` |
| 语音超分辨率 | `speech_super_resolution` | `MossFormer2_SR_48K` | `/Users/boom/Model/SR/MossFormer2_SR_48K` |
| 音视频目标说话人提取 | `target_speaker_extraction` | `AV_MossFormer2_TSE_16K` | `/Users/boom/Model/SE/AV_MossFormer2_TSE_16K` |

## Demo

demo 需要从项目根目录运行。常用入口：

```bash
.venv/bin/python demo/demo_se.py assets/clearvoice_samples/input.wav
.venv/bin/python demo/demo_sr.py
.venv/bin/python demo/demo_ss.py
.venv/bin/python demo/demo_Numpy2Numpy.py
.venv/bin/python demo/demo_tse.py
```

更多模型专项 demo：

```bash
.venv/bin/python demo/demo_frcrn_se.py
.venv/bin/python demo/demo_mossformergan_se.py
.venv/bin/python demo/demo_tensor2tensor.py
```

demo 覆盖的主要输入形态包括：

- 单文件音频输入。
- 目录批量输入。
- `.scp` 列表输入。
- numpy / torch tensor 到 tensor 输出。
- 视频目录和视频 `.scp` 输入的目标说话人提取。

推理结果默认写入 `outputs/`，或由 demo 中的顶部路径变量控制。

## WebUI

三个 WebUI 是独立产品入口，均从项目根目录启动。默认都使用 `7860` 端口，同一时间运行多个 WebUI 时需要改成不同端口。

语音增强：

```bash
.venv/bin/uvicorn products.speech_enhance_web.backend.app:app --host 127.0.0.1 --port 7860
```

语音分离：

```bash
.venv/bin/uvicorn products.speech_separation_web.backend.app:app --host 127.0.0.1 --port 7860
```

语音超分辨率：

```bash
.venv/bin/uvicorn products.speech_super_resolution_web.backend.app:app --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860`。各产品接口和输出目录见对应产品 README：

- `products/speech_enhance_web/README.md`
- `products/speech_separation_web/README.md`
- `products/speech_super_resolution_web/README.md`

## 基础验证

轻量语法校验：

```bash
.venv/bin/python -m py_compile demo/*.py
.venv/bin/python -m py_compile products/*_web/backend/*.py
```

基础导入校验：

```bash
.venv/bin/python -c "from clearvoice import ClearVoice; print(ClearVoice)"
```

完整推理验证前，请先确认目标模型权重已经放在 `/Users/boom/Model/{SE,SS,SR}/` 对应目录下。缺少权重、WebUI 依赖或 FFmpeg 时，应记录为环境限制，不要把静态校验等同于真实推理已通过。

## 清理

清理 Python 缓存和常见运行产物：

```bash
make clean
```

`outputs/` 是 demo 和 WebUI 的默认输出目录；如需释放空间，可以按需清理其中生成的文件。`outputs/`、`.venv/`、`*.egg-info/`、`uv.lock`、项目内模型权重目录和系统缓存文件不会作为源码能力纳入版本控制。
