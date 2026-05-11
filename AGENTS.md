# LightClear Project Instructions

## Scope

- 本文件适用于 `/Users/boom/workspace/LightClear` 整个仓库。
- 将本目录视为独立项目根目录，用于环境、安装、运行、demo、WebUI、模型资源和输出文件。
- 不要把父级工作区或兄弟项目当作本项目的 Python 根目录。
- 不要复用兄弟项目的 `.venv`、模型目录、临时文件或运行产物，除非用户明确要求跨项目处理。

## Python Environment

- 本项目环境统一用 `uv venv --python 3.12` 创建，默认生成项目内 `.venv/`。
- 使用 `.venv/bin/python` 执行本项目代码。
- 依赖安装到本项目环境内：
  - 基础运行：`uv pip install -e .`
  - FastAPI WebUI 运行：`uv pip install -e ".[web]"`
- `pyproject.toml` 允许 `>=3.10`，但本项目实际协作和验证默认按 Python 3.12 处理。
- macOS 或无 NVIDIA GPU 机器按 CPU 环境做链路验证；不要默认假设 CUDA 可用。
- 如果需要 CUDA，先确认目标机器、PyTorch wheel 和 `torch.cuda.is_available()`，再运行依赖 GPU 的实际推理。

## Execution Rules

- 运行命令前先进入项目根目录 `/Users/boom/workspace/LightClear`。
- 优先使用项目相对路径，例如 `demo/...`、`assets/...`、`models/...`、`outputs/...`、`third_party/...`。
- 不要在脚本里硬编码 `/Users/boom/...` 这类机器绝对路径。
- demo 和 WebUI 入口需要从项目根目录启动，确保 `assets/`、`models/`、`outputs/` 的相对路径一致。
- 运行产物默认写入 `outputs/`；这是可清理目录，不作为源码能力的一部分。
- 不要把模型权重、缓存、生成音频、临时目录或 WebUI 运行产物纳入版本控制。

## Project Layout

- `pyproject.toml`：项目依赖和打包配置；当前将 `third_party/clearvoice` 打包为 `clearvoice*`。
- `third_party/clearvoice/`：ClearVoice 的实际运行时、模型网络、数据处理、音视频工具和推理配置。
- `third_party/clearvoice/config/inference/`：各模型的推理配置；`checkpoint_dir` 与 `models/` 下目录约定必须同步。
- `assets/clearvoice_samples/`：本项目 demo 使用的示例音频、示例视频、目录输入和 `.scp` 输入。
- `models/se_models/`：语音增强和目标说话人提取模型目录。
- `models/ss_models/`：语音分离模型目录。
- `models/sr_models/`：语音超分辨率模型目录。
- `models/*/models.txt`：模型下载和本地目录索引，不等于本地权重一定已经完整存在。
- `demo/`：公开能力 demo，展示 ClearVoice 的真实加载、输入、推理和输出路径。
- `products/speech_enhance_web/`：语音增强 WebUI，默认面向 MossFormer2_SE_48K；后端提供 FastAPI `/api/*` 接口，前端使用 HTML/CSS/JS/Tailwind 静态资源。
- `products/speech_separation_web/`：语音分离 WebUI，默认面向 MossFormer2_SS_16K；输出两路说话人分离结果。
- `products/speech_super_resolution_web/`：语音超分辨率 WebUI，默认面向 MossFormer2_SR_48K。
- `outputs/`：demo / WebUI 默认输出目录，可按需清理。

## Model And Task Boundaries

- 当前公共入口是 `from clearvoice import ClearVoice`。
- 语音增强任务使用 `task="speech_enhancement"`，当前模型包括：
  - `MossFormer2_SE_48K`
  - `FRCRN_SE_16K`
  - `MossFormerGAN_SE_16K`
- 语音分离任务使用 `task="speech_separation"`，当前模型包括：
  - `MossFormer2_SS_16K`
- 语音超分辨率任务使用 `task="speech_super_resolution"`，当前模型包括：
  - `MossFormer2_SR_48K`
- 音视频目标说话人提取任务使用 `task="target_speaker_extraction"`，当前模型包括：
  - `AV_MossFormer2_TSE_16K`
- 如果新增或改名模型，至少同步检查：
  - `models/*/models.txt`
  - `third_party/clearvoice/config/inference/*.yaml`
  - `third_party/clearvoice/network_wrapper.py`
  - 对应 `demo/*.py`
- 不要因为本地某个模型目录暂时缺失就删除 demo 支持；只要模型记录在 `models.txt` 中，demo 可以保留对应公开用法。

## Demo Rules

- `demo/` 是模型能力、参数和真实调用方式的长期参考，不作为临时脚本目录。
- demo 脚本保持单文件独立实现，不封装任何函数，包括 `main()`。
- demo 按顶层顺序代码书写：import -> 全局关键参数 -> 本地 `third_party` 路径引导 -> 模型加载 -> 输入构造 -> 推理调用 -> 结果输出。
- 关键路径、模型名、任务名、输入文件、输出路径、采样率、设备或批处理参数等放在文件顶部的全局变量中，便于直接改参验证。
- demo 要展示 ClearVoice 的真实使用方式，不把核心加载、输入处理、推理调用或结果读取藏到辅助函数、包装类或跨文件工具层。
- 单个 demo 不要求覆盖全部能力；多个 demo 合起来应能充分体现文件输入、目录输入、`.scp` 输入、numpy/tensor 输入、语音增强、语音分离、语音超分辨率和目标说话人提取等用法。
- 新增 demo 时优先补齐缺失的真实公开路径，例如批量输入、目录输入、`.scp` 输入、numpy 到 numpy、不同模型任务、参数控制或模型特有能力。
- 不要在 demo 中增加 Hugging Face / ModelScope 下载流程、训练流程或微调流程，除非用户明确要求；默认只针对已放入 `models/` 的本地模型做加载推理。
- `products/` 下的 WebUI 入口可以使用函数、类和模块拆分；上述无函数封装规则只强制适用于 `demo/` 下作为主要能力证明的 `.py` 脚本。

## Product WebUI Rules

- FastAPI WebUI 当前产品入口：
  - `.venv/bin/uvicorn products.speech_enhance_web.backend.app:app --host 127.0.0.1 --port 7860`
  - `.venv/bin/uvicorn products.speech_separation_web.backend.app:app --host 127.0.0.1 --port 7860`
  - `.venv/bin/uvicorn products.speech_super_resolution_web.backend.app:app --host 127.0.0.1 --port 7860`
- FastAPI WebUI 依赖 `fastapi`、`uvicorn`、`python-multipart` 和 `matplotlib`，运行前使用 `uv pip install -e ".[web]"`。
- FastAPI WebUI 后端接口放在各产品的 `backend/`，前端静态资源放在各产品的 `frontend/`；不要把接口逻辑写进前端脚本。
- FastAPI WebUI 输出默认写入各产品自己的 `outputs/<product>/`，上传文件缓存写入该目录下的 `uploads/`。
- SE/SS/SR WebUI 按任务拆成独立 `products/` 入口；不要把某个入口扩展成全模型管理平台。

## Third Party Rules

- `third_party/clearvoice` 是 vendored 运行时，非必要不要直接修改上游模型实现。
- 优先在 `demo/`、`products/` 或项目边界层做路径和入口适配。
- 如果行为变更必须进入 `third_party/clearvoice`，保持改动最小，并同步检查所有受影响 demo。
- `network_wrapper.py` 根据 task/model 加载配置和网络类；修改任务名、模型名或配置路径时，要把模型索引和 demo 一起更新。
- 不要引入额外的二次封装层来替代 `ClearVoice` 公共入口，除非用户明确要求产品化 API。

## Verification

- 轻量语法校验：
  - `.venv/bin/python -m py_compile demo/*.py`
  - `.venv/bin/python -m py_compile products/*_web/backend/*.py`
- 基础导入校验：
  - `.venv/bin/python -c "from clearvoice import ClearVoice; print(ClearVoice)"`
- 常用 demo 验证：
  - `.venv/bin/python demo/demo_se.py assets/clearvoice_samples/input.wav`
  - `.venv/bin/python demo/demo_sr.py`
  - `.venv/bin/python demo/demo_ss.py`
  - `.venv/bin/python demo/demo_Numpy2Numpy.py`
- 运行完整推理前先确认对应模型权重目录存在，并与 `third_party/clearvoice/config/inference/*.yaml` 中的 `checkpoint_dir` 一致。
- 如果环境缺少模型权重、WebUI 依赖或音视频系统依赖，记录为环境限制，不要把静态校验误报为运行时能力已经通过。

## Maintenance

- 如果环境创建方式、模型目录、配置路径、demo 规范、WebUI 入口或输出目录发生变化，同步更新本文件。
- 如果修改 `pyproject.toml` 的包发现规则，确认 `clearvoice` 仍能从 `.venv/bin/python` 直接导入。
- 如果新增 README 或部署说明，内容应贴合当前脚本和真实入口，不要保留不存在的 CLI、服务端或云端能力。
- 清理时保留 `demo/`、`assets/`、`models/*/models.txt`、`third_party/clearvoice/config/` 和 `products/*_web/` 的公开入口；只清理明确的缓存、输出和临时文件。
