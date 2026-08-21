# ClearVoice

## 👉🏻[HuggingFace Space 演示](https://huggingface.co/spaces/alibabasglab/ClearVoice)👈🏻 |  👉🏻[ModelScope Space 演示](https://modelscope.cn/studios/iic/ClearerVoice-Studio)👈🏻 

## 目录

- [1. 简介](#1-简介)
- [2. 使用方法](#2-使用方法)

## 1. 简介

ClearVoice 为 `语音增强`、`语音分离`、`语音超分辨率` 和 `音视频目标说话人提取` 提供了一个统一的推理平台。它旨在简化预训练模型在您的语音处理任务中的应用或集成到您的项目中。目前，我们提供以下预训练模型：

| 任务 (采样率) | 模型 (HuggingFace 链接)|
|-------|--------------------------|
|语音增强 (16kHz & 48kHz)| `MossFormer2_SE_48K` ([HuggingFace](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)), `FRCRN_SE_16K` ([HuggingFace](https://huggingface.co/alibabasglab/FRCRN_SE_16K)), `MossFormerGAN_SE_16K` ([HuggingFace](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K)) | 
|语音分离 (16kHz)|`MossFormer2_SS_16K` ([HuggingFace](https://huggingface.co/alibabasglab/MossFormer2_SS_16K))|
|语音超分辨率 (48kHz)|`MossFormer2_SR_48K`([HuggingFace](https://huggingface.co/alibabasglab/MossFormer2_SR_48K))|
|音视频目标说话人提取 (16kHz)|`AV_MossFormer2_TSE_16K` ([HuggingFace](https://huggingface.co/alibabasglab/AV_MossFormer2_TSE_16K))|

您无需手动下载预训练模型——它们会在推理过程中自动从 HuggingFace 获取。如果模型未能成功下载到 `./clearvoice/checkpoints`，您可以从 [ModelScope](https://modelscope.cn/models/iic/ClearerVoice-Studio/summary) 手动下载。

## 2. 使用方法

### 通过 PyPI 安装

1. **通过 PyPI 安装 ClearVoice：**
    ``` sh
    pip install clearvoice
    ```

2. **在您的 Python 代码中：**
    ``` python
    from clearvoice import ClearVoice
    ```
### 安装 FFmpeg (可选)

Clearvoice 依赖 FFmpeg 进行音频格式转换。如果您只处理 `.wav` 文件，则不需要 FFmpeg。对于所有其他格式，请按照以下说明安装 FFmpeg。

1. 在 **Ubuntu/Debian** 上：
   ``` sh
   sudo apt update
   sudo apt install ffmpeg
   ```
2. 在 **macOS** 上 (使用 Homebrew)：
   ``` sh
   brew install ffmpeg
   ```
3. 在 **Windows** 上：
    从 https://ffmpeg.org/download.html 下载静态构建版本。
    解压并将 bin 文件夹（包含 ffprobe.exe）添加到系统路径 (PATH) 中。
   
### 从 GitHub 安装

1. **克隆 GitHub 仓库并安装依赖：**

    ``` sh
    git clone https://github.com/modelscope/ClearerVoice-Studio.git
    cd ClearerVoice-Studio/clearvoice
    pip install --editable .
    ```
2. **在您的 Python 代码中：**
    ``` python
    from clearvoice import ClearVoice
    ```

3. **演示脚本**

    ``` sh
    cd ClearerVoice-Studio/clearvoice
    python demo.py
    ```

    或者 

    ``` sh
    cd ClearerVoice-Studio/clearvoice
    python demo_with_more_comments.py
    ```

    或者
    ``` sh
    cd ClearerVoice-Studio/clearvoice
    python demo_Numpy2Numpy.py
    ```
- 您可以通过在 `demo.py`、`demo_with_more_comments.py` 和 `demo_Numpy2Numpy.py` 中将相应案例设置为 True 来激活。
- 在 `demo_Numpy2Numpy.py` 中，我们为 ClearVoice 添加了一个新接口，支持从 numpy 输入到 numpy 输出，而不是文件 I/O。
- 支持的音频格式：.flac .wav
- 支持的视频格式：.avi .mp4 .mov .webm

### Python 脚本示例

使用 `MossFormer2_SE_48K` 模型进行全带 (48kHz) 语音增强任务：

```python
from clearvoice import ClearVoice

myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

# 处理单个音频文件
output_wav = myClearVoice(input_path='samples/input.wav', online_write=False)
myClearVoice.write(output_wav, output_path='samples/output_MossFormer2_SE_48K.wav')

# 处理音频目录
myClearVoice(input_path='samples/path_to_input_wavs', online_write=True, output_path='samples/path_to_output_wavs')

# 处理音频列表文件 (.scp)
myClearVoice(input_path='samples/scp/audio_samples.scp', online_write=True, output_path='samples/path_to_output_wavs_scp')
```

参数说明：
- `task`: 选择三个任务之一：`speech_enhancement` (语音增强)、`speech_separation` (语音分离) 和 `target_speaker_extraction` (目标说话人提取)
- `model_names`: 模型名称列表，为任务选择一个或多个模型
- `input_path`: 输入音频/视频文件路径、输入音频/视频目录或列表文件 (.scp)
- `online_write`: 设置为 `True` 以在处理过程中直接将增强/分离后的音频/视频保存到本地文件，否则返回增强/分离后的音频。（处理单个音频文件时，`speech_enhancement` 和 `speech_separation` 仅支持 `False`）
- `output_path`: 保存增强/分离后的音频/视频文件的文件或目录路径

这里给出了一个较详细的中文使用教程：https://stable-learn.com/zh/clearvoice-studio-tutorial 

## 3. 模型性能

**语音增强模型：**

我们在流行的基准测试集上评估了发布的语音增强模型：[VoiceBank+DEMAND](https://paperswithcode.com/dataset/demand) 测试集 (16kHz & 48kHz) 和 [DNS-Challenge-2020](https://paperswithcode.com/dataset/deep-noise-suppression-2020) (Interspeech) 测试集 (无混响, 16kHz)。与大多数为每个测试集量身定制模型的论文不同，我们在这里的评估在两个测试集上使用了统一的模型。评估指标由 [SpeechScore](https://github.com/modelscope/ClearerVoice-Studio/tree/main/speechscore) 生成。

**VoiceBank+DEMAND 测试集 (在 16kHz 上测试)**
|模型               |PESQ    |NB_PESQ |CBAK    |COVL    |CSIG    |STOI    |SISDR    |SNR      |SRMR    |SSNR    |P808_MOS|SIG     |BAK     |OVRL    |ISR      |SAR      |SDR      |FWSEGSNR |LLR     |LSD     |MCD|
|-----               |---     |------- |----    |----    |----    |----    |-----    |---      |----    |----    |------  |---     |---     |----    |---      |---      |---      |-------- |---     |---     |---|
|Noisy               |1.97    | 3.32   |2.79    |2.70    |3.32    |0.92    |8.44     |9.35     |7.81    |6.13    |3.05    |3.37    |3.32    |2.79    |28.11    |8.53     |8.44     |14.77    |0.78    |1.40    |4.15|
|FRCRN_SE_16K        |3.23    | 3.86   |3.47    |**3.83**|4.29    |0.95    |19.22    |19.16    |9.21    |7.60    |**3.59**|3.46    |**4.11**|3.20    |12.66    |21.16    |11.71    |**20.76**|0.37    |0.98    |**0.56**|
|MossFormerGAN_SE_16K|**3.47**|**3.96**|**3.50**|3.73    |**4.40**|**0.96**|**19.45**|**19.36**|9.07    |**9.09**|3.57    |**3.50**|4.09    |**3.23**|25.98    |21.18    |**19.42**|20.20    |**0.34**|**0.79**|0.70|
|MossFormer2_SE_48K  |3.16    | 3.77   |3.32    |3.58    |4.14    |0.95    |19.38    |19.22    |**9.61**|6.86    |3.53    |**3.50**|4.07    |3.22    |**12.05**|**21.84**|11.47    |16.69    |0.57    |1.72    |0.62|

**DNS-Challenge-2020 测试集 (在 16kHz 上测试)**
|模型               |PESQ    |NB_PESQ |CBAK    |COVL    |CSIG    |STOI    |SISDR    |SNR      |SRMR    |SSNR    |P808_MOS|SIG     |BAK     |OVRL    |ISR      |SAR      |SDR      |FWSEGSNR |LLR     |LSD     |MCD|
|-----               |---     |------- |----    |----    |----    |----    |-----    |---      |----    |----    |------  |---     |---     |----    |---      |---      |---      |-------- |---     |---     |---|
|Noisy               |1.58    | 2.16   |2.66    |2.06    |2.72    |0.91    |9.07     |9.95     |6.13    |9.35    |3.15    |3.39    |2.61    |2.48    |34.57    |9.09     |9.06     |15.87    |1.07    |1.88    |6.42|
|FRCRN_SE_16K        |3.24    | 3.66   |3.76    |3.63    |4.31    |**0.98**|19.99    |19.89    |8.77    |7.60    |4.03    |3.58    |4.15    |3.33    |**8.90** |20.14    |7.93     |**22.59**|0.50    |1.69    |0.97|
|MossFormerGAN_SE_16K|**3.57**|**3.88**|**3.93**|**3.92**|**4.56**|**0.98**|**20.60**|**20.44**|8.68    |**14.03**|**4.05**|**3.58**|**4.18**|**3.36**|8.88    |**20.81**|**7.98** |21.62    |**0.45**|**1.65**|**0.89**|
|MossFormer2_SE_48K  |2.94    | 3.45   |3.36    |2.94    |3.47    |0.97    |17.75    |17.65    |**9.26**|11.86    |3.92   |3.51    |4.13    |3.26    |8.55     |18.40    |7.48     |16.10    |0.98    |3.02    |1.15|

**VoiceBank+DEMAND 测试集 (在 48kHz 上测试)** (我们包含了使用 SpeechScore 对其他开源模型的评估)
|模型               |PESQ    |NB_PESQ |CBAK    |COVL    |CSIG    |STOI    |SISDR    |SNR      |SRMR    |SSNR    |P808_MOS|SIG     |BAK     |OVRL    |ISR      |SAR      |SDR      |FWSEGSNR |LLR     |LSD     |MCD|
|-----               |---     |------- |----    |----    |----    |----    |-----    |---      |----    |----    |------  |---     |---     |----    |---      |---      |---      |-------- |---     |---     |---|
|Noisy               |1.97    | 2.87   |2.79    |2.70    |3.32    |0.92    |8.39     |9.30     |7.81    |6.13    |3.07    |3.35    |3.12    |2.69    |33.75    |8.42     |8.39     |13.98    |0.75    |1.45    |5.41|
|MossFormer2_SE_48K  |**3.15**|**3.77**|**3.33**|**3.64**|**4.23**|**0.95**|**19.36**|**19.22**|9.61    |7.03    |**3.53**|  3.41  |**4.10**|**3.15**|**4.08**|**21.23** |4.06     |14.45    |NA      |1.86    |**0.53**|
|Resemble_enhance    |2.84    | 3.58   |3.14    |NA      |NA      |0.94    |12.42    |12.79    |9.08    |7.07    |**3.53**|**3.42**|  3.99  |3.12    |13.62    |12.66    |10.31    |14.56    |1.50    |1.66    |  1.54  |
|DeepFilterNet       |3.03    | 3.71   |3.29    |3.55    |4.20    |0.94    |15.71    |15.66    |**9.66**|**7.19**|3.47    |3.40    |4.00    |3.10    |28.01    |16.20    |**15.79**|**15.69**|**0.55**|**0.94**|  1.77  |

- Resemble_enhance ([Github](https://github.com/resemble-ai/resemble-enhance)) 是 Resemble-AI 自 2023 年以来推出的开源 44.1kHz 纯语音增强平台，我们在评估前将其重采样为 48khz。
- DeepFilterNet ([Github](https://github.com/Rikorose/DeepFilterNet)) 是一个低复杂度的全带音频 (48kHz) 语音增强框架，使用深度滤波。
> **注意：** 我们观察到在使用 48 kHz 模型处理后，两个语音指标 LLR 和 LSD 出现异常。我们将进一步调查该问题以确定原因。

**语音分离模型：**

我们在流行的基准测试集上评估了我们的语音分离模型 `MossFormer2_SS_16K`：LRS2_2Mix (16 kHz), WSJ0-2Mix (8 kHz), Libri2Mix (8 kHz), WHAM! (8 kHz)。我们将我们的模型与以下最先进的模型进行了比较：[Conv-TasNet](https://arxiv.org/abs/1809.07454), [DualPathRNN](https://arxiv.org/abs/1910.06379), [DPTNet](https://arxiv.org/abs/2007.13975), [SepFormer](https://arxiv.org/abs/2010.13154), [TDANet](https://openreview.net/pdf?id=fzberKYWKsI), [TF-GridNet](https://arxiv.org/abs/2209.03952), [SPMamba](https://arxiv.org/abs/2404.02063)。测试结果取自 [TDANet Github 仓库](https://github.com/JusperLee/TDANet) 和 [SPMamba GitHub 仓库](https://github.com/JusperLee/SPMamba)。评估使用了 [SI-SNRi](https://arxiv.org/abs/1811.02508) (SI-SNR 改善量) 性能指标。

|模型 |LRS2_2Mix (16 kHz)|WSJ0-2Mix (8 kHz)|Libri2Mix (8kHz)|WHAM! (8 kHz)|
|------|------------------|-----------------|----------------|-------------|
|Conv-TasNet |10.6|15.3|12.2|12.7|
|DualPathRNN|12.7|18.8|16.1|13.7|
|DPTNet     |13.3|20.2|16.7|14.9|     
|SepFormer  |13.5|20.4|17.0|14.4|
|TDANet Large|14.2|18.5|17.4|15.2|
|TF-GridNet   |-|**22.8**|19.8|16.9|
|SPMamba      |-|22.5|**19.9**|**17.4**|
|MossFormer2_SS_16K|**15.5**|22.0|16.7|**17.4**|

> **注意：** 展示的 MossFormer2_SS_16K 结果来自我们的统一模型，未经在各个数据集上重新训练。该 16 kHz 模型用于 16 kHz 测试集上的语音分离，然后在下采样到 8 kHz 的音频上计算分数。所有对比模型均在每个数据集上分别进行了训练和测试。

**语音超分辨率模型：**

我们使用 VoiceBank+DEMAND 48 kHz 测试集展示了我们的语音超分辨率模型 `MossFormer2_SR_48K` 的有效性。对于超分辨率评估，测试集被下采样到 16 kHz、24 kHz 和 32 kHz。评估使用了对数谱距离 (LSD) 和 PESQ 指标。考虑到语音质量受低采样率和背景噪声的双重影响，我们还结合了语音增强模型 `MossFormer2_SE_48K` 在超分辨率处理前降低噪声。结果如下表所示。

|模型 | 16 kHz | 24 kHz | 32 kHz | 48 kHz |PESQ|
|------|--------|--------|--------|--------|-----|
|原始|2.80    | 2.60   |  2.29  |1.46    |1.97|
|增强后|**1.93**  |**1.52**    |   **1.50** |**1.42**    |**3.15** |

对于 48 kHz 的情况，未应用语音超分辨率。最后两列显示 `MossFormer2_SE_48K` 显著提高了 16 kHz 的 PESQ 分数，但对 LSD 的改善微乎其微。因此，16 kHz、24 kHz 和 32 kHz 下的 LSD 改善主要归功于 `MossFormer2_SR_48K`。
