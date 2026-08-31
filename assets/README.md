# LightClear 共享样例资源

仓库根 `assets/` 是 explore 层共享输入素材池，不是产品运行时资源。`products/**/assets` 仍由各产品自有。

## 布局

```text
assets/
  audio/noisy/          # 带噪语音，供语音增强
  audio/mixture/        # 混合语音，供说话人分离
  audio/bandlimited/    # 带限语音，供超分辨率
  audio/music/          # 带伴奏混合，供人声隔离 / 四轨 / 六轨
  video/speech/         # 含人脸与语音的视频，供目标说话人提取
  manifests/            # 批处理 .scp 清单，路径为绝对路径
```

## 分类规则

- 第一级按介质：`audio/` / `video/`。
- 第二级按用途：`noisy`、`mixture`、`bandlimited`、`music`、`speech`。
- `mixture` 是双说话人；`music` 是人声加伴奏。不要混放。
- 同内容只保留一份。产品需要样例时拷进产品自己的 `assets/`，不要运行时读取本目录。

## 使用规则

- Demo / CLI 顶部显式写出绝对路径，例如 `/Users/boom/workspace/LightClear/assets/audio/noisy/input.wav`。
- 批量输入优先用 `manifests/*.scp`。ClearVoice 目录输入会处理该目录下全部文件；多于 scp 的 noisy / bandlimited / video 文件、以及没有 scp 的 music 文件，见「不进 scp 的额外样例」。
- 不要把本目录当作产品默认样例池去 `glob`。

## 批处理清单

| 文件 | 覆盖 |
| --- | --- |
| `manifests/se.scp` | `audio/noisy/` 下 `input.wav`、`input2.wav`（16kHz）、`p232_005.wav`、`p232_007.wav`（48kHz，48k 语音增强批次样本）、`speech1.{flac,mp3,ogg,wav}`、`speech2.{aac,aiff,wav}` |
| `manifests/ss.scp` | `audio/mixture/` 全部文件 |
| `manifests/sr.scp` | `audio/bandlimited/LJ001-0001.wav` … `LJ001-0010.wav` |
| `manifests/tse.scp` | `video/speech/001.avi`、`002.avi` |

## 不进 scp 的额外样例

ClearVoice 把用途目录交给目录输入 API 时，会处理同目录下全部文件，因此 `noisy/`、`bandlimited/`、`video/speech/` 里下列文件会和对应 scp 一起跑。`audio/music/` 没有 scp，只给 Demucs demo 写死路径。标了「仅供本机演示」的不要当公开素材。

| 文件 | 采样率 / 格式 | 用途 |
| --- | --- | --- |
| `audio/music/test.mp3` | 44.1kHz MP3 | 上游 Demucs 烟雾片段，人声能量很低 |
| `audio/music/next_station_heaven.mp3` | 44.1kHz MP3 | 带唱短片段（蔡卓妍《下一站天后》DJ 混音约 0:50 起 30 秒），仅供本机演示 |
| `audio/noisy/test.wav` | 48kHz WAV | `demo_mossformergan_se.py` 单文件输入。48k 增强批次已覆盖 `p232_*.wav` |
| `audio/noisy/speech2_short.wav` | 44.1kHz float WAV | 立体声短样本，目录模式额外文件 |
| `audio/noisy/aviation_glideslope_too_low.wav` | 16kHz WAV | 航空 GPWS「下滑道太低」告警带噪语音，特殊场景，仅供本机演示 |
| `audio/noisy/aviation_decision_height.wav` | 16kHz WAV | 航空 GPWS「决断高度」告警带噪语音，特殊场景，仅供本机演示 |
| `audio/bandlimited/input_sr.wav` | 16kHz WAV | `demo_sr.py` 单文件输入。超分批次已覆盖 LJ 十条 |
| `video/speech/003.avi` | 视频 | 目标说话人提取目录模式额外样本 |
