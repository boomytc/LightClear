# LightClear 共享样例资源

仓库根 `assets/` 是 explore 层共享输入素材池，不是产品运行时资源。`products/**/assets` 仍由各产品自有。

## 布局

```text
assets/
  audio/noisy/          # 带噪语音，供语音增强
  audio/mixture/        # 混合语音，供说话人分离
  audio/bandlimited/    # 带限语音，供超分辨率
  audio/music/          # 带伴奏混合，供人声隔离 / 分轨
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
- 批量输入优先用 `manifests/*.scp` 列出的路径。把用途目录交给 ClearVoice 目录输入 API 时，目录内文件可能比对应 scp 更多（例如 `audio/noisy/` 含 `test.wav`，`video/speech/` 含 `003.avi`）。
- 不要把本目录当作产品默认样例池去 `glob`。

## 清单

| 文件 | 用途 |
| --- | --- |
| `manifests/se.scp` | 语音增强批处理 |
| `manifests/ss.scp` | 说话人分离批处理 |
| `manifests/sr.scp` | 语音超分辨率批处理 |
| `manifests/tse.scp` | 目标说话人提取批处理 |
| `audio/music/test.mp3` | 上游 Demucs 烟雾片段，人声能量很低 |
| `audio/music/next_station_heaven.mp3` | 带唱短片段（蔡卓妍《下一站天后》DJ 混音约 0:50 起 30 秒），仅供本机演示 |
