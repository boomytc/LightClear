from __future__ import annotations

from io import BytesIO
from pathlib import Path
import base64

import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


FONT_PATH = Path(__file__).resolve().parent / "assets" / "fonts" / "SimHei.ttf"

if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    font_prop = fm.FontProperties(fname=str(FONT_PATH))
    plt.rcParams["font.sans-serif"] = [font_prop.get_name()]
    plt.rcParams["axes.unicode_minus"] = False


def load_audio_native(file_path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
    return audio, int(sample_rate)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2)))


def _peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def _centroid(audio: np.ndarray, sample_rate: int) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)))


def figure_to_data_uri(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _plot_waveforms(series: list[tuple[str, np.ndarray, int, str]], max_seconds: float):
    fig = plt.figure(figsize=(12, 2.6 * len(series)))
    for index, (title, audio, sample_rate, color) in enumerate(series, start=1):
        max_samples = min(len(audio), max(1, int(sample_rate * max_seconds)))
        time_axis = np.linspace(0, max_samples / sample_rate, max_samples)
        axis = fig.add_subplot(len(series), 1, index)
        axis.plot(time_axis, audio[:max_samples], color=color, alpha=0.85, linewidth=0.6)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_ylabel("振幅")
        axis.grid(True, alpha=0.25)
        if index == len(series):
            axis.set_xlabel("时间 (秒)")
    fig.tight_layout()
    return fig


def _plot_power_spectrum(series: list[tuple[str, np.ndarray, int, str]]):
    fig = plt.figure(figsize=(12, 5))
    axis = fig.add_subplot(111)
    for title, audio, sample_rate, color in series:
        nperseg = min(1024, len(audio))
        freqs, psd = signal.welch(audio, sample_rate, nperseg=max(8, nperseg))
        axis.semilogx(freqs, 10 * np.log10(psd + 1e-10), color=color, alpha=0.85, label=f"{title} ({sample_rate} Hz)")
    axis.set_title("功率谱密度对比", fontsize=12, fontweight="bold")
    axis.set_xlabel("频率 (Hz)")
    axis.set_ylabel("功率谱密度 (dB/Hz)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    return fig


def _plot_mel_spectrum(series: list[tuple[str, np.ndarray, int]]):
    fig = plt.figure(figsize=(12, 3.4 * len(series)))
    for index, (title, audio, sample_rate) in enumerate(series, start=1):
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        axis = fig.add_subplot(len(series), 1, index)
        image = axis.imshow(
            mel_db,
            aspect="auto",
            origin="lower",
            extent=[0, len(audio) / sample_rate, 0, sample_rate / 2],
        )
        axis.set_title(f"{title} Mel 频谱图", fontsize=12, fontweight="bold")
        axis.set_ylabel("频率 (Hz)")
        if index == len(series):
            axis.set_xlabel("时间 (秒)")
        fig.colorbar(image, ax=axis, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def _enhance_rows(input_audio: np.ndarray, output_audio: np.ndarray, sample_rate: int) -> list[dict[str, str]]:
    min_len = min(len(input_audio), len(output_audio))
    original = input_audio[:min_len]
    enhanced = output_audio[:min_len]
    residual = original - enhanced
    residual_power = float(np.mean(residual**2))
    signal_power = float(np.mean(enhanced**2))
    residual_ratio_db = 0.0 if residual_power == 0 else float(10 * np.log10(signal_power / residual_power))
    input_rms = _rms(original)
    output_rms = _rms(enhanced)
    rms_change = ((output_rms / (input_rms + 1e-10)) - 1) * 100 if input_rms > 0 else 0.0
    return [
        {
            "指标类别": "采样率 (Hz)",
            "输入": str(sample_rate),
            "输出": str(sample_rate),
            "说明": "增强不改变采样率合同",
        },
        {
            "指标类别": "时长 (秒)",
            "输入": f"{min_len / sample_rate:.2f}",
            "输出": f"{min_len / sample_rate:.2f}",
            "说明": "按时长截齐后对比",
        },
        {
            "指标类别": "RMS",
            "输入": f"{input_rms:.4f}",
            "输出": f"{output_rms:.4f}",
            "说明": f"能量变化 {rms_change:+.1f}%",
        },
        {
            "指标类别": "峰值",
            "输入": f"{_peak(original):.4f}",
            "输出": f"{_peak(enhanced):.4f}",
            "说明": "去噪后峰值可能下降",
        },
        {
            "指标类别": "残差比 (dB)",
            "输入": "-",
            "输出": f"{residual_ratio_db:.2f}",
            "说明": "输入减输出的残差相对输出能量，数值越大差异越明显",
        },
        {
            "指标类别": "谱质心 (Hz)",
            "输入": f"{_centroid(original, sample_rate):.1f}",
            "输出": f"{_centroid(enhanced, sample_rate):.1f}",
            "说明": "仅描述频谱重心，不当作带宽扩展",
        },
    ]


def _separate_rows(
    mix: np.ndarray,
    speaker_1: np.ndarray,
    speaker_2: np.ndarray,
    sample_rate: int,
) -> list[dict[str, str]]:
    duration = min(len(mix), len(speaker_1), len(speaker_2)) / sample_rate
    mix_rms = _rms(mix)
    s1_rms = _rms(speaker_1)
    s2_rms = _rms(speaker_2)
    return [
        {
            "指标类别": "采样率 (Hz)",
            "混合": str(sample_rate),
            "说话人 1": str(sample_rate),
            "说话人 2": str(sample_rate),
            "说明": "分离不改变采样率合同",
        },
        {
            "指标类别": "时长 (秒)",
            "混合": f"{duration:.2f}",
            "说话人 1": f"{len(speaker_1) / sample_rate:.2f}",
            "说话人 2": f"{len(speaker_2) / sample_rate:.2f}",
            "说明": "两路与混合对齐播放",
        },
        {
            "指标类别": "RMS",
            "混合": f"{mix_rms:.4f}",
            "说话人 1": f"{s1_rms:.4f}",
            "说话人 2": f"{s2_rms:.4f}",
            "说明": "两路能量相对混合音，不是去噪指标",
        },
        {
            "指标类别": "峰值",
            "混合": f"{_peak(mix):.4f}",
            "说话人 1": f"{_peak(speaker_1):.4f}",
            "说话人 2": f"{_peak(speaker_2):.4f}",
            "说明": "各路峰值独立计算",
        },
        {
            "指标类别": "谱质心 (Hz)",
            "混合": f"{_centroid(mix, sample_rate):.1f}",
            "说话人 1": f"{_centroid(speaker_1, sample_rate):.1f}",
            "说话人 2": f"{_centroid(speaker_2, sample_rate):.1f}",
            "说明": "描述各路频谱重心",
        },
    ]


def _super_resolve_rows(
    input_audio: np.ndarray,
    output_audio: np.ndarray,
    input_sr: int,
    output_sr: int,
) -> list[dict[str, str]]:
    if output_sr > input_sr:
        sr_note = f"输出 {output_sr} Hz 高于输入 {input_sr} Hz，带宽按超分合同扩展"
    elif output_sr == input_sr:
        sr_note = "输入与输出采样率相同，未观察到采样率提升"
    else:
        sr_note = f"输出 {output_sr} Hz 低于输入 {input_sr} Hz"
    return [
        {
            "指标类别": "采样率 (Hz)",
            "输入": str(input_sr),
            "输出": str(output_sr),
            "说明": sr_note,
        },
        {
            "指标类别": "时长 (秒)",
            "输入": f"{len(input_audio) / input_sr:.2f}",
            "输出": f"{len(output_audio) / output_sr:.2f}",
            "说明": "按各自原生采样率计算",
        },
        {
            "指标类别": "RMS",
            "输入": f"{_rms(input_audio):.4f}",
            "输出": f"{_rms(output_audio):.4f}",
            "说明": "原生波形能量，未为对齐而重采样",
        },
        {
            "指标类别": "峰值",
            "输入": f"{_peak(input_audio):.4f}",
            "输出": f"{_peak(output_audio):.4f}",
            "说明": "原生峰值",
        },
        {
            "指标类别": "谱质心 (Hz)",
            "输入": f"{_centroid(input_audio, input_sr):.1f}",
            "输出": f"{_centroid(output_audio, output_sr):.1f}",
            "说明": "各自采样率下的频谱重心",
        },
        {
            "指标类别": "Nyquist (Hz)",
            "输入": str(input_sr // 2),
            "输出": str(output_sr // 2),
            "说明": "可表示的最高频率随采样率变化",
        },
    ]


def build_analysis_payload(
    tool_id: str,
    input_path: Path,
    output_paths: dict[str, Path],
    max_seconds: float,
) -> dict[str, object]:
    input_audio, input_sr = load_audio_native(input_path)

    if tool_id == "separate":
        speaker_1, speaker_1_sr = load_audio_native(output_paths["speaker-1"])
        speaker_2, speaker_2_sr = load_audio_native(output_paths["speaker-2"])
        series = [
            ("混合音频", input_audio, input_sr, "#2563eb"),
            ("说话人 1", speaker_1, speaker_1_sr, "#0f766e"),
            ("说话人 2", speaker_2, speaker_2_sr, "#7c3aed"),
        ]
        return {
            "tool": tool_id,
            "input_sample_rate": input_sr,
            "output_sample_rate": speaker_1_sr,
            "duration_seconds": len(input_audio) / input_sr,
            "metrics_rows": _separate_rows(input_audio, speaker_1, speaker_2, input_sr),
            "waveform_image": figure_to_data_uri(_plot_waveforms(series, max_seconds)),
            "power_spectrum_image": figure_to_data_uri(_plot_power_spectrum(series)),
            "mel_spectrum_image": figure_to_data_uri(
                _plot_mel_spectrum([(title, audio, sr) for title, audio, sr, _ in series])
            ),
        }

    output_id = next(iter(output_paths))
    output_audio, output_sr = load_audio_native(output_paths[output_id])
    if tool_id == "super_resolve":
        output_title = "超分音频"
        rows = _super_resolve_rows(input_audio, output_audio, input_sr, output_sr)
    else:
        output_title = "增强音频"
        rows = _enhance_rows(input_audio, output_audio, input_sr)
    series = [
        ("输入音频", input_audio, input_sr, "#2563eb"),
        (output_title, output_audio, output_sr, "#0f766e"),
    ]
    return {
        "tool": tool_id,
        "input_sample_rate": input_sr,
        "output_sample_rate": output_sr,
        "duration_seconds": len(input_audio) / input_sr,
        "metrics_rows": rows,
        "waveform_image": figure_to_data_uri(_plot_waveforms(series, max_seconds)),
        "power_spectrum_image": figure_to_data_uri(_plot_power_spectrum(series)),
        "mel_spectrum_image": figure_to_data_uri(
            _plot_mel_spectrum([(title, audio, sr) for title, audio, sr, _ in series])
        ),
    }
