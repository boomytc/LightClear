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


def load_audio(file_path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return audio, int(sr)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0


def _peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def _dynamic_range(audio: np.ndarray) -> float:
    if not audio.size or float(np.mean(np.abs(audio))) <= 0:
        return 0.0
    return float(20 * np.log10(np.max(np.abs(audio)) / np.mean(np.abs(audio))))


def _zcr(audio: np.ndarray) -> float:
    if not audio.size:
        return 0.0
    return float(np.mean(librosa.feature.zero_crossing_rate(audio)))


def _centroid(audio: np.ndarray, sample_rate: int) -> float:
    if not audio.size:
        return 0.0
    return float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)))


def _change_percent(value: float, baseline: float) -> float:
    return ((value / (baseline + 1e-10)) - 1) * 100 if baseline > 0 else 0.0


def build_metrics_rows(
    input_audio: np.ndarray,
    output_audio: np.ndarray,
    input_sample_rate: int,
    output_sample_rate: int,
) -> list[dict[str, str]]:
    """对比原音频与超分音频，两路都按各自原生采样率统计。

    超分模型输出采样率通常高于输入；先按输入采样率重采样再比较会把带宽扩展
    这个核心变化抹掉，所以这里不做对齐重采样。
    """
    rate_delta = output_sample_rate - input_sample_rate
    if rate_delta > 0:
        rate_note = f"带宽按超分合同从 {input_sample_rate} Hz 扩到 {output_sample_rate} Hz"
        rate_change = f"{rate_delta:+d} Hz"
    elif rate_delta == 0:
        rate_note = "输入与输出采样率相同，未观察到采样率提升"
        rate_change = "无变化"
    else:
        rate_note = f"输出 {output_sample_rate} Hz 低于输入 {input_sample_rate} Hz"
        rate_change = f"{rate_delta:+d} Hz"

    input_rms = _rms(input_audio)
    output_rms = _rms(output_audio)
    rms_change = _change_percent(output_rms, input_rms)
    rms_desc = (
        "能量适度降低，高频重建未抬整体电平"
        if rms_change < -30
        else "能量基本保持，重建集中在高频细节"
        if rms_change < 0
        else "能量略有增加，高频细节被重建"
    )

    input_peak = _peak(input_audio)
    output_peak = _peak(output_audio)
    peak_change = _change_percent(output_peak, input_peak)
    peak_desc = (
        "峰值显著降低，重建信号更平滑"
        if peak_change < -30
        else "峰值略有降低，重建信号更平稳"
        if peak_change < 0
        else "峰值略有增加，重建带进新的高频分量"
    )

    input_dynamic = _dynamic_range(input_audio)
    output_dynamic = _dynamic_range(output_audio)
    dynamic_delta = output_dynamic - input_dynamic
    dynamic_desc = (
        "动态范围扩大，细节层次增加"
        if dynamic_delta > 2
        else "动态范围略有扩大"
        if dynamic_delta > 0
        else "动态范围基本保持"
    )

    input_centroid = _centroid(input_audio, input_sample_rate)
    output_centroid = _centroid(output_audio, output_sample_rate)
    centroid_delta = output_centroid - input_centroid
    centroid_desc = (
        "谱质心上移，高频成分增加"
        if centroid_delta > 100
        else "谱质心略有上移，高频成分增加"
        if centroid_delta > 0
        else "谱质心基本保持"
    )

    input_zcr = _zcr(input_audio)
    output_zcr = _zcr(output_audio)
    zcr_change = _change_percent(output_zcr, input_zcr)
    zcr_desc = (
        "零交叉率明显变化，过零密度随带宽改变"
        if abs(zcr_change) > 30
        else "零交叉率基本保持"
    )

    return [
        {
            "指标类别": "采样率 (Hz)",
            "原始音频": str(input_sample_rate),
            "超分音频": str(output_sample_rate),
            "变化量": rate_change,
            "变化说明": rate_note,
        },
        {
            "指标类别": "奈奎斯特频率 (Hz)",
            "原始音频": str(input_sample_rate // 2),
            "超分音频": str(output_sample_rate // 2),
            "变化量": f"{output_sample_rate // 2 - input_sample_rate // 2:+d} Hz",
            "变化说明": "可表示的最高频率随采样率变化",
        },
        {
            "指标类别": "音频时长 (秒)",
            "原始音频": f"{len(input_audio) / input_sample_rate:.2f}",
            "超分音频": f"{len(output_audio) / output_sample_rate:.2f}",
            "变化量": "无变化",
            "变化说明": "按各自原生采样率计算",
        },
        {
            "指标类别": "RMS 能量",
            "原始音频": f"{input_rms:.4f}",
            "超分音频": f"{output_rms:.4f}",
            "变化量": f"{rms_change:+.2f}%",
            "变化说明": rms_desc,
        },
        {
            "指标类别": "峰值",
            "原始音频": f"{input_peak:.4f}",
            "超分音频": f"{output_peak:.4f}",
            "变化量": f"{peak_change:+.2f}%",
            "变化说明": peak_desc,
        },
        {
            "指标类别": "动态范围 (dB)",
            "原始音频": f"{input_dynamic:.2f}",
            "超分音频": f"{output_dynamic:.2f}",
            "变化量": f"{dynamic_delta:+.2f} dB",
            "变化说明": dynamic_desc,
        },
        {
            "指标类别": "谱质心 (Hz)",
            "原始音频": f"{input_centroid:.1f}",
            "超分音频": f"{output_centroid:.1f}",
            "变化量": f"{centroid_delta:+.1f} Hz",
            "变化说明": centroid_desc,
        },
        {
            "指标类别": "零交叉率",
            "原始音频": f"{input_zcr:.4f}",
            "超分音频": f"{output_zcr:.4f}",
            "变化量": f"{zcr_change:+.2f}%",
            "变化说明": zcr_desc,
        },
    ]


def make_waveform_figure(series: list[tuple[str, np.ndarray, int, str]], max_seconds: float):
    fig = plt.figure(figsize=(12, 2.8 * len(series)))
    for index, (title, audio, sample_rate, color) in enumerate(series, start=1):
        max_samples = min(len(audio), max(1, int(sample_rate * max_seconds)))
        time_axis = np.linspace(0, max_samples / sample_rate, max_samples)
        axis = fig.add_subplot(len(series), 1, index)
        axis.plot(time_axis, audio[:max_samples], color=color, alpha=0.85, linewidth=0.6)
        axis.set_title(f"{title}波形（{sample_rate} Hz）", fontsize=12, fontweight="bold")
        axis.set_ylabel("振幅")
        axis.grid(True, alpha=0.25)
        if index == len(series):
            axis.set_xlabel("时间 (秒)")
    fig.tight_layout()
    return fig


def make_power_spectrum_figure(series: list[tuple[str, np.ndarray, int, str]]):
    fig = plt.figure(figsize=(12, 5))
    axis = fig.add_subplot(111)
    for title, audio, sample_rate, color in series:
        if not audio.size:
            continue
        nperseg = max(8, min(1024, audio.size))
        freqs, psd = signal.welch(audio, sample_rate, nperseg=nperseg)
        axis.semilogx(
            freqs,
            10 * np.log10(psd + 1e-10),
            color=color,
            alpha=0.85,
            label=f"{title}（{sample_rate} Hz）",
        )
    axis.set_title("功率谱密度对比", fontsize=12, fontweight="bold")
    axis.set_xlabel("频率 (Hz)")
    axis.set_ylabel("功率谱密度 (dB/Hz)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    return fig


def make_mel_spectrum_figure(series: list[tuple[str, np.ndarray, int, str]]):
    fig = plt.figure(figsize=(12, 3.6 * len(series)))
    for index, (title, audio, sample_rate, _color) in enumerate(series, start=1):
        axis = fig.add_subplot(len(series), 1, index)
        if not audio.size:
            axis.set_title(f"{title} Mel 频谱图（空）", fontsize=12, fontweight="bold")
            continue
        mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sample_rate), ref=np.max)
        image = axis.imshow(
            mel_db,
            aspect="auto",
            origin="lower",
            extent=[0, len(audio) / sample_rate, 0, sample_rate / 2],
        )
        axis.set_title(f"{title} Mel 频谱图（{sample_rate} Hz）", fontsize=12, fontweight="bold")
        axis.set_ylabel("频率 (Hz)")
        if index == len(series):
            axis.set_xlabel("时间 (秒)")
        fig.colorbar(image, ax=axis, format="%+2.0f dB")
    fig.tight_layout()
    return fig


def figure_to_data_uri(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_analysis_payload(
    original_path: Path,
    enhanced_path: Path,
    max_seconds: float,
) -> dict[str, object]:
    original_audio, input_sample_rate = load_audio(original_path)
    enhanced_audio, output_sample_rate = load_audio(enhanced_path)

    series = [
        ("原始音频", original_audio, input_sample_rate, "#2563eb"),
        ("超分音频", enhanced_audio, output_sample_rate, "#0f766e"),
    ]
    metrics_rows = build_metrics_rows(
        original_audio,
        enhanced_audio,
        input_sample_rate,
        output_sample_rate,
    )

    return {
        "sample_rate": input_sample_rate,
        "input_sample_rate": input_sample_rate,
        "output_sample_rate": output_sample_rate,
        "duration_seconds": len(original_audio) / input_sample_rate,
        "metrics_rows": metrics_rows,
        "waveform_image": figure_to_data_uri(make_waveform_figure(series, max_seconds)),
        "power_spectrum_image": figure_to_data_uri(make_power_spectrum_figure(series)),
        "mel_spectrum_image": figure_to_data_uri(make_mel_spectrum_figure(series)),
    }
