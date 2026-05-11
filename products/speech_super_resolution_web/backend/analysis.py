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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FONT_PATH = Path(__file__).resolve().parent / "assets" / "fonts" / "SimHei.ttf"

if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    font_prop = fm.FontProperties(fname=str(FONT_PATH))
    plt.rcParams["font.sans-serif"] = [font_prop.get_name()]
    plt.rcParams["axes.unicode_minus"] = False


def load_audio(file_path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return audio, sr


def calculate_audio_metrics(
    original_audio: np.ndarray,
    enhanced_audio: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    min_len = min(len(original_audio), len(enhanced_audio))
    original = original_audio[:min_len]
    enhanced = enhanced_audio[:min_len]
    noise = original - enhanced

    original_rms = float(np.sqrt(np.mean(original**2)))
    enhanced_rms = float(np.sqrt(np.mean(enhanced**2)))
    original_peak = float(np.max(np.abs(original)))
    enhanced_peak = float(np.max(np.abs(enhanced)))
    noise_power = float(np.mean(noise**2))
    signal_power = float(np.mean(enhanced**2))
    snr_improvement = 0.0 if noise_power == 0 else float(10 * np.log10(signal_power / noise_power))

    return {
        "duration": min_len / sample_rate,
        "original_rms": original_rms,
        "enhanced_rms": enhanced_rms,
        "original_peak": original_peak,
        "enhanced_peak": enhanced_peak,
        "original_dynamic_range": float(
            20 * np.log10(np.max(np.abs(original)) / (np.mean(np.abs(original)) + 1e-10))
        ),
        "enhanced_dynamic_range": float(
            20 * np.log10(np.max(np.abs(enhanced)) / (np.mean(np.abs(enhanced)) + 1e-10))
        ),
        "snr_improvement": snr_improvement,
        "original_zcr": float(np.mean(librosa.feature.zero_crossing_rate(original))),
        "enhanced_zcr": float(np.mean(librosa.feature.zero_crossing_rate(enhanced))),
        "original_spectral_centroid": float(
            np.mean(librosa.feature.spectral_centroid(y=original, sr=sample_rate))
        ),
        "enhanced_spectral_centroid": float(
            np.mean(librosa.feature.spectral_centroid(y=enhanced, sr=sample_rate))
        ),
    }


def build_metrics_rows(
    original_audio: np.ndarray,
    enhanced_audio: np.ndarray,
    sample_rate: int,
) -> list[dict[str, str]]:
    metrics = calculate_audio_metrics(original_audio, enhanced_audio, sample_rate)

    original_rms = metrics["original_rms"]
    enhanced_rms = metrics["enhanced_rms"]
    energy_change = ((enhanced_rms / (original_rms + 1e-10)) - 1) * 100 if original_rms > 0 else 0
    energy_desc = (
        "能量适度降低，噪声抑制良好"
        if energy_change < -30
        else "能量略有降低，噪声得到控制"
        if energy_change < 0
        else "能量增加，可能增强了信号强度"
    )

    original_peak = metrics["original_peak"]
    enhanced_peak = metrics["enhanced_peak"]
    peak_change = ((enhanced_peak / (original_peak + 1e-10)) - 1) * 100 if original_peak > 0 else 0
    peak_desc = (
        "峰值显著降低，噪声峰值被有效抑制"
        if peak_change < -30
        else "峰值适度降低，信号更加平滑"
        if peak_change < 0
        else "峰值增加，信号得到增强"
    )

    original_dr = metrics["original_dynamic_range"]
    enhanced_dr = metrics["enhanced_dynamic_range"]
    dr_improvement = enhanced_dr - original_dr
    dr_desc = (
        "动态范围显著改善，音频层次更丰富"
        if dr_improvement > 2
        else "动态范围略有改善，音质有所提升"
        if dr_improvement > 0
        else "动态范围基本保持，处理较为保守"
    )

    snr_improvement = metrics["snr_improvement"]
    snr_desc = (
        "信噪比显著改善，噪声抑制效果优秀"
        if snr_improvement > 5
        else "信噪比适度改善，噪声得到良好控制"
        if snr_improvement > 0
        else "信噪比略有下降，可能过度处理"
        if snr_improvement < -2
        else "信噪比基本保持"
    )

    original_centroid = metrics["original_spectral_centroid"]
    enhanced_centroid = metrics["enhanced_spectral_centroid"]
    centroid_change = enhanced_centroid - original_centroid
    centroid_desc = (
        "频谱质心上移，高频成分增强，清晰度提升"
        if centroid_change > 100
        else "频谱质心略有上移，音质略有改善"
        if centroid_change > 0
        else "频谱质心下移，高频成分减少，可能过度滤波"
    )

    original_zcr = metrics["original_zcr"]
    enhanced_zcr = metrics["enhanced_zcr"]
    zcr_change = ((enhanced_zcr / (original_zcr + 1e-10)) - 1) * 100 if original_zcr > 0 else 0
    zcr_desc = (
        "零交叉率显著降低，噪声和毛刺明显减少"
        if zcr_change < -30
        else "零交叉率适度降低，信号更加平滑"
        if zcr_change < 0
        else "零交叉率基本保持，信号特征稳定"
    )

    return [
        {
            "指标类别": "音频时长 (秒)",
            "原始音频": f"{metrics['duration']:.2f}",
            "超分音频": f"{metrics['duration']:.2f}",
            "变化量": "无变化",
            "变化说明": "时长保持不变",
        },
        {
            "指标类别": "采样率 (Hz)",
            "原始音频": str(sample_rate),
            "超分音频": str(sample_rate),
            "变化量": "无变化",
            "变化说明": "采样率保持不变",
        },
        {
            "指标类别": "RMS 能量",
            "原始音频": f"{original_rms:.4f}",
            "超分音频": f"{enhanced_rms:.4f}",
            "变化量": f"{energy_change:+.2f}%",
            "变化说明": energy_desc,
        },
        {
            "指标类别": "峰值",
            "原始音频": f"{original_peak:.4f}",
            "超分音频": f"{enhanced_peak:.4f}",
            "变化量": f"{peak_change:+.2f}%",
            "变化说明": peak_desc,
        },
        {
            "指标类别": "动态范围 (dB)",
            "原始音频": f"{original_dr:.2f}",
            "超分音频": f"{enhanced_dr:.2f}",
            "变化量": f"{dr_improvement:+.2f} dB",
            "变化说明": dr_desc,
        },
        {
            "指标类别": "信噪比改善 (dB)",
            "原始音频": "-",
            "超分音频": f"{snr_improvement:.2f}",
            "变化量": f"{snr_improvement:+.2f} dB",
            "变化说明": snr_desc,
        },
        {
            "指标类别": "谱质心 (Hz)",
            "原始音频": f"{original_centroid:.2f}",
            "超分音频": f"{enhanced_centroid:.2f}",
            "变化量": f"{centroid_change:+.2f} Hz",
            "变化说明": centroid_desc,
        },
        {
            "指标类别": "零交叉率",
            "原始音频": f"{original_zcr:.4f}",
            "超分音频": f"{enhanced_zcr:.4f}",
            "变化量": f"{zcr_change:+.2f}%",
            "变化说明": zcr_desc,
        },
    ]


def make_waveform_figure(
    original_audio: np.ndarray,
    enhanced_audio: np.ndarray,
    sample_rate: int,
    *,
    max_seconds: float,
):
    max_samples = min(
        len(original_audio),
        len(enhanced_audio),
        max(1, int(sample_rate * max_seconds)),
    )
    time_axis = np.linspace(0, max_samples / sample_rate, max_samples)
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(time_axis, original_audio[:max_samples], color="#2563eb", alpha=0.8, linewidth=0.6)
    ax1.set_title("原始音频波形", fontsize=12, fontweight="bold")
    ax1.set_ylabel("振幅")
    ax1.grid(True, alpha=0.25)

    ax2.plot(time_axis, enhanced_audio[:max_samples], color="#0f766e", alpha=0.85, linewidth=0.6)
    ax2.set_title("超分音频波形", fontsize=12, fontweight="bold")
    ax2.set_xlabel("时间 (秒)")
    ax2.set_ylabel("振幅")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def make_power_spectrum_figure(
    original_audio: np.ndarray,
    enhanced_audio: np.ndarray,
    sample_rate: int,
):
    nperseg = min(1024, len(original_audio), len(enhanced_audio))
    f1, psd1 = signal.welch(original_audio, sample_rate, nperseg=nperseg)
    f2, psd2 = signal.welch(enhanced_audio, sample_rate, nperseg=nperseg)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.semilogx(f1, 10 * np.log10(psd1 + 1e-10), color="#2563eb", alpha=0.8, label="原始音频")
    ax.semilogx(f2, 10 * np.log10(psd2 + 1e-10), color="#0f766e", alpha=0.85, label="超分音频")
    ax.set_title("功率谱密度对比", fontsize=12, fontweight="bold")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("功率谱密度 (dB/Hz)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def make_mel_spectrum_figure(
    original_audio: np.ndarray,
    enhanced_audio: np.ndarray,
    sample_rate: int,
):
    mel1 = librosa.feature.melspectrogram(y=original_audio, sr=sample_rate)
    mel2 = librosa.feature.melspectrogram(y=enhanced_audio, sr=sample_rate)
    mel1_db = librosa.power_to_db(mel1, ref=np.max)
    mel2_db = librosa.power_to_db(mel2, ref=np.max)

    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    im1 = ax1.imshow(
        mel1_db,
        aspect="auto",
        origin="lower",
        extent=[0, len(original_audio) / sample_rate, 0, sample_rate / 2],
    )
    ax1.set_title("原始音频 Mel 频谱图", fontsize=12, fontweight="bold")
    ax1.set_ylabel("频率 (Hz)")
    fig.colorbar(im1, ax=ax1, format="%+2.0f dB")

    im2 = ax2.imshow(
        mel2_db,
        aspect="auto",
        origin="lower",
        extent=[0, len(enhanced_audio) / sample_rate, 0, sample_rate / 2],
    )
    ax2.set_title("超分音频 Mel 频谱图", fontsize=12, fontweight="bold")
    ax2.set_xlabel("时间 (秒)")
    ax2.set_ylabel("频率 (Hz)")
    fig.colorbar(im2, ax=ax2, format="%+2.0f dB")

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
    original_audio, sample_rate = load_audio(original_path)
    enhanced_audio, _ = load_audio(enhanced_path, sample_rate=sample_rate)
    metrics_rows = build_metrics_rows(original_audio, enhanced_audio, sample_rate)

    return {
        "sample_rate": sample_rate,
        "duration_seconds": len(original_audio) / sample_rate,
        "metrics_rows": metrics_rows,
        "waveform_image": figure_to_data_uri(
            make_waveform_figure(
                original_audio,
                enhanced_audio,
                sample_rate,
                max_seconds=max_seconds,
            )
        ),
        "power_spectrum_image": figure_to_data_uri(
            make_power_spectrum_figure(original_audio, enhanced_audio, sample_rate)
        ),
        "mel_spectrum_image": figure_to_data_uri(
            make_mel_spectrum_figure(original_audio, enhanced_audio, sample_rate)
        ),
    }
