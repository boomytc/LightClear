from __future__ import annotations

from pathlib import Path
import time

import streamlit as st

from audio_analysis import (
    build_metrics_rows,
    load_audio,
    make_mel_spectrum_figure,
    make_power_spectrum_figure,
    make_waveform_figure,
)
from runtime import (
    AUDIO_EXTENSIONS,
    EnhancementResult,
    audio_mime_type,
    enhance_audio_file,
    list_sample_audio,
    load_mossformer2_se,
    make_output_path,
    model_checkpoint_dir,
    model_is_available,
    resolve_project_output_dir,
    write_uploaded_audio,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "mossformer2_se_streamlit_webui"
UPLOAD_DIR = APP_OUTPUT_ROOT / "uploads"
DEFAULT_OUTPUT_DIR = "outputs/mossformer2_se_streamlit_webui/enhanced"


@st.cache_resource(show_spinner=False)
def cached_mossformer2_se():
    return load_mossformer2_se(PROJECT_ROOT)


def append_log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")


def reset_result() -> None:
    st.session_state.result = None
    st.session_state.analysis_error = ""


def ensure_session_state() -> None:
    defaults = {
        "logs": [],
        "result": None,
        "analysis_error": "",
        "upload_digest": "",
        "upload_path": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #f7f9fb;
            color: #1f2933;
        }
        [data-testid="stHeader"] {
            background: rgba(247, 249, 251, 0.92);
        }
        section[data-testid="stSidebar"] {
            background: #eef3f5;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #1f2933 !important;
        }
        .block-container {
            padding-top: 1.75rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }
        .block-container h1,
        .block-container h2,
        .block-container h3,
        .block-container h4,
        .block-container p,
        .block-container label {
            color: #1f2933;
        }
        .lc-header {
            border-bottom: 1px solid #d9e2e7;
            padding-bottom: 1rem;
            margin-bottom: 1.2rem;
        }
        .lc-title {
            color: #1f2933;
            font-size: 2rem;
            font-weight: 750;
            margin: 0;
            letter-spacing: 0;
        }
        .lc-subtitle {
            color: #52616b;
            font-size: 0.98rem;
            margin-top: 0.35rem;
        }
        .lc-path {
            color: #334e68;
            background: #eef5f6;
            border: 1px solid #c9dde2;
            border-radius: 6px;
            padding: 0.55rem 0.7rem;
            overflow-wrap: anywhere;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.85rem;
        }
        div[data-testid="stMetric"] {
            background: #fbfcfd;
            border: 1px solid #dfe7ec;
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div,
        div[data-testid="stMetric"] p {
            color: #1f2933 !important;
        }
        div.stButton > button {
            border-radius: 6px;
            min-height: 2.6rem;
            font-weight: 650;
        }
        div.stButton > button[kind="primary"] {
            background: #0f766e;
            border-color: #0f766e;
            color: #ffffff;
        }
        div.stButton > button[kind="primary"] p,
        div.stButton > button[kind="primary"] span {
            color: #ffffff !important;
        }
        div.stButton > button p {
            color: inherit !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="lc-header">
          <div class="lc-title">LightClear MossFormer2 SE</div>
          <div class="lc-subtitle">Streamlit web UI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def selected_input_path(source: str) -> Path | None:
    if source == "示例文件":
        samples = list_sample_audio(PROJECT_ROOT)
        if not samples:
            st.sidebar.warning("未找到示例音频。")
            return None

        selected = st.sidebar.selectbox(
            "示例音频",
            options=samples,
            format_func=lambda path: path.name,
        )
        return selected

    uploaded_file = st.sidebar.file_uploader(
        "上传音频",
        type=list(AUDIO_EXTENSIONS),
        accept_multiple_files=False,
    )
    if uploaded_file is None:
        return None

    upload_path, digest = write_uploaded_audio(uploaded_file, UPLOAD_DIR, uploaded_file.name)
    if digest != st.session_state.upload_digest:
        st.session_state.upload_digest = digest
        st.session_state.upload_path = str(upload_path)
        reset_result()
        append_log(f"已上传音频: {upload_path.name}")
    return Path(st.session_state.upload_path)


def run_enhancement(input_path: Path, output_dir_text: str) -> EnhancementResult | None:
    total_start = time.perf_counter()
    try:
        output_dir = resolve_project_output_dir(PROJECT_ROOT, output_dir_text)
        output_path = make_output_path(output_dir, input_path)

        with st.spinner("模型准备中..."):
            model_ready_start = time.perf_counter()
            model_handle = cached_mossformer2_se()
            model_ready_seconds = time.perf_counter() - model_ready_start

        with st.spinner("正在处理音频..."):
            result = enhance_audio_file(
                model_handle=model_handle,
                input_path=input_path,
                output_path=output_path,
                model_ready_seconds=model_ready_seconds,
                total_start_time=total_start,
            )

        append_log(f"输出文件已保存: {result.output_path}")
        append_log(f"模型准备时间: {result.model_ready_seconds:.2f} 秒")
        append_log(f"音频处理时间: {result.process_seconds:.2f} 秒")
        append_log(f"总执行时间: {result.total_seconds:.2f} 秒")
        return result
    except Exception as exc:
        append_log(f"处理失败: {exc}")
        st.error(f"处理失败: {exc}")
        return None


def render_audio_panels(input_path: Path | None, result: EnhancementResult | None) -> None:
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("原始音频")
        if input_path is None:
            st.info("等待音频输入。")
        else:
            st.audio(str(input_path), format=audio_mime_type(input_path))
            st.markdown(f'<div class="lc-path">{input_path}</div>', unsafe_allow_html=True)

    with right:
        st.subheader("增强音频")
        if result is None:
            st.info("等待处理结果。")
            return

        st.audio(str(result.output_path), format=audio_mime_type(result.output_path))
        st.markdown(f'<div class="lc-path">{result.output_path}</div>', unsafe_allow_html=True)
        st.download_button(
            "下载增强音频",
            data=result.output_path.read_bytes(),
            file_name=result.output_path.name,
            mime=audio_mime_type(result.output_path),
            use_container_width=True,
        )


def render_timing(result: EnhancementResult | None) -> None:
    if result is None:
        return

    st.subheader("时间统计")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("模型首次加载", f"{result.model_initial_load_seconds:.2f} 秒")
    c2.metric("本次模型准备", f"{result.model_ready_seconds:.2f} 秒")
    c3.metric("音频处理", f"{result.process_seconds:.2f} 秒")
    c4.metric("总执行", f"{result.total_seconds:.2f} 秒")


def render_analysis(
    result: EnhancementResult | None,
    show_original: bool,
    show_enhanced: bool,
    max_seconds: float,
) -> None:
    if result is None:
        return

    st.subheader("对比分析")
    try:
        original_audio, sample_rate = load_audio(result.input_path)
        enhanced_audio, _ = load_audio(result.output_path, sample_rate=sample_rate)
    except Exception as exc:
        st.session_state.analysis_error = str(exc)
        st.warning(f"分析数据加载失败: {exc}")
        return

    waveform_tab, spectrum_tab, metrics_tab = st.tabs(["波形", "频谱", "指标"])

    with waveform_tab:
        if not show_original and not show_enhanced:
            st.warning("至少选择一个波形。")
        else:
            fig = make_waveform_figure(
                original_audio,
                enhanced_audio,
                sample_rate,
                show_original=show_original,
                show_enhanced=show_enhanced,
                max_seconds=max_seconds,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)

    with spectrum_tab:
        spectrum_type = st.radio(
            "频谱类型",
            ["功率谱密度", "Mel 频谱"],
            horizontal=True,
        )
        if spectrum_type == "功率谱密度":
            fig = make_power_spectrum_figure(original_audio, enhanced_audio, sample_rate)
        else:
            fig = make_mel_spectrum_figure(original_audio, enhanced_audio, sample_rate)
        st.pyplot(fig, clear_figure=True, use_container_width=True)

    with metrics_tab:
        st.dataframe(
            build_metrics_rows(original_audio, enhanced_audio, sample_rate),
            use_container_width=True,
            hide_index=True,
        )


def render_logs() -> None:
    st.subheader("处理日志")
    log_text = "\n".join(st.session_state.logs[-80:])
    st.code(log_text or "暂无日志", language="text")


def main() -> None:
    st.set_page_config(
        page_title="LightClear MossFormer2 SE",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_session_state()
    inject_styles()
    render_header()

    with st.sidebar:
        st.header("控制")
        if model_is_available(PROJECT_ROOT):
            st.success(f"模型目录: {model_checkpoint_dir(PROJECT_ROOT).relative_to(PROJECT_ROOT)}")
        else:
            st.warning(f"模型目录未就绪: {model_checkpoint_dir(PROJECT_ROOT).relative_to(PROJECT_ROOT)}")

        source = st.radio("音频来源", ["示例文件", "上传文件"])
        input_path = selected_input_path(source)
        output_dir_text = st.text_input("输出目录", DEFAULT_OUTPUT_DIR)
        process_disabled = input_path is None or not input_path.exists()
        if st.button("开始增强", type="primary", disabled=process_disabled, use_container_width=True):
            reset_result()
            append_log(f"开始处理: {input_path}")
            st.session_state.result = run_enhancement(input_path, output_dir_text)
        if st.button("清空结果", use_container_width=True):
            reset_result()
            append_log("已清空当前结果")
        st.divider()
        with st.expander("显示设置", expanded=True):
            show_original = st.checkbox("显示原始波形", value=True)
            show_enhanced = st.checkbox("显示增强波形", value=True)
            max_seconds = st.slider("波形窗口 (秒)", min_value=1.0, max_value=30.0, value=8.0, step=1.0)

    result = st.session_state.result
    render_audio_panels(input_path, result)
    render_timing(result)
    render_analysis(result, show_original, show_enhanced, max_seconds)
    render_logs()


if __name__ == "__main__":
    main()
